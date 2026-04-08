package graph

import (
	"context"
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/sharpner/ultramemory/llm"
	"github.com/sharpner/ultramemory/secureindex"
	"github.com/sharpner/ultramemory/store"
)

func TestExtractorProcessJobSecureModeStoresEncryptedArtifacts(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	db := openSecureExtractorDB(t)
	defer db.Close() //nolint:errcheck

	payload := IngestPayload{GroupID: "secure-group"}
	var err error
	payload.ContentCiphertext, err = secureindex.EncryptString("secret-key", "ingest-content", "private cats note with enough words to exceed fifty characters easily")
	if err != nil {
		t.Fatalf("encrypt content: %v", err)
	}
	payload.SourceCiphertext, err = secureindex.EncryptString("secret-key", "ingest-source", "/tmp/private-note.md")
	if err != nil {
		t.Fatalf("encrypt source: %v", err)
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}

	extractor := New(db, secureExtractorStub{}, fixedEmbedder{}, 0.92, 1, "secret-key")
	if err := extractor.ProcessJob(ctx, string(raw), 0); err != nil {
		t.Fatalf("process job: %v", err)
	}

	var episodeUUID string
	var episodeContent string
	var episodeSource string
	var embeddingBlob []byte
	err = db.SQL().QueryRowContext(ctx,
		`SELECT uuid, content, source, embedding FROM episodes WHERE group_id = ?`,
		"secure-group",
	).Scan(&episodeUUID, &episodeContent, &episodeSource, &embeddingBlob)
	if err != nil {
		t.Fatalf("select episode: %v", err)
	}
	if !secureindex.IsEncrypted(episodeContent) {
		t.Fatalf("episode content is not encrypted: %q", episodeContent)
	}
	if !secureindex.IsEncrypted(episodeSource) {
		t.Fatalf("episode source is not encrypted: %q", episodeSource)
	}
	if len(embeddingBlob) != 0 {
		t.Fatalf("expected no raw embedding blob in secure mode, got %d bytes", len(embeddingBlob))
	}

	content, err := secureindex.DecryptString("secret-key", "episode-content", episodeContent)
	if err != nil {
		t.Fatalf("decrypt episode content: %v", err)
	}
	if content != "private cats note with enough words to exceed fifty characters easily" {
		t.Fatalf("unexpected episode content: %q", content)
	}

	source, err := secureindex.DecryptString("secret-key", "episode-source", episodeSource)
	if err != nil {
		t.Fatalf("decrypt episode source: %v", err)
	}
	if source != "/tmp/private-note.md" {
		t.Fatalf("unexpected episode source: %q", source)
	}

	entities, err := db.CountEntities(ctx, "secure-group")
	if err != nil {
		t.Fatalf("count entities: %v", err)
	}
	if entities != 2 {
		t.Fatalf("entities = %d, want 2", entities)
	}

	edges, err := db.CountEdges(ctx, "secure-group")
	if err != nil {
		t.Fatalf("count edges: %v", err)
	}
	if edges != 1 {
		t.Fatalf("edges = %d, want 1", edges)
	}
	if encryptedEntityFTSCount(t, ctx, db) != 0 {
		t.Fatal("expected no entity FTS rows in secure mode")
	}
	if encryptedEdgeFTSCount(t, ctx, db) != 0 {
		t.Fatal("expected no edge FTS rows in secure mode")
	}

	secureRows, err := db.AllSecureEpisodes(ctx, "secure-group", "kpt-v1")
	if err != nil {
		t.Fatalf("all secure episodes: %v", err)
	}
	if len(secureRows) != 1 {
		t.Fatalf("secure rows = %d, want 1", len(secureRows))
	}
	if secureRows[0].EpisodeUUID != episodeUUID {
		t.Fatalf("secure row episode mismatch: got %q want %q", secureRows[0].EpisodeUUID, episodeUUID)
	}
	if len(secureRows[0].Public) == 0 {
		t.Fatal("expected stored secure public layer")
	}
}

type secureExtractorStub struct{}

func (secureExtractorStub) ExtractEntities(context.Context, string) (*llm.ExtractedEntities, error) {
	return &llm.ExtractedEntities{Entities: []llm.ExtractedEntity{
		{Name: "Katze", EntityType: "Concept", Description: "Eine Katze im Notiztext"},
		{Name: "Sofa", EntityType: "Place", Description: "Ein Sofa im Notiztext"},
	}}, nil
}

func (secureExtractorStub) ExtractEdges(context.Context, []llm.ExtractedEntity, string) (*llm.ExtractedEdges, error) {
	return &llm.ExtractedEdges{Edges: []llm.ExtractedEdge{
		{RelationType: "SLEEPS_ON", SourceEntityID: 0, TargetEntityID: 1, Fact: "Die Katze schläft auf dem Sofa"},
	}}, nil
}

type fixedEmbedder struct{}

func (fixedEmbedder) Embed(context.Context, string) ([]float32, error) {
	return []float32{0.6, 0.2, -0.1, 0.4}, nil
}

func (fixedEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, 0, len(texts))
	for range texts {
		vec, err := fixedEmbedder{}.Embed(ctx, "")
		if err != nil {
			return nil, err
		}
		out = append(out, vec)
	}
	return out, nil
}

func openSecureExtractorDB(t *testing.T) *store.DB {
	t.Helper()

	db, err := store.Open(filepath.Join(t.TempDir(), "extract-secure.db"))
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	return db
}

func encryptedEntityFTSCount(t *testing.T, ctx context.Context, db *store.DB) int {
	t.Helper()

	var n int
	if err := db.SQL().QueryRowContext(ctx, `SELECT count(*) FROM entities_fts`).Scan(&n); err != nil {
		t.Fatalf("count entities_fts: %v", err)
	}
	return n
}

func encryptedEdgeFTSCount(t *testing.T, ctx context.Context, db *store.DB) int {
	t.Helper()

	var n int
	if err := db.SQL().QueryRowContext(ctx, `SELECT count(*) FROM edges_fts`).Scan(&n); err != nil {
		t.Fatalf("count edges_fts: %v", err)
	}
	return n
}
