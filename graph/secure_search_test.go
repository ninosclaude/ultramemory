package graph

import (
	"context"
	"testing"

	"github.com/sharpner/ultramemory/secureindex"
	"github.com/sharpner/ultramemory/store"
)

func TestSecureSearch_ReusesHybridRoles(t *testing.T) {
	db := openExtractTestDB(t)
	ctx := context.Background()
	groupID := "secure-search"
	personalKey := "alpha-key"

	episodeVec := []float32{1, 0, 0, 0}
	method := secureindex.NewMethod(personalKey, len(episodeVec))
	state := method.EncodeDoc(episodeVec)

	content, err := secureindex.EncryptString(personalKey, "episode-content", "Alice discussed the Atlas project timeline in session seven.")
	if err != nil {
		t.Fatalf("encrypt episode content: %v", err)
	}
	source, err := secureindex.EncryptString(personalKey, "episode-source", "locomo/conv-26/session_7")
	if err != nil {
		t.Fatalf("encrypt episode source: %v", err)
	}
	if err := db.UpsertEpisode(ctx, store.Episode{
		UUID:    "ep1",
		Content: content,
		GroupID: groupID,
		Source:  source,
	}); err != nil {
		t.Fatalf("upsert secure episode shell: %v", err)
	}
	if err := db.ReplaceSecureEpisodeIndex(ctx, groupID, "kpt-v1", []store.SecureEpisodeIndexRow{{
		EpisodeUUID:  "ep1",
		GroupID:      groupID,
		Method:       "kpt-v1",
		Content:      content,
		Source:       source,
		Public:       state.Public,
		BaseWaveReal: state.BaseWaveReal,
		BaseWaveImag: state.BaseWaveImag,
		WaveReal:     state.WaveReal,
		WaveImag:     state.WaveImag,
		ModeWeight:   state.ModeWeight,
		ModeEnergy:   state.ModeEnergy,
	}}); err != nil {
		t.Fatalf("replace secure index: %v", err)
	}

	aliceName, err := secureindex.EncryptString(personalKey, "entity-name", "Alice")
	if err != nil {
		t.Fatalf("encrypt entity name: %v", err)
	}
	aliceDesc, err := secureindex.EncryptString(personalKey, "entity-description", "Engineer on Atlas")
	if err != nil {
		t.Fatalf("encrypt entity description: %v", err)
	}
	atlasName, err := secureindex.EncryptString(personalKey, "entity-name", "Atlas")
	if err != nil {
		t.Fatalf("encrypt entity name: %v", err)
	}
	atlasDesc, err := secureindex.EncryptString(personalKey, "entity-description", "Project codename")
	if err != nil {
		t.Fatalf("encrypt entity description: %v", err)
	}

	aliceUUID, err := db.UpsertEntity(ctx, store.Entity{
		UUID:        "alice",
		Name:        aliceName,
		LookupKey:   "lk-alice",
		EntityType:  "Person",
		GroupID:     groupID,
		Description: aliceDesc,
	})
	if err != nil {
		t.Fatalf("upsert alice: %v", err)
	}
	atlasUUID, err := db.UpsertEntity(ctx, store.Entity{
		UUID:        "atlas",
		Name:        atlasName,
		LookupKey:   "lk-atlas",
		EntityType:  "Project",
		GroupID:     groupID,
		Description: atlasDesc,
	})
	if err != nil {
		t.Fatalf("upsert atlas: %v", err)
	}
	if err := db.LinkEntityEpisode(ctx, aliceUUID, "ep1"); err != nil {
		t.Fatalf("link alice episode: %v", err)
	}
	if err := db.LinkEntityEpisode(ctx, atlasUUID, "ep1"); err != nil {
		t.Fatalf("link atlas episode: %v", err)
	}

	edgeName, err := secureindex.EncryptString(personalKey, "edge-name", "WORKS_ON")
	if err != nil {
		t.Fatalf("encrypt edge name: %v", err)
	}
	edgeFact, err := secureindex.EncryptString(personalKey, "edge-fact", "Alice works on Atlas and owns the timeline.")
	if err != nil {
		t.Fatalf("encrypt edge fact: %v", err)
	}
	if err := db.UpsertEdge(ctx, store.Edge{
		UUID:       "edge1",
		SourceUUID: aliceUUID,
		TargetUUID: atlasUUID,
		Name:       edgeName,
		LookupKey:  "lk-edge1",
		Fact:       edgeFact,
		GroupID:    groupID,
		Episodes:   `["ep1"]`,
	}); err != nil {
		t.Fatalf("upsert edge: %v", err)
	}

	if err := db.WriteCommunityIDs(ctx, groupID, map[int64][]string{
		0: {aliceUUID, atlasUUID},
	}); err != nil {
		t.Fatalf("write community ids: %v", err)
	}
	report, err := secureindex.EncryptString(personalKey, "community-report", "People: Alice. Key facts: Alice works on Atlas and owns the timeline.")
	if err != nil {
		t.Fatalf("encrypt report: %v", err)
	}
	if err := db.StoreCommunityReport(ctx, groupID, 0, report); err != nil {
		t.Fatalf("store report: %v", err)
	}

	results, err := SecureSearch(ctx, db, secureSearchEmbedder{vec: episodeVec}, personalKey, groupID, "kpt-v1", "Atlas timeline", 10)
	if err != nil {
		t.Fatalf("secure search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected secure results")
	}

	var hasCommunity bool
	var hasEdge bool
	var hasEpisode bool
	for _, result := range results {
		if result.Type == "community" && result.Body != "" {
			hasCommunity = true
		}
		if result.Type == "edge" && result.Source == "locomo/conv-26/session_7" {
			hasEdge = true
		}
		if result.Type == "episode" && result.Source == "locomo/conv-26/session_7" {
			hasEpisode = true
		}
		if result.Type == "entity" {
			t.Fatalf("entity leaked into results: %+v", result)
		}
	}
	if !hasCommunity {
		t.Fatalf("expected community report in results: %+v", results)
	}
	if !hasEdge {
		t.Fatalf("expected edge result with secure source in results: %+v", results)
	}
	if !hasEpisode {
		t.Fatalf("expected episode result in results: %+v", results)
	}

	var n int
	if err := db.SQL().QueryRowContext(ctx, `SELECT count(*) FROM entities_fts`).Scan(&n); err != nil {
		t.Fatalf("count entities_fts: %v", err)
	}
	if n != 0 {
		t.Fatalf("expected no entities_fts rows, got %d", n)
	}
	if err := db.SQL().QueryRowContext(ctx, `SELECT count(*) FROM edges_fts`).Scan(&n); err != nil {
		t.Fatalf("count edges_fts: %v", err)
	}
	if n != 0 {
		t.Fatalf("expected no edges_fts rows, got %d", n)
	}
}

type secureSearchEmbedder struct {
	vec []float32
}

func (e secureSearchEmbedder) Embed(context.Context, string) ([]float32, error) {
	return e.vec, nil
}

func (e secureSearchEmbedder) EmbedBatch(context.Context, []string) ([][]float32, error) {
	return nil, nil
}
