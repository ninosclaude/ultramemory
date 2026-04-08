package store

import (
	"context"
	"testing"

	"github.com/sharpner/ultramemory/secureindex"
)

func TestResolveSecureEpisodesMergesDuplicates(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	db := openTestDB(t)
	groupID := "secure-group"
	personalKey := "alpha-key"

	method := secureindex.NewMethod(personalKey, 4)
	mustInsertSecureEpisode(t, ctx, db, groupID, personalKey, method, "ep-1", "cats sleep on sofas", "cats.md", []float32{1, 0, 0, 0})
	mustInsertSecureEpisode(t, ctx, db, groupID, personalKey, method, "ep-2", "cats sleep on sofas", "cats-copy.md", []float32{1, 0, 0, 0})
	mustInsertSecureEpisode(t, ctx, db, groupID, personalKey, method, "ep-3", "rockets cross the sky", "space.md", []float32{0, 1, 0, 0})

	dryRun, err := db.ResolveSecureEpisodes(ctx, groupID, "kpt-v1", personalKey, 0.75, true)
	if err != nil {
		t.Fatalf("dry-run resolve secure episodes: %v", err)
	}
	if dryRun.ClustersFound != 1 {
		t.Fatalf("clusters found = %d, want 1", dryRun.ClustersFound)
	}
	if dryRun.EpisodesMerged != 1 {
		t.Fatalf("episodes merged = %d, want 1", dryRun.EpisodesMerged)
	}

	result, err := db.ResolveSecureEpisodes(ctx, groupID, "kpt-v1", personalKey, 0.75, false)
	if err != nil {
		t.Fatalf("resolve secure episodes: %v", err)
	}
	if result.EpisodesMerged != 1 {
		t.Fatalf("episodes merged = %d, want 1", result.EpisodesMerged)
	}

	episodes, err := db.CountEpisodes(ctx, groupID)
	if err != nil {
		t.Fatalf("count episodes: %v", err)
	}
	if episodes != 2 {
		t.Fatalf("episodes = %d, want 2", episodes)
	}

	secureRows, err := db.CountSecureEpisodes(ctx, groupID, "kpt-v1")
	if err != nil {
		t.Fatalf("count secure rows: %v", err)
	}
	if secureRows != 2 {
		t.Fatalf("secure rows = %d, want 2", secureRows)
	}
}

func TestComputeSecureCurvaturesBuildsEpisodeGraph(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	db := openTestDB(t)
	groupID := "secure-graph"
	personalKey := "alpha-key"

	method := secureindex.NewMethod(personalKey, 4)
	mustInsertSecureEpisode(t, ctx, db, groupID, personalKey, method, "ep-1", "cats chase light", "cats.md", []float32{1, 0, 0, 0})
	mustInsertSecureEpisode(t, ctx, db, groupID, personalKey, method, "ep-2", "cats sleep on rugs", "cats-2.md", []float32{0.95, 0.05, 0, 0})
	mustInsertSecureEpisode(t, ctx, db, groupID, personalKey, method, "ep-3", "rockets fly to mars", "space.md", []float32{0, 1, 0, 0})

	curvatures, stats, err := db.ComputeSecureCurvatures(ctx, groupID, "kpt-v1", personalKey)
	if err != nil {
		t.Fatalf("compute secure curvatures: %v", err)
	}
	if stats.TotalEdges == 0 {
		t.Fatal("expected at least one secure curvature edge")
	}
	if len(curvatures) != stats.TotalEdges {
		t.Fatalf("curvature count = %d, want %d", len(curvatures), stats.TotalEdges)
	}
	if curvatures[0].SourceName == "" {
		t.Fatal("expected source label on secure curvature edge")
	}
	if curvatures[0].TargetName == "" {
		t.Fatal("expected target label on secure curvature edge")
	}
}

func mustInsertSecureEpisode(t *testing.T, ctx context.Context, db *DB, groupID, personalKey string, method *secureindex.Method, episodeUUID, content, source string, embedding []float32) {
	t.Helper()

	contentCiphertext, err := secureindex.EncryptString(personalKey, "episode-content", content)
	if err != nil {
		t.Fatalf("encrypt content: %v", err)
	}
	sourceCiphertext, err := secureindex.EncryptString(personalKey, "episode-source", source)
	if err != nil {
		t.Fatalf("encrypt source: %v", err)
	}
	if err := db.UpsertEpisode(ctx, Episode{
		UUID:    episodeUUID,
		Content: contentCiphertext,
		GroupID: groupID,
		Source:  sourceCiphertext,
	}); err != nil {
		t.Fatalf("upsert episode: %v", err)
	}

	state := method.EncodeDoc(embedding)
	if err := db.UpsertSecureEpisodeIndexRow(ctx, SecureEpisodeIndexRow{
		EpisodeUUID:  episodeUUID,
		GroupID:      groupID,
		Method:       "kpt-v1",
		Public:       state.Public,
		BaseWaveReal: state.BaseWaveReal,
		BaseWaveImag: state.BaseWaveImag,
		WaveReal:     state.WaveReal,
		WaveImag:     state.WaveImag,
		ModeWeight:   state.ModeWeight,
		ModeEnergy:   state.ModeEnergy,
	}); err != nil {
		t.Fatalf("upsert secure row: %v", err)
	}
}
