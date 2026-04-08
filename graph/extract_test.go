package graph

import (
	"context"
	"os"
	"testing"

	"github.com/sharpner/ultramemory/store"
)

func TestProcess_EntityCreated(t *testing.T) {
	db := openExtractTestDB(t)
	ctx := context.Background()
	client := newMockClient(t,
		entityJSON("Jonathan Harker", "Count Dracula"),
		edgeJSON("TRAVELS_TO", "Harker travels to Transylvania"),
	)
	ext := New(db, client, client, 0.5, 1, "")

	if err := ext.Process(ctx, "Jonathan Harker travels to meet Count Dracula.", "test.txt", "g"); err != nil {
		t.Fatal(err)
	}
	ext.Wait()

	n, err := db.CountEntities(ctx, "g")
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Errorf("expected 2 entities after Process, got %d", n)
	}
}

func TestProcess_EdgeCreated(t *testing.T) {
	db := openExtractTestDB(t)
	ctx := context.Background()
	client := newMockClient(t,
		entityJSON("Jonathan Harker", "Count Dracula"),
		edgeJSON("TRAVELS_TO", "Harker travels to Transylvania"),
	)
	ext := New(db, client, client, 0.5, 1, "")

	if err := ext.Process(ctx, "Jonathan Harker travels to meet Count Dracula.", "test.txt", "g"); err != nil {
		t.Fatal(err)
	}
	ext.Wait()

	n, err := db.CountEdges(ctx, "g")
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Errorf("expected 1 edge after Process, got %d", n)
	}
}

func TestProcess_EpisodeLinked(t *testing.T) {
	db := openExtractTestDB(t)
	ctx := context.Background()
	client := newMockClient(t,
		entityJSON("Jonathan Harker"),
		`{"edges":[]}`,
	)
	ext := New(db, client, client, 0.5, 1, "")

	if err := ext.Process(ctx, "Jonathan Harker arrived at the castle.", "castle.txt", "g"); err != nil {
		t.Fatal(err)
	}
	ext.Wait()

	n, err := db.CountEpisodes(ctx, "g")
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Errorf("expected 1 episode after Process, got %d", n)
	}
}

func TestProcess_ExactNameDedup(t *testing.T) {
	db := openExtractTestDB(t)
	ctx := context.Background()

	client := newMockClient(t, entityJSON("Jonathan Harker"), `{"edges":[]}`)
	ext := New(db, client, client, 0.5, 1, "")

	if err := ext.Process(ctx, "Jonathan Harker arrived.", "a.txt", "g"); err != nil {
		t.Fatal(err)
	}
	ext.Wait()

	client2 := newMockClient(t, entityJSON("JONATHAN HARKER"), `{"edges":[]}`)
	ext2 := New(db, client2, client2, 0.5, 1, "")

	if err := ext2.Process(ctx, "JONATHAN HARKER left.", "b.txt", "g"); err != nil {
		t.Fatal(err)
	}
	ext2.Wait()

	n, _ := db.CountEntities(ctx, "g")
	if n != 1 {
		t.Errorf("exact-name dedup (case-insensitive): expected 1 entity, got %d", n)
	}
}

func TestProcess_SourcePopulated(t *testing.T) {
	db := openExtractTestDB(t)
	ctx := context.Background()
	client := newMockClient(t, entityJSON("Jonathan Harker"), `{"edges":[]}`)
	ext := New(db, client, client, 0.5, 1, "")

	if err := ext.Process(ctx, "Jonathan Harker arrived.", "dracula/chapter1.txt", "g"); err != nil {
		t.Fatal(err)
	}
	ext.Wait()

	// Retrieve entity UUID via FTS
	entities, err := db.SearchEntitiesFTS(ctx, "Jonathan", "g", 1)
	if err != nil || len(entities) == 0 {
		t.Fatalf("entity not found: err=%v, count=%d", err, len(entities))
	}
	src := db.FirstEntitySource(ctx, entities[0].UUID, "g")
	if src != "dracula/chapter1.txt" {
		t.Errorf("expected source %q, got %q", "dracula/chapter1.txt", src)
	}
}

func openExtractTestDB(t *testing.T) *store.DB {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "extract-test-*.db")
	if err != nil {
		t.Fatalf("tempfile: %v", err)
	}
	_ = f.Close()
	db, err := store.Open(f.Name())
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}

func newTestExtractor(db *store.DB, threshold float64) *Extractor {
	return &Extractor{db: db, sem: make(chan struct{}, 1), resolveThreshold: threshold}
}

func TestResolveOrCreate_SemanticDedup(t *testing.T) {
	db := openExtractTestDB(t)
	ctx := context.Background()
	ext := newTestExtractor(db, 0.80)

	// Seed "Jonathan Harker" with embedding [1,0,0,0].
	uuidA, err := db.UpsertEntity(ctx, store.Entity{
		UUID: "a", Name: "Jonathan Harker", EntityType: "person", GroupID: "g",
		Embedding: []float32{1, 0, 0, 0},
	})
	if err != nil {
		t.Fatal(err)
	}

	// "Harker Jonathan" with cosine ~0.99 against A — must resolve to A.
	uuidB, err := ext.resolveOrCreate(ctx, "Harker Jonathan", "person", "g", []float32{0.99, 0.14, 0, 0}, "")
	if err != nil {
		t.Fatal(err)
	}

	if uuidA != uuidB {
		t.Errorf("semantic dedup: expected %q, got %q", uuidA, uuidB)
	}
	n, _ := db.CountEntities(ctx, "g")
	if n != 1 {
		t.Errorf("expected 1 entity after dedup, got %d", n)
	}
}

func TestResolveOrCreate_TypeGuard(t *testing.T) {
	db := openExtractTestDB(t)
	ctx := context.Background()
	ext := newTestExtractor(db, 0.80)

	uuidA, err := db.UpsertEntity(ctx, store.Entity{
		UUID: "a", Name: "Jonathan Harker", EntityType: "person", GroupID: "g",
		Embedding: []float32{1, 0, 0, 0},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Same name pattern + similar embedding but different entity_type — must not merge.
	uuidB, err := ext.resolveOrCreate(ctx, "Harker Jonathan", "tool", "g", []float32{0.99, 0.14, 0, 0}, "")
	if err != nil {
		t.Fatal(err)
	}

	if uuidA == uuidB {
		t.Errorf("type guard: different entity_type must not merge, but both got %q", uuidA)
	}
}

func TestResolveOrCreate_NoEmbeddingFallsThrough(t *testing.T) {
	db := openExtractTestDB(t)
	ctx := context.Background()
	ext := newTestExtractor(db, 0.92)

	uuidA, err := db.UpsertEntity(ctx, store.Entity{
		UUID: "a", Name: "Jonathan Harker", EntityType: "person", GroupID: "g",
		Embedding: []float32{1, 0, 0, 0},
	})
	if err != nil {
		t.Fatal(err)
	}

	// nil embedding → FTS resolution skipped → new entity.
	uuidB, err := ext.resolveOrCreate(ctx, "Harker Jonathan", "person", "g", nil, "")
	if err != nil {
		t.Fatal(err)
	}

	if uuidA == uuidB {
		t.Errorf("nil embedding must not trigger semantic merge")
	}
}
