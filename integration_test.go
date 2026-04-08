//go:build !mistral

// Integration tests for memory-local.
// Requires Ollama running locally with gemma3:4b and mxbai-embed-large pulled.
// Run with: go test ./cmd/memory-local/ -v -timeout 5m -run TestIntegration
package main

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/sharpner/ultramemory/graph"
	"github.com/sharpner/ultramemory/ingest"
	"github.com/sharpner/ultramemory/llm"
	"github.com/sharpner/ultramemory/store"
)

const (
	testOllamaURL      = "http://localhost:11434"
	testExtractModel   = "gemma3:4b"
	testEmbeddingModel = "mxbai-embed-large"
	testGroupID        = "test"
)

func skipIfNoOllama(t *testing.T, client *llm.Client) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := client.Ping(ctx); err != nil {
		t.Skipf("Ollama not available: %v", err)
	}
}

func openTestDB(t *testing.T) *store.DB {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "memory-test-*.db")
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

// TestIntegrationIngestAndSearch ingests testdata/ and checks that known
// entities and relationships can be recalled via search.
func TestIntegrationIngestAndSearch(t *testing.T) {
	client := llm.New(testOllamaURL, testExtractModel, testEmbeddingModel)
	skipIfNoOllama(t, client)

	db := openTestDB(t)
	ctx := context.Background()

	// Ingest testdata/
	walker := ingest.New(db, testGroupID)
	n, err := walker.Walk(ctx, "testdata")
	if err != nil {
		t.Fatalf("ingest walk: %v", err)
	}
	t.Logf("enqueued %d chunks", n)
	if n == 0 {
		t.Fatal("no chunks enqueued — testdata missing?")
	}

	// Process all jobs synchronously (no goroutines, deterministic for tests)
	ext := graph.New(db, client, client, 0.92, 1, "")
	processed := 0
	failed := 0
	for {
		job, err := db.NextJob(ctx)
		if err != nil {
			t.Fatalf("next job: %v", err)
		}
		if job == nil {
			break // queue empty
		}
		if err := ext.ProcessJob(ctx, job.Payload, job.Attempts); err != nil {
			t.Logf("job %d failed: %v", job.ID, err)
			if err2 := db.FailJob(ctx, job.ID, err.Error()); err2 != nil {
				t.Fatalf("fail job: %v", err2)
			}
			failed++
			continue
		}
		if err := db.CompleteJob(ctx, job.ID); err != nil {
			t.Fatalf("complete job: %v", err)
		}
		processed++
	}
	t.Logf("processed=%d failed=%d", processed, failed)

	// Wait for all async embedding goroutines to finish before searching.
	ext.Wait()

	// Verify graph was built
	entities, err := db.CountEntities(ctx, testGroupID)
	if err != nil {
		t.Fatalf("count entities: %v", err)
	}
	edges, err := db.CountEdges(ctx, testGroupID)
	if err != nil {
		t.Fatalf("count edges: %v", err)
	}
	t.Logf("entities=%d edges=%d", entities, edges)

	if entities < 3 {
		t.Errorf("expected ≥3 entities, got %d", entities)
	}
	if edges < 2 {
		t.Errorf("expected ≥2 edges, got %d", edges)
	}

	// Recall: each query must find the expected entity despite noise documents.
	// Precision: the correct entity must rank HIGHER than the confusable noise entity.
	recallCases := []struct {
		query     string
		contains  string // must appear in results (recall)
		rankAbove string // must rank higher than this confusable entity (precision)
	}{
		// Alice Schmidt (engineer) must rank above Alice Cooper (musician)
		{"Alice Schmidt software engineer TechCorp", "alice schmidt", "alice cooper"},
		// TechCorp Berlin must rank above TechGlobal Munich
		{"TechCorp Berlin PostgreSQL", "techcorp berlin", "techglobal"},
		// Bob Müller (CTO) must rank above Bob Dylan (musician)
		{"Bob Müller CTO TechCorp Berlin", "bob müller", "bob dylan"},
		// PostgreSQL (used by TechCorp) must rank above MongoDB (used by TechGlobal)
		{"PostgreSQL database TechCorp", "postgresql", "mongodb"},
		// Zalando Berlin must rank above Goldman Partners New York
		{"Berlin startup Zalando Germany", "zalando", "goldman"},
		// Goldman finance must rank above Alice Schmidt
		{"Goldman Partners New York finance", "goldman", "alice schmidt"},
	}

	for _, tc := range recallCases {
		t.Run(tc.query, func(t *testing.T) {
			results, err := graph.Search(ctx, db, client, tc.query, testGroupID, 10)
			if err != nil {
				t.Fatalf("search: %v", err)
			}
			if len(results) == 0 {
				t.Errorf("no results for %q", tc.query)
				return
			}

			// Recall: expected entity must appear somewhere in results
			correctRank := -1
			for i, r := range results {
				if strings.Contains(strings.ToLower(r.Title+" "+r.Body), tc.contains) {
					correctRank = i
					break
				}
			}
			if correctRank == -1 {
				t.Errorf("RECALL FAIL: query %q: %q not found\ngot: %v",
					tc.query, tc.contains, resultTexts(results))
				return
			}

			// Precision: correct entity must rank higher than the confusable one
			noiseRank := -1
			for i, r := range results {
				if strings.Contains(strings.ToLower(r.Title+" "+r.Body), tc.rankAbove) {
					noiseRank = i
					break
				}
			}
			if noiseRank != -1 && noiseRank < correctRank {
				t.Errorf("PRECISION FAIL: query %q: noise %q (rank %d) ranked above correct %q (rank %d)\ngot: %v",
					tc.query, tc.rankAbove, noiseRank+1, tc.contains, correctRank+1, resultTexts(results))
			}
		})
	}
}

// TestIntegrationEntityExtraction tests the LLM extraction in isolation.
func TestIntegrationEntityExtraction(t *testing.T) {
	client := llm.New(testOllamaURL, testExtractModel, testEmbeddingModel)
	skipIfNoOllama(t, client)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	text := "Alice Schmidt works at TechCorp Berlin. Bob Müller is the CTO of TechCorp Berlin."
	entities, err := client.ExtractEntities(ctx, text)
	if err != nil {
		t.Fatalf("extract entities: %v", err)
	}
	t.Logf("extracted %d entities: %+v", len(entities.Entities), entities.Entities)

	names := make([]string, 0, len(entities.Entities))
	for _, e := range entities.Entities {
		names = append(names, strings.ToLower(e.Name))
	}

	mustContain := []string{"alice", "bob", "techcorp"}
	for _, want := range mustContain {
		found := false
		for _, name := range names {
			if strings.Contains(name, want) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("expected entity containing %q, got names: %v", want, names)
		}
	}

	if len(entities.Entities) < 2 {
		t.Errorf("expected ≥2 entities, got %d", len(entities.Entities))
	}
}

// TestIntegrationEdgeExtraction tests relationship extraction.
func TestIntegrationEdgeExtraction(t *testing.T) {
	client := llm.New(testOllamaURL, testExtractModel, testEmbeddingModel)
	skipIfNoOllama(t, client)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	text := "Alice Schmidt works at TechCorp Berlin. Bob Müller is the CTO of TechCorp Berlin."
	extracted, err := client.ExtractEntities(ctx, text)
	if err != nil {
		t.Fatalf("extract entities: %v", err)
	}

	edges, err := client.ExtractEdges(ctx, extracted.Entities, text)
	if err != nil {
		t.Fatalf("extract edges: %v", err)
	}
	t.Logf("extracted %d edges: %+v", len(edges.Edges), edges.Edges)

	if len(edges.Edges) == 0 {
		t.Error("expected ≥1 edge, got 0")
	}
}

// TestIntegrationEmbedding tests that embeddings are generated and have correct dimensions.
func TestIntegrationEmbedding(t *testing.T) {
	client := llm.New(testOllamaURL, testExtractModel, testEmbeddingModel)
	skipIfNoOllama(t, client)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	vec, err := client.Embed(ctx, "Alice works at TechCorp")
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	if len(vec) == 0 {
		t.Fatal("empty embedding")
	}
	t.Logf("embedding dim=%d", len(vec))
	// mxbai-embed-large produces 1024-dim vectors
	if len(vec) < 100 {
		t.Errorf("suspiciously small embedding: dim=%d", len(vec))
	}
}

func resultTexts(results []graph.SearchResult) []string {
	out := make([]string, len(results))
	for i, r := range results {
		out[i] = r.Title + ": " + r.Body
	}
	return out
}
