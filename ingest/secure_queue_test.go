package ingest

import (
	"context"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sharpner/ultramemory/graph"
	"github.com/sharpner/ultramemory/secureindex"
	"github.com/sharpner/ultramemory/store"
)

func TestWalkerEncryptsQueuePayloadWhenPersonalKeySet(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	db := openQueueTestDB(t)
	defer db.Close() //nolint:errcheck

	walker := New(db, "secure-group").WithPersonalKey("secret-key")
	total := 0
	content := strings.Repeat("private cats and dogs ", 8)
	source := filepath.Join(t.TempDir(), "private-note.md")

	if err := walker.enqueueChunks(ctx, content, source, &total); err != nil {
		t.Fatalf("enqueue chunks: %v", err)
	}
	if total == 0 {
		t.Fatal("expected queued chunks")
	}

	job, err := db.NextJob(ctx)
	if err != nil {
		t.Fatalf("next job: %v", err)
	}
	if job == nil {
		t.Fatal("expected queued job")
	}
	if strings.Contains(job.Payload, "private cats and dogs") {
		t.Fatalf("job payload leaked plaintext: %q", job.Payload)
	}
	if strings.Contains(job.Payload, "private-note.md") {
		t.Fatalf("job payload leaked source: %q", job.Payload)
	}

	var payload graph.IngestPayload
	if err := json.Unmarshal([]byte(job.Payload), &payload); err != nil {
		t.Fatalf("decode payload: %v", err)
	}
	if payload.Content != "" {
		t.Fatalf("expected plaintext content to be empty, got %q", payload.Content)
	}
	if payload.Source != "" {
		t.Fatalf("expected plaintext source to be empty, got %q", payload.Source)
	}
	if payload.ContentCiphertext == "" {
		t.Fatal("expected encrypted content payload")
	}
	if payload.SourceCiphertext == "" {
		t.Fatal("expected encrypted source payload")
	}

	decryptedContent, err := secureindex.DecryptString("secret-key", "ingest-content", payload.ContentCiphertext)
	if err != nil {
		t.Fatalf("decrypt content payload: %v", err)
	}
	if !strings.Contains(decryptedContent, "private cats and dogs") {
		t.Fatalf("unexpected decrypted content: %q", decryptedContent)
	}

	decryptedSource, err := secureindex.DecryptString("secret-key", "ingest-source", payload.SourceCiphertext)
	if err != nil {
		t.Fatalf("decrypt source payload: %v", err)
	}
	if decryptedSource != source {
		t.Fatalf("source mismatch: got %q want %q", decryptedSource, source)
	}
}

func openQueueTestDB(t *testing.T) *store.DB {
	t.Helper()

	db, err := store.Open(filepath.Join(t.TempDir(), "queue-secure.db"))
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	return db
}
