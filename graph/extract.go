// Package graph orchestrates entity/edge extraction and graph building.
package graph

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/sharpner/ultramemory/llm"
	"github.com/sharpner/ultramemory/secureindex"
	"github.com/sharpner/ultramemory/store"
)

// IngestPayload is the JSON payload stored in the job queue.
type IngestPayload struct {
	Content           string `json:"content"`
	Source            string `json:"source"`
	GroupID           string `json:"group_id"`
	ContentCiphertext string `json:"content_ciphertext,omitempty"`
	SourceCiphertext  string `json:"source_ciphertext,omitempty"`
}

// Extractor runs the full graph-building pipeline for a document chunk.
// The semaphore limits concurrent extraction calls.
// Embedding runs outside the semaphore since it may use a different model than extraction.
type Extractor struct {
	db               *store.DB
	extractor        llm.EntityExtractor // entity/edge extraction (Ollama or Mistral API)
	embedder         llm.Embedder
	sem              chan struct{}  // limits concurrent LLM extraction calls
	muEntity         sync.Mutex     // serialise entity upserts to avoid duplicates under concurrency
	embedWG          sync.WaitGroup // tracks in-flight embedding goroutines
	secureMu         sync.Mutex
	resolveThreshold float64
	personalKey      string
	secureMethod     *secureindex.Method
	secureDim        int
}

// New creates a new Extractor. extractor handles entity/edge extraction (Ollama or Mistral API).
// embedder handles embeddings for the active build profile.
// llmParallel controls how many concurrent extraction calls are allowed.
// resolveThreshold is the minimum cosine similarity for entity deduplication (e.g. 0.92).
func New(db *store.DB, extractor llm.EntityExtractor, embedder llm.Embedder, resolveThreshold float64, llmParallel int, personalKey string) *Extractor {
	if llmParallel < 1 {
		llmParallel = 1
	}
	return &Extractor{
		db:               db,
		extractor:        extractor,
		embedder:         embedder,
		sem:              make(chan struct{}, llmParallel),
		resolveThreshold: resolveThreshold,
		personalKey:      personalKey,
	}
}

// ProcessJob deserialises a queue job and runs the full extraction pipeline.
// attempts indicates how many previous failures occurred — used to increase
// LLM temperature on retries so the model produces different output.
func (e *Extractor) ProcessJob(ctx context.Context, payload string, attempts int) error {
	var p IngestPayload
	if err := json.Unmarshal([]byte(payload), &p); err != nil {
		return fmt.Errorf("decode payload: %w", err)
	}
	content, err := p.DecodeContent(e.personalKey)
	if err != nil {
		return err
	}
	source, err := p.DecodeSource(e.personalKey)
	if err != nil {
		return err
	}

	// On retries, bump temperature to get different LLM output.
	// attempt 0 → temp 0 (deterministic), 1 → 0.3, 2 → 0.6
	temp := float64(attempts) * 0.3
	if temp > 0.8 {
		temp = 0.8
	}
	e.setExtractorTemperature(temp)
	defer e.setExtractorTemperature(0) // reset for next job

	return e.Process(ctx, content, source, p.GroupID)
}

// setExtractorTemperature sets temperature on whichever LLM backend is active.
func (e *Extractor) setExtractorTemperature(t float64) {
	type tempSetter interface{ SetTemperature(float64) }
	if ts, ok := e.extractor.(tempSetter); ok {
		ts.SetTemperature(t)
	}
}

// Process runs entity extraction, edge extraction, and embedding for one text chunk.
// The LLM semaphore serialises calls; embedding is fire-and-forget per entity.
func (e *Extractor) Process(ctx context.Context, content, source, groupID string) error {
	if e.personalKey != "" {
		return e.processSecure(ctx, content, source, groupID)
	}

	epUUID := uuid.New().String()

	// ── 1. Store raw episode immediately ─────────────────────────────────────
	ep := store.Episode{
		UUID:    epUUID,
		Content: content,
		GroupID: groupID,
		Source:  source,
	}
	if err := e.db.UpsertEpisode(ctx, ep); err != nil {
		return fmt.Errorf("store episode: %w", err)
	}

	// Episode embeddings power episode search and KPT indexing even when
	// entity extraction returns nothing, so launch them immediately.
	e.embedWG.Add(1)
	go e.embedEpisode(context.Background(), epUUID, content)

	// ── 2. Acquire LLM semaphore (max 1 gemma3:4b at a time) ────────────────
	select {
	case e.sem <- struct{}{}:
	case <-ctx.Done():
		return ctx.Err()
	}

	start := time.Now()

	// ── 3. Entity extraction ─────────────────────────────────────────────────
	extracted, err := e.extractor.ExtractEntities(ctx, content)
	if err != nil {
		<-e.sem
		return fmt.Errorf("extract entities: %w", err)
	}

	if len(extracted.Entities) == 0 {
		<-e.sem
		slog.Debug("no entities found", "source", source)
		return nil
	}

	// ── 4. Edge extraction ───────────────────────────────────────────────────
	edges, err := e.extractor.ExtractEdges(ctx, extracted.Entities, content)
	if err != nil {
		<-e.sem
		return fmt.Errorf("extract edges: %w", err)
	}
	<-e.sem // release semaphore — LLM work done

	slog.Info("extracted",
		"source", shortPath(source),
		"entities", len(extracted.Entities),
		"edges", len(edges.Edges),
		"llm_ms", time.Since(start).Milliseconds(),
	)

	// ── 5. Batch-embed all entity descriptions + edge facts in one API call ──
	// Collecting all texts upfront avoids N×round-trips to the embedding backend.
	// Entities come first (indices 0..len-1), edges follow.
	batchTexts := make([]string, 0, len(extracted.Entities)+len(edges.Edges))
	for _, ent := range extracted.Entities {
		t := ent.Description
		if t == "" {
			t = entityEmbedText(ent.Name, ent.EntityType)
		}
		batchTexts = append(batchTexts, t)
	}
	for _, ex := range edges.Edges {
		if ex.Fact != "" {
			batchTexts = append(batchTexts, ex.Fact)
		} else {
			batchTexts = append(batchTexts, "") // placeholder — will yield nil embedding
		}
	}

	batchEmbs, batchErr := e.embedder.EmbedBatch(ctx, batchTexts)
	if batchErr != nil {
		slog.Debug("embed batch failed, falling back to sequential", "err", batchErr)
		batchEmbs = nil // signals fallback below
	}

	embAt := func(i int) []float32 {
		if batchEmbs != nil && i < len(batchEmbs) {
			return batchEmbs[i]
		}
		return nil
	}

	// ── 6. Resolve entities: dedup → upsert → link ───────────────────────────
	entityUUIDs := make([]string, len(extracted.Entities))
	for i, ent := range extracted.Entities {
		vec := embAt(i)
		if vec == nil {
			// Batch failed — fall back to single embed.
			t := ent.Description
			if t == "" {
				t = entityEmbedText(ent.Name, ent.EntityType)
			}
			vec, err = e.embedder.Embed(ctx, t)
			if err != nil {
				slog.Debug("embed entity failed", "name", ent.Name, "err", err)
				vec = nil
			}
		}

		e.muEntity.Lock()
		canonical, err := e.resolveOrCreate(ctx, ent.Name, ent.EntityType, groupID, vec, ent.Description)
		e.muEntity.Unlock()
		if err != nil {
			return fmt.Errorf("resolve entity %q: %w", ent.Name, err)
		}
		entityUUIDs[i] = canonical

		if err := e.db.LinkEntityEpisode(ctx, canonical, epUUID); err != nil {
			return fmt.Errorf("link entity-episode: %w", err)
		}

		// Incremental mutual-kNN update: integrate new entity into the
		// semantic neighbor graph. Cost: O(n × d), ~100ms at 100k entities.
		if vec != nil {
			if err := e.db.UpdateMutualKNN(ctx, canonical, groupID, vec, 20); err != nil {
				slog.Debug("mutual-knn update failed", "entity", ent.Name, "err", err)
			}
		}
	}

	// ── 7. Store edges (embeddings from batch, fallback to sequential) ────────
	entCount := len(extracted.Entities)
	for ei, ex := range edges.Edges {
		if ex.SourceEntityID < 0 || ex.SourceEntityID >= len(entityUUIDs) {
			continue
		}
		if ex.TargetEntityID < 0 || ex.TargetEntityID >= len(entityUUIDs) {
			continue
		}
		edgeEmb := embAt(entCount + ei)
		if edgeEmb == nil && ex.Fact != "" {
			// Batch failed — fall back to single embed.
			edgeEmb, err = e.embedder.Embed(ctx, ex.Fact)
			if err != nil {
				slog.Debug("embed edge fact failed", "fact", ex.Fact, "err", err)
			}
		}
		if err := e.db.UpsertEdge(ctx, store.Edge{
			UUID:       uuid.New().String(),
			SourceUUID: entityUUIDs[ex.SourceEntityID],
			TargetUUID: entityUUIDs[ex.TargetEntityID],
			Name:       ex.RelationType,
			Fact:       ex.Fact,
			GroupID:    groupID,
			ValidAt:    ex.ValidAt,
			InvalidAt:  ex.InvalidAt,
			Episodes:   fmt.Sprintf(`["%s"]`, epUUID),
			Embedding:  edgeEmb,
		}); err != nil {
			return fmt.Errorf("upsert edge: %w", err)
		}
	}

	return nil
}

func (p IngestPayload) DecodeContent(personalKey string) (string, error) {
	if p.ContentCiphertext == "" {
		return p.Content, nil
	}
	if personalKey == "" {
		return "", errors.New("encrypted ingest payload requires MEMORY_PERSONAL_KEY")
	}
	content, err := secureindex.DecryptString(personalKey, "ingest-content", p.ContentCiphertext)
	if err != nil {
		return "", fmt.Errorf("decrypt content payload: %w", err)
	}
	return content, nil
}

func (p IngestPayload) DecodeSource(personalKey string) (string, error) {
	if p.SourceCiphertext == "" {
		return p.Source, nil
	}
	if personalKey == "" {
		return "", errors.New("encrypted ingest payload requires MEMORY_PERSONAL_KEY")
	}
	source, err := secureindex.DecryptString(personalKey, "ingest-source", p.SourceCiphertext)
	if err != nil {
		return "", fmt.Errorf("decrypt source payload: %w", err)
	}
	return source, nil
}

func (e *Extractor) processSecure(ctx context.Context, content, source, groupID string) error {
	embedding, err := e.embedder.Embed(ctx, content)
	if err != nil {
		return fmt.Errorf("embed secure episode: %w", err)
	}

	contentCiphertext, err := secureindex.EncryptString(e.personalKey, "episode-content", content)
	if err != nil {
		return fmt.Errorf("encrypt episode content: %w", err)
	}
	sourceCiphertext, err := secureindex.EncryptString(e.personalKey, "episode-source", source)
	if err != nil {
		return fmt.Errorf("encrypt episode source: %w", err)
	}

	epUUID := uuid.New().String()
	if err := e.db.UpsertEpisode(ctx, store.Episode{
		UUID:    epUUID,
		Content: contentCiphertext,
		GroupID: groupID,
		Source:  sourceCiphertext,
	}); err != nil {
		return fmt.Errorf("store secure episode: %w", err)
	}

	method := e.secureEncoder(len(embedding))
	state := method.EncodeDoc(embedding)
	err = e.db.UpsertSecureEpisodeIndexRow(ctx, store.SecureEpisodeIndexRow{
		EpisodeUUID:  epUUID,
		GroupID:      groupID,
		Method:       "kpt-v1",
		Public:       state.Public,
		BaseWaveReal: state.BaseWaveReal,
		BaseWaveImag: state.BaseWaveImag,
		WaveReal:     state.WaveReal,
		WaveImag:     state.WaveImag,
		ModeWeight:   state.ModeWeight,
		ModeEnergy:   state.ModeEnergy,
	})
	if err != nil {
		return fmt.Errorf("store secure episode index: %w", err)
	}

	select {
	case e.sem <- struct{}{}:
	case <-ctx.Done():
		return ctx.Err()
	}

	extracted, err := e.extractor.ExtractEntities(ctx, content)
	if err != nil {
		<-e.sem
		return fmt.Errorf("extract entities: %w", err)
	}
	if len(extracted.Entities) == 0 {
		<-e.sem
		return nil
	}

	edges, err := e.extractor.ExtractEdges(ctx, extracted.Entities, content)
	if err != nil {
		<-e.sem
		return fmt.Errorf("extract edges: %w", err)
	}
	<-e.sem

	entityUUIDs := make([]string, len(extracted.Entities))
	for i, ent := range extracted.Entities {
		e.muEntity.Lock()
		canonical, err := e.resolveOrCreate(ctx, ent.Name, ent.EntityType, groupID, nil, ent.Description)
		e.muEntity.Unlock()
		if err != nil {
			return fmt.Errorf("resolve secure entity %q: %w", ent.Name, err)
		}
		entityUUIDs[i] = canonical
		if err := e.db.LinkEntityEpisode(ctx, canonical, epUUID); err != nil {
			return fmt.Errorf("link secure entity-episode: %w", err)
		}
	}

	for _, ex := range edges.Edges {
		if ex.SourceEntityID < 0 || ex.SourceEntityID >= len(entityUUIDs) {
			continue
		}
		if ex.TargetEntityID < 0 || ex.TargetEntityID >= len(entityUUIDs) {
			continue
		}
		lookupKey, err := secureindex.LookupDigest(e.personalKey, "edge-lookup",
			entityUUIDs[ex.SourceEntityID]+"|"+entityUUIDs[ex.TargetEntityID]+"|"+strings.ToLower(ex.RelationType))
		if err != nil {
			return fmt.Errorf("edge lookup digest: %w", err)
		}
		nameCiphertext, err := secureindex.EncryptString(e.personalKey, "edge-name", ex.RelationType)
		if err != nil {
			return fmt.Errorf("encrypt edge name: %w", err)
		}
		factCiphertext, err := secureindex.EncryptString(e.personalKey, "edge-fact", ex.Fact)
		if err != nil {
			return fmt.Errorf("encrypt edge fact: %w", err)
		}
		if err := e.db.UpsertEdge(ctx, store.Edge{
			UUID:       uuid.New().String(),
			SourceUUID: entityUUIDs[ex.SourceEntityID],
			TargetUUID: entityUUIDs[ex.TargetEntityID],
			Name:       nameCiphertext,
			LookupKey:  lookupKey,
			Fact:       factCiphertext,
			GroupID:    groupID,
			ValidAt:    ex.ValidAt,
			InvalidAt:  ex.InvalidAt,
			Episodes:   fmt.Sprintf(`["%s"]`, epUUID),
		}); err != nil {
			return fmt.Errorf("upsert secure edge: %w", err)
		}
	}

	return nil
}

func (e *Extractor) secureEncoder(dim int) *secureindex.Method {
	e.secureMu.Lock()
	defer e.secureMu.Unlock()

	if e.secureMethod != nil && e.secureDim == dim {
		return e.secureMethod
	}

	e.secureMethod = secureindex.NewMethod(e.personalKey, dim)
	e.secureDim = dim
	return e.secureMethod
}

// Wait blocks until all pending embedding goroutines have finished.
// Useful in tests to ensure embeddings are written before searching.
func (e *Extractor) Wait() {
	e.embedWG.Wait()
}

// resolveOrCreate looks up an existing entity via FTS + cosine similarity, or
// inserts a new one. Caller must hold muEntity.
func (e *Extractor) resolveOrCreate(ctx context.Context, name, entityType, groupID string, embedding []float32, description string) (string, error) {
	if e.personalKey != "" {
		lookupKey, err := secureindex.LookupDigest(e.personalKey, "entity-lookup", strings.ToLower(name))
		if err != nil {
			return "", fmt.Errorf("entity lookup digest: %w", err)
		}
		nameCiphertext, err := secureindex.EncryptString(e.personalKey, "entity-name", name)
		if err != nil {
			return "", fmt.Errorf("encrypt entity name: %w", err)
		}
		descriptionCiphertext, err := secureindex.EncryptString(e.personalKey, "entity-description", description)
		if err != nil {
			return "", fmt.Errorf("encrypt entity description: %w", err)
		}
		return e.db.UpsertEntity(ctx, store.Entity{
			UUID:        uuid.New().String(),
			Name:        nameCiphertext,
			LookupKey:   lookupKey,
			EntityType:  entityType,
			GroupID:     groupID,
			Description: descriptionCiphertext,
		})
	}
	if len(embedding) > 0 {
		candidates, err := e.db.SearchEntitiesFTS(ctx, name, groupID, 5)
		if err != nil {
			return "", fmt.Errorf("fts candidates: %w", err)
		}
		for _, c := range candidates {
			if c.EntityType != entityType || len(c.Embedding) == 0 {
				continue
			}
			if store.CosineSimilarity(embedding, c.Embedding) >= e.resolveThreshold {
				return c.UUID, nil
			}
		}
	}
	return e.db.UpsertEntity(ctx, store.Entity{
		UUID:        uuid.New().String(),
		Name:        name,
		EntityType:  entityType,
		GroupID:     groupID,
		Embedding:   embedding,
		Description: description,
	})
}

func (e *Extractor) embedEpisode(ctx context.Context, uuid, content string) {
	defer e.embedWG.Done()
	vec, err := e.embedder.Embed(ctx, content)
	if err != nil {
		slog.Debug("embed episode failed", "err", err)
		return
	}
	if _, err := e.db.SQL().ExecContext(ctx,
		`UPDATE episodes SET embedding = ? WHERE uuid = ?`,
		store.EncodeEmbedding(vec), uuid,
	); err != nil {
		slog.Debug("store episode embedding failed", "err", err)
	}
}

// entityEmbedText generates a sentence for embedding an entity name.
// Wrapping in a typed sentence produces better embeddings than bare names.
func entityEmbedText(name, entityType string) string {
	switch entityType {
	case "Person":
		return "A person named " + name
	case "Organization":
		return "An organization called " + name
	case "Place":
		return "A place called " + name
	case "Product":
		return "A product called " + name
	case "Event":
		return "An event called " + name
	default:
		return "A concept called " + name
	}
}

func shortPath(s string) string {
	if len(s) <= 50 {
		return s
	}
	return "…" + s[len(s)-47:]
}
