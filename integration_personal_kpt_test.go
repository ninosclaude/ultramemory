package main

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/sharpner/ultramemory/graph"
	"github.com/sharpner/ultramemory/llm"
	"github.com/sharpner/ultramemory/secureindex"
	"github.com/sharpner/ultramemory/store"
)

func TestPersonalKPTFlowWithExistingUltramemoryPieces(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()
	groupID := testGroupID

	episodes := []store.Episode{
		{
			UUID:      "ep-cats-1",
			Content:   "Cats wear hats and nap on windowsills.",
			GroupID:   groupID,
			Source:    "cats-1.md",
			Embedding: testEpisodeVector(64, 0),
		},
		{
			UUID:      "ep-cats-2",
			Content:   "Kittens purr loudly when they find warm blankets.",
			GroupID:   groupID,
			Source:    "cats-2.md",
			Embedding: testEpisodeVector(64, 0),
		},
		{
			UUID:      "ep-space-1",
			Content:   "Rockets leave earth and satellites orbit the planet.",
			GroupID:   groupID,
			Source:    "space-1.md",
			Embedding: testEpisodeVector(64, 1),
		},
	}
	for _, episode := range episodes {
		if err := db.UpsertEpisode(ctx, episode); err != nil {
			t.Fatalf("upsert episode %s: %v", episode.UUID, err)
		}
	}

	method := secureindex.NewMethod("alpha-key", 64)
	rows := make([]store.SecureEpisodeIndexRow, 0, len(episodes))
	for _, episode := range episodes {
		state := method.EncodeDoc(episode.Embedding)
		rows = append(rows, store.SecureEpisodeIndexRow{
			EpisodeUUID:  episode.UUID,
			GroupID:      groupID,
			Method:       secureMethod,
			Public:       state.Public,
			BaseWaveReal: state.BaseWaveReal,
			BaseWaveImag: state.BaseWaveImag,
			WaveReal:     state.WaveReal,
			WaveImag:     state.WaveImag,
			ModeWeight:   state.ModeWeight,
			ModeEnergy:   state.ModeEnergy,
		})
	}
	if err := db.ReplaceSecureEpisodeIndex(ctx, groupID, secureMethod, rows); err != nil {
		t.Fatalf("replace secure episode index: %v", err)
	}

	loaded, err := db.AllSecureEpisodes(ctx, groupID, secureMethod)
	if err != nil {
		t.Fatalf("load secure episodes: %v", err)
	}
	if len(loaded) != len(episodes) {
		t.Fatalf("expected %d secure episodes, got %d", len(episodes), len(loaded))
	}

	states := secureStates(loaded)
	goodQuery := method.EncodeQuery(testEpisodeVector(64, 0))
	goodHits := method.Search(states, goodQuery, 3)
	if loaded[goodHits[0].Index].EpisodeUUID != "ep-cats-1" && loaded[goodHits[0].Index].EpisodeUUID != "ep-cats-2" {
		t.Fatalf("expected cat episode first for good key, got %s", loaded[goodHits[0].Index].EpisodeUUID)
	}

	wrongMethod := secureindex.NewMethod("wrong-key", 64)
	wrongQuery := wrongMethod.EncodeQuery(testEpisodeVector(64, 0))
	wrongHits := wrongMethod.Search(states, wrongQuery, 3)
	if wrongHits[0].Score >= 0.05 {
		t.Fatalf("expected wrong-key collapse, got %.4f", wrongHits[0].Score)
	}
	if goodHits[0].Score <= wrongHits[0].Score+0.20 {
		t.Fatalf("expected large good/wrong margin, got good=%.4f wrong=%.4f", goodHits[0].Score, wrongHits[0].Score)
	}

	clusters := method.Cluster(states, 2, 1.0, 0.30)
	if len(clusters) == 0 {
		t.Fatal("expected personal clusters")
	}

	mock := &testPersonalMockLLM{
		entitiesJSON: testEntityJSON("Cat Topic", "Space Topic"),
		edgesJSON:    testEdgeJSON("REL", "Cat Topic references Space Topic"),
	}
	results, err := graph.Search(ctx, db, mock, "cats blankets hats", groupID, 5)
	if err != nil {
		t.Fatalf("graph search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected normal ultramemory search results")
	}
}

func testEpisodeVector(dim int, family int) []float32 {
	out := make([]float32, dim)
	block := dim / 4
	start := family * block
	for i := 0; i < block; i++ {
		out[start+i] = 1
	}
	for i := family; i < dim; i += 9 {
		out[i] += 0.15
	}
	return out
}

type testPersonalMockLLM struct {
	entitiesJSON string
	edgesJSON    string
}

func (m *testPersonalMockLLM) ExtractEntities(context.Context, string) (*llm.ExtractedEntities, error) {
	var result llm.ExtractedEntities
	if err := json.Unmarshal([]byte(m.entitiesJSON), &result); err == nil {
		return &result, nil
	}

	var direct []llm.ExtractedEntity
	if err := json.Unmarshal([]byte(m.entitiesJSON), &direct); err != nil {
		return nil, err
	}
	return &llm.ExtractedEntities{Entities: direct}, nil
}

func (m *testPersonalMockLLM) ExtractEdges(context.Context, []llm.ExtractedEntity, string) (*llm.ExtractedEdges, error) {
	var result llm.ExtractedEdges
	if err := json.Unmarshal([]byte(m.edgesJSON), &result); err == nil {
		return &result, nil
	}

	var direct []llm.ExtractedEdge
	if err := json.Unmarshal([]byte(m.edgesJSON), &direct); err != nil {
		return nil, err
	}
	return &llm.ExtractedEdges{Edges: direct}, nil
}

func (m *testPersonalMockLLM) Embed(context.Context, string) ([]float32, error) {
	return make([]float32, 4), nil
}

func (m *testPersonalMockLLM) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	vectors := make([][]float32, len(texts))
	for i := range texts {
		vectors[i] = make([]float32, 4)
	}
	return vectors, nil
}

func testEntityJSON(names ...string) string {
	type ent struct {
		Name       string `json:"name"`
		EntityType string `json:"entity_type"`
	}
	entities := make([]ent, len(names))
	for i, name := range names {
		entities[i] = ent{Name: name, EntityType: "Person"}
	}
	b, _ := json.Marshal(map[string]any{"extracted_entities": entities})
	return string(b)
}

func testEdgeJSON(relation, fact string) string {
	b, _ := json.Marshal(map[string]any{"edges": []map[string]any{{
		"relation_type":    relation,
		"source_entity_id": 0,
		"target_entity_id": 1,
		"fact":             fact,
		"valid_at":         nil,
		"invalid_at":       nil,
	}}})
	return string(b)
}
