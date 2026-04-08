package store

import (
	"context"
	"reflect"
	"testing"
)

func TestSecureEpisodeIndexRoundTrip(t *testing.T) {
	db := openTestDB(t)
	ctx := context.Background()
	groupID := "grp"
	method := "kpt-v1"

	err := db.UpsertEpisode(ctx, Episode{
		UUID:      "ep-1",
		Content:   "cats wear hats",
		GroupID:   groupID,
		Source:    "test.md",
		Embedding: []float32{1, 0.2, 0.1},
	})
	if err != nil {
		t.Fatalf("upsert episode: %v", err)
	}

	rows := []SecureEpisodeIndexRow{{
		EpisodeUUID:  "ep-1",
		GroupID:      groupID,
		Method:       method,
		Public:       []float32{0.1, 0.2},
		BaseWaveReal: []float32{0.3, 0.4, 0.5},
		BaseWaveImag: []float32{0.6, 0.7, 0.8},
		WaveReal:     []float32{0.9, 1.0, 1.1, 1.2},
		WaveImag:     []float32{1.3, 1.4, 1.5, 1.6},
		ModeWeight:   []float32{0.7, 0.3},
		ModeEnergy:   []float32{0.2, 0.1},
	}}

	err = db.ReplaceSecureEpisodeIndex(ctx, groupID, method, rows)
	if err != nil {
		t.Fatalf("replace secure index: %v", err)
	}

	got, err := db.AllSecureEpisodes(ctx, groupID, method)
	if err != nil {
		t.Fatalf("load secure index: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("expected 1 secure row, got %d", len(got))
	}
	if !reflect.DeepEqual(got[0].BaseWaveReal, rows[0].BaseWaveReal) {
		t.Fatalf("base real mismatch: got %v want %v", got[0].BaseWaveReal, rows[0].BaseWaveReal)
	}
	if !reflect.DeepEqual(got[0].BaseWaveImag, rows[0].BaseWaveImag) {
		t.Fatalf("base imag mismatch: got %v want %v", got[0].BaseWaveImag, rows[0].BaseWaveImag)
	}
	if !reflect.DeepEqual(got[0].WaveReal, rows[0].WaveReal) {
		t.Fatalf("wave real mismatch: got %v want %v", got[0].WaveReal, rows[0].WaveReal)
	}
	if !reflect.DeepEqual(got[0].WaveImag, rows[0].WaveImag) {
		t.Fatalf("wave imag mismatch: got %v want %v", got[0].WaveImag, rows[0].WaveImag)
	}
	if !reflect.DeepEqual(got[0].ModeWeight, rows[0].ModeWeight) {
		t.Fatalf("mode weight mismatch: got %v want %v", got[0].ModeWeight, rows[0].ModeWeight)
	}
	if !reflect.DeepEqual(got[0].ModeEnergy, rows[0].ModeEnergy) {
		t.Fatalf("mode energy mismatch: got %v want %v", got[0].ModeEnergy, rows[0].ModeEnergy)
	}
}
