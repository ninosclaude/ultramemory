package secureindex

import "testing"

func TestSameKeyScoresAboveWrongKey(t *testing.T) {
	doc1 := normalize(testVector(64, 0))
	doc2 := normalize(testVector(64, 1))

	builder := NewMethod("alpha-key", len(doc1))
	docs := []State{
		builder.EncodeDoc(doc1),
		builder.EncodeDoc(doc2),
	}

	goodQuery := builder.EncodeQuery(doc1)
	goodHits := builder.Search(docs, goodQuery, 2)
	if goodHits[0].Index != 0 {
		t.Fatalf("expected doc1 to rank first with correct key, got index %d", goodHits[0].Index)
	}

	wrongQuery := NewMethod("wrong-key", len(doc1)).EncodeQuery(doc1)
	wrongHits := NewMethod("wrong-key", len(doc1)).Search(docs, wrongQuery, 2)

	if goodHits[0].Score <= wrongHits[0].Score+0.15 {
		t.Fatalf("expected strong same-key margin, got good=%0.4f wrong=%0.4f", goodHits[0].Score, wrongHits[0].Score)
	}
	if wrongHits[0].Score >= 0.35 {
		t.Fatalf("expected wrong-key collapse, got wrong=%0.4f", wrongHits[0].Score)
	}
}

func TestClusterKeepsSimilarDocsTogether(t *testing.T) {
	method := NewMethod("cluster-key", 64)
	docs := []State{
		method.EncodeDoc(normalize(testVector(64, 0))),
		method.EncodeDoc(normalize(testVector(64, 0))),
		method.EncodeDoc(normalize(testVector(64, 1))),
		method.EncodeDoc(normalize(testVector(64, 1))),
	}

	clusters := method.Cluster(docs, 2, 1.0, 0.35)
	if len(clusters) == 0 {
		t.Fatalf("expected at least one cluster")
	}

	foundPair := false
	for _, cluster := range clusters {
		if len(cluster.Members) < 2 {
			continue
		}
		if cluster.Members[0] == 0 && cluster.Members[1] == 1 {
			foundPair = true
			break
		}
	}
	if !foundPair {
		t.Fatalf("expected docs 0 and 1 to cluster together, got %+v", clusters)
	}
}

func testVector(dim int, family int) []float32 {
	out := make([]float32, dim)
	block := dim / 4
	for i := 0; i < block; i++ {
		out[i+family*block] = 1
	}
	for i := 0; i < dim; i += 7 {
		out[i] += 0.1
	}
	return out
}
