package secureindex

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"sort"

	"gonum.org/v1/gonum/graph/community"
	"gonum.org/v1/gonum/graph/simple"
)

const (
	defaultModes       = 8
	defaultTopK        = 3
	defaultPublicStats = 6
	defaultAlpha       = 0.46
)

// State is the stored keyed representation for one embedding.
type State struct {
	Public     []float32
	Base       []float32
	WaveReal   []float32
	WaveImag   []float32
	ModeWeight []float32
	ModeEnergy []float32
	KeyProbe   []float32
}

// SearchHit is one ranked keyed-search result.
type SearchHit struct {
	Index       int
	Score       float64
	Coherence   float64
	BaseOverlap float64
	ModeSupport float64
	Energy      float64
}

// Cluster groups document indices in the authorized keyed graph.
type Cluster struct {
	ID      int
	Members []int
}

// Method encodes/query-scores vectors in a keyed wave-style space.
type Method struct {
	key        string
	inputDim   int
	hiddenDim  int
	modes      int
	topK       int
	alpha      float64
	carrierMat [][]float32
	phaseMat   [][]float32
	routerMat  [][]float32
	phaseShift []float64
	gateBase   []float32
	gateReal   []float32
	gateImag   []float32
	keyProbe   []float32
}

// NewMethod creates a keyed encoder/scorer for the given embedding dimension.
func NewMethod(key string, inputDim int) *Method {
	if inputDim <= 0 {
		panic("secureindex: inputDim must be positive")
	}

	hiddenDim := inputDim
	if hiddenDim > 256 {
		hiddenDim = 256
	}

	r := rand.New(rand.NewSource(seedFromKey(key, "method", inputDim)))
	m := &Method{
		key:        key,
		inputDim:   inputDim,
		hiddenDim:  hiddenDim,
		modes:      defaultModes,
		topK:       defaultTopK,
		alpha:      defaultAlpha,
		carrierMat: randomMatrix(r, hiddenDim, inputDim),
		phaseMat:   randomMatrix(r, hiddenDim, inputDim),
		routerMat:  randomMatrix(r, defaultModes, inputDim),
		phaseShift: make([]float64, defaultModes),
		gateBase:   randomVector(r, hiddenDim),
		gateReal:   randomVector(r, hiddenDim),
		gateImag:   randomVector(r, hiddenDim),
		keyProbe:   randomVector(r, 16),
	}
	for i := range m.phaseShift {
		m.phaseShift[i] = (r.Float64()*2 - 1) * math.Pi
	}
	return m
}

// EncodeDoc encodes one document embedding into its stored keyed state.
func (m *Method) EncodeDoc(vec []float32) State {
	mustDim(vec, m.inputDim)

	basePlain := normalize(blend(matVec(m.carrierMat, vec), m.gateBase, 0.72, 0.28))
	phaseCarrier := matVec(m.phaseMat, vec)
	modeLogits := matVec(m.routerMat, vec)
	modeWeight := sparseSoftmax(modeLogits, m.topK)
	modeEnergy := normalizedAbs(modeLogits)
	phaseMix := weightedPhase(m.phaseShift, modeWeight)

	waveReal := make([]float32, m.hiddenDim)
	waveImag := make([]float32, m.hiddenDim)
	for i := 0; i < m.hiddenDim; i++ {
		amp := float64(float32(math.Tanh(float64(basePlain[i]))))
		theta := float64(phaseCarrier[i]) + phaseMix
		waveReal[i] = float32(amp * math.Cos(theta))
		waveImag[i] = float32(amp * math.Sin(theta))
	}
	waveReal = normalize(blend(waveReal, m.gateReal, 0.45, 0.55))
	waveImag = normalize(blend(waveImag, m.gateImag, 0.45, 0.55))

	perm, signs := m.scramble(modeWeight, m.hiddenDim)
	baseStored := applyScramble(basePlain, perm, signs)
	waveRealStored := applyScramble(waveReal, perm, signs)
	waveImagStored := applyScramble(waveImag, perm, signs)

	public := buildPublic(modeWeight, basePlain)
	return State{
		Public:     public,
		Base:       baseStored,
		WaveReal:   waveRealStored,
		WaveImag:   waveImagStored,
		ModeWeight: modeWeight,
		ModeEnergy: modeEnergy,
		KeyProbe:   append([]float32(nil), m.keyProbe...),
	}
}

// EncodeQuery encodes one query embedding in the same keyed space without storage scrambling.
func (m *Method) EncodeQuery(vec []float32) State {
	mustDim(vec, m.inputDim)

	base := normalize(blend(matVec(m.carrierMat, vec), m.gateBase, 0.72, 0.28))
	phaseCarrier := matVec(m.phaseMat, vec)
	modeLogits := matVec(m.routerMat, vec)
	modeWeight := sparseSoftmax(modeLogits, m.topK)
	modeEnergy := normalizedAbs(modeLogits)
	phaseMix := weightedPhase(m.phaseShift, modeWeight)

	waveReal := make([]float32, m.hiddenDim)
	waveImag := make([]float32, m.hiddenDim)
	for i := 0; i < m.hiddenDim; i++ {
		amp := float64(float32(math.Tanh(float64(base[i]))))
		theta := float64(phaseCarrier[i]) + phaseMix
		waveReal[i] = float32(amp * math.Cos(theta))
		waveImag[i] = float32(amp * math.Sin(theta))
	}
	return State{
		Public:     buildPublic(modeWeight, base),
		Base:       normalize(base),
		WaveReal:   normalize(blend(waveReal, m.gateReal, 0.45, 0.55)),
		WaveImag:   normalize(blend(waveImag, m.gateImag, 0.45, 0.55)),
		ModeWeight: modeWeight,
		ModeEnergy: modeEnergy,
		KeyProbe:   append([]float32(nil), m.keyProbe...),
	}
}

// ScoreDocAgainstQuery scores one stored doc state against an in-memory query state.
func (m *Method) ScoreDocAgainstQuery(doc, query State) SearchHit {
	mustStoredShape(doc)
	mustStoredShape(query)

	docBase, docReal, docImag := m.unscramble(doc)

	coherence := clamp01((dot(docReal, query.WaveReal) + dot(docImag, query.WaveImag) + 1.0) / 2.0)
	baseOverlap := clamp01((dot(docBase, query.Base) + 1.0) / 2.0)
	modeSupport := clamp01(dot(doc.ModeWeight, query.ModeWeight))
	energy := clamp01(1.0 - meanAbsDiff(doc.ModeEnergy, query.ModeEnergy))
	keyAgreement := math.Pow(clamp01((dot(doc.KeyProbe, query.KeyProbe)+1.0)/2.0), 8)
	score := keyAgreement * (m.alpha*(0.80*coherence+0.10*energy+0.10*modeSupport) + (1.0-m.alpha)*baseOverlap*baseOverlap)

	return SearchHit{
		Score:       score,
		Coherence:   coherence,
		BaseOverlap: baseOverlap,
		ModeSupport: modeSupport,
		Energy:      energy,
	}
}

// ScoreDocs scores two stored keyed states against each other.
func (m *Method) ScoreDocs(a, b State) float64 {
	mustStoredShape(a)
	mustStoredShape(b)

	aBase, aReal, aImag := m.unscramble(a)
	bBase, bReal, bImag := m.unscramble(b)

	coherence := clamp01((dot(aReal, bReal) + dot(aImag, bImag) + 1.0) / 2.0)
	baseOverlap := clamp01((dot(aBase, bBase) + 1.0) / 2.0)
	modeSupport := clamp01(dot(a.ModeWeight, b.ModeWeight))
	energy := clamp01(1.0 - meanAbsDiff(a.ModeEnergy, b.ModeEnergy))
	keyAgreement := math.Pow(clamp01((dot(a.KeyProbe, b.KeyProbe)+1.0)/2.0), 8)
	return keyAgreement * (m.alpha*(0.80*coherence+0.10*energy+0.10*modeSupport) + (1.0-m.alpha)*baseOverlap*baseOverlap)
}

// Search ranks stored docs for a keyed query.
func (m *Method) Search(docs []State, query State, limit int) []SearchHit {
	if limit <= 0 || limit > len(docs) {
		limit = len(docs)
	}
	hits := make([]SearchHit, 0, len(docs))
	for i, doc := range docs {
		hit := m.ScoreDocAgainstQuery(doc, query)
		hit.Index = i
		hits = append(hits, hit)
	}
	sort.Slice(hits, func(i, j int) bool {
		return hits[i].Score > hits[j].Score
	})
	return hits[:limit]
}

// Cluster builds a keyed kNN graph over stored docs and runs Louvain community detection.
func (m *Method) Cluster(docs []State, k int, resolution float64, minScore float64) []Cluster {
	if len(docs) == 0 {
		return nil
	}
	if k <= 0 {
		k = 10
	}
	if resolution <= 0 {
		resolution = 1.0
	}

	g := simple.NewWeightedUndirectedGraph(0, 0)
	for i := range docs {
		g.AddNode(simple.Node(i))
	}

	type nb struct {
		index int
		score float64
	}
	for i := range docs {
		neighbors := make([]nb, 0, len(docs)-1)
		for j := range docs {
			if i == j {
				continue
			}
			score := m.ScoreDocs(docs[i], docs[j])
			if score < minScore {
				continue
			}
			neighbors = append(neighbors, nb{index: j, score: score})
		}
		sort.Slice(neighbors, func(a, b int) bool {
			return neighbors[a].score > neighbors[b].score
		})
		if len(neighbors) > k {
			neighbors = neighbors[:k]
		}
		for _, neighbor := range neighbors {
			g.SetWeightedEdge(g.NewWeightedEdge(simple.Node(i), simple.Node(neighbor.index), neighbor.score))
		}
	}

	reduced := community.Modularize(g, resolution, nil)
	partitions := reduced.Communities()
	clusters := make([]Cluster, 0, len(partitions))
	for idx, members := range partitions {
		cluster := Cluster{ID: idx, Members: make([]int, 0, len(members))}
		for _, member := range members {
			cluster.Members = append(cluster.Members, int(member.ID()))
		}
		sort.Ints(cluster.Members)
		clusters = append(clusters, cluster)
	}
	sort.Slice(clusters, func(i, j int) bool {
		if len(clusters[i].Members) == len(clusters[j].Members) {
			return clusters[i].ID < clusters[j].ID
		}
		return len(clusters[i].Members) > len(clusters[j].Members)
	})
	return clusters
}

func (m *Method) unscramble(state State) ([]float32, []float32, []float32) {
	perm, signs := m.scramble(state.ModeWeight, len(state.Base))
	return undoScramble(state.Base, perm, signs), undoScramble(state.WaveReal, perm, signs), undoScramble(state.WaveImag, perm, signs)
}

func (m *Method) scramble(modeWeight []float32, n int) ([]int, []float32) {
	seed := seedFromModeWeight(m.key, modeWeight)
	r := rand.New(rand.NewSource(seed))
	perm := r.Perm(n)
	signs := make([]float32, n)
	for i := range signs {
		signs[i] = 1.0
		if r.Intn(2) == 0 {
			signs[i] = -1.0
		}
	}
	return perm, signs
}

func seedFromModeWeight(key string, modeWeight []float32) int64 {
	h := sha256.New()
	_, _ = h.Write([]byte("ultramemory/bregman-v1/"))
	_, _ = h.Write([]byte(key))
	var buf [4]byte
	for _, value := range modeWeight {
		binary.LittleEndian.PutUint32(buf[:], math.Float32bits(value))
		_, _ = h.Write(buf[:])
	}
	sum := h.Sum(nil)
	return int64(binary.LittleEndian.Uint64(sum[:8]))
}

func buildPublic(modeWeight, base []float32) []float32 {
	public := make([]float32, 0, len(modeWeight)+defaultPublicStats)
	public = append(public, modeWeight...)
	chunkSize := int(math.Ceil(float64(len(base)) / float64(defaultPublicStats)))
	for offset := 0; offset < len(base); offset += chunkSize {
		end := offset + chunkSize
		if end > len(base) {
			end = len(base)
		}
		total := float32(0)
		for _, value := range base[offset:end] {
			total += float32(math.Abs(float64(value)))
		}
		public = append(public, total/float32(end-offset))
	}
	for len(public) < len(modeWeight)+defaultPublicStats {
		public = append(public, 0)
	}
	return public
}

func randomMatrix(r *rand.Rand, rows, cols int) [][]float32 {
	out := make([][]float32, rows)
	scale := 1.0 / math.Sqrt(float64(cols))
	for i := range out {
		row := make([]float32, cols)
		norm := 0.0
		for j := range row {
			value := r.NormFloat64() * scale
			row[j] = float32(value)
			norm += value * value
		}
		norm = math.Sqrt(norm)
		if norm == 0 {
			norm = 1
		}
		for j := range row {
			row[j] /= float32(norm)
		}
		out[i] = row
	}
	return out
}

func randomVector(r *rand.Rand, dim int) []float32 {
	out := make([]float32, dim)
	for i := range out {
		out[i] = float32(r.NormFloat64())
	}
	return normalize(out)
}

func matVec(mat [][]float32, vec []float32) []float32 {
	out := make([]float32, len(mat))
	for i, row := range mat {
		sum := float32(0)
		for j, value := range row {
			sum += value * vec[j]
		}
		out[i] = sum
	}
	return out
}

func sparseSoftmax(logits []float32, topK int) []float32 {
	if topK <= 0 || topK > len(logits) {
		topK = len(logits)
	}
	type pair struct {
		index int
		value float32
	}
	pairs := make([]pair, len(logits))
	for i, value := range logits {
		pairs[i] = pair{index: i, value: value}
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].value > pairs[j].value
	})
	pairs = pairs[:topK]

	maxValue := pairs[0].value
	total := float64(0)
	out := make([]float32, len(logits))
	for _, pair := range pairs {
		weight := math.Exp(float64(pair.value - maxValue))
		out[pair.index] = float32(weight)
		total += weight
	}
	if total == 0 {
		return out
	}
	for i := range out {
		out[i] /= float32(total)
	}
	return out
}

func weightedPhase(phaseShift []float64, weight []float32) float64 {
	total := 0.0
	for i, value := range weight {
		total += phaseShift[i] * float64(value)
	}
	return total
}

func normalizedAbs(values []float32) []float32 {
	out := make([]float32, len(values))
	total := float32(0)
	for i, value := range values {
		out[i] = float32(math.Abs(float64(value)))
		total += out[i]
	}
	if total == 0 {
		return out
	}
	for i := range out {
		out[i] /= total
	}
	return out
}

func applyScramble(values []float32, perm []int, signs []float32) []float32 {
	out := make([]float32, len(values))
	for i, src := range perm {
		out[i] = values[src] * signs[i]
	}
	return out
}

func undoScramble(values []float32, perm []int, signs []float32) []float32 {
	out := make([]float32, len(values))
	for i, src := range perm {
		out[src] = values[i] * signs[i]
	}
	return out
}

func normalize(values []float32) []float32 {
	out := append([]float32(nil), values...)
	sum := float64(0)
	for _, value := range out {
		sum += float64(value * value)
	}
	if sum == 0 {
		return out
	}
	norm := float32(math.Sqrt(sum))
	for i := range out {
		out[i] /= norm
	}
	return out
}

func blend(a, b []float32, wa, wb float32) []float32 {
	limit := len(a)
	if len(b) < limit {
		limit = len(b)
	}
	out := make([]float32, limit)
	for i := 0; i < limit; i++ {
		out[i] = wa*a[i] + wb*b[i]
	}
	return out
}

func dot(a, b []float32) float64 {
	sum := float64(0)
	limit := len(a)
	if len(b) < limit {
		limit = len(b)
	}
	for i := 0; i < limit; i++ {
		sum += float64(a[i] * b[i])
	}
	return sum
}

func meanAbsDiff(a, b []float32) float64 {
	limit := len(a)
	if len(b) < limit {
		limit = len(b)
	}
	if limit == 0 {
		return 1
	}
	total := 0.0
	for i := 0; i < limit; i++ {
		total += math.Abs(float64(a[i] - b[i]))
	}
	return total / float64(limit)
}

func clamp01(value float64) float64 {
	if value < 0 {
		return 0
	}
	if value > 1 {
		return 1
	}
	return value
}

func seedFromKey(key, suffix string, dim int) int64 {
	sum := sha256.Sum256([]byte(fmt.Sprintf("ultramemory/%s/%s/%d", key, suffix, dim)))
	return int64(binary.LittleEndian.Uint64(sum[:8]))
}

func mustDim(vec []float32, want int) {
	if len(vec) != want {
		panic(fmt.Sprintf("secureindex: got dim %d, want %d", len(vec), want))
	}
}

func mustStoredShape(state State) {
	if len(state.Base) == 0 || len(state.WaveReal) == 0 || len(state.WaveImag) == 0 {
		panic("secureindex: stored state is missing required vectors")
	}
	if len(state.KeyProbe) == 0 {
		panic("secureindex: stored state is missing key probe")
	}
}
