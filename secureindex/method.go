package secureindex

import (
	"crypto/hkdf"
	"crypto/sha512"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"sort"

	"gonum.org/v1/gonum/graph/community"
	"gonum.org/v1/gonum/graph/simple"
	"gonum.org/v1/gonum/mat"
)

const (
	kptMethodName           = "keyed_wave_superpose_embedding_v0"
	defaultModes            = 8
	defaultDocTopK          = 3
	defaultQueryTopK        = 1
	defaultRouteTemperature = 0.22
	defaultRouteScale       = 1.25
	defaultCollapseGain     = 2.2
	defaultPhaseScale       = 0.78
	defaultEnvelopeGain     = 0.45
	defaultDecoyFloor       = 0.24
	defaultCoherenceWeight  = 0.46
	defaultPublicRatio      = 0.18
	defaultPublicMask       = 0.84
	defaultPublicChunk      = 6
)

// State is the stored keyed representation for one embedding.
type State struct {
	Public       []float32
	BaseWaveReal []float32
	BaseWaveImag []float32
	WaveReal     []float32
	WaveImag     []float32
	ModeWeight   []float32
	ModeEnergy   []float32
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

// Method encodes/query-scores vectors in the full KPT state space.
type Method struct {
	key              string
	inputDim         int
	hiddenDim        int
	modes            int
	docTopK          int
	queryTopK        int
	routeTemperature float64
	routeScale       float64
	collapseGain     float64
	phaseScale       float64
	envelopeGain     float64
	decoyFloor       float64
	coherenceWeight  float64
	publicMask       float64
	publicChunk      int
	publicDim        int
	routeBank        [][]float32
	routeBias        []float32
	realProj         [][]float32
	imagProj         [][]float32
	phaseBias        []float32
	modeCos          [][]float32
	modeSin          [][]float32
	publicMix        [][]float32
}

// NewMethod creates a keyed KPT encoder/scorer for the given embedding dimension.
func NewMethod(key string, inputDim int) *Method {
	if inputDim <= 0 {
		panic("secureindex: inputDim must be positive")
	}

	hiddenDim := inputDim
	publicDim := int(float64(inputDim) * defaultPublicRatio)
	if publicDim < 8 {
		publicDim = 8
	}

	r := rand.New(rand.NewSource(methodSeed(kptMethodName, key, inputDim, 0)))
	m := &Method{
		key:              key,
		inputDim:         inputDim,
		hiddenDim:        hiddenDim,
		modes:            defaultModes,
		docTopK:          defaultDocTopK,
		queryTopK:        defaultQueryTopK,
		routeTemperature: defaultRouteTemperature,
		routeScale:       defaultRouteScale,
		collapseGain:     defaultCollapseGain,
		phaseScale:       defaultPhaseScale,
		envelopeGain:     defaultEnvelopeGain,
		decoyFloor:       defaultDecoyFloor,
		coherenceWeight:  defaultCoherenceWeight,
		publicMask:       defaultPublicMask,
		publicChunk:      defaultPublicChunk,
		publicDim:        publicDim,
		routeBank:        randomMatrix(r, defaultModes, hiddenDim),
		routeBias:        randomUniformVector(r, defaultModes, -0.35, 0.35),
		realProj:         randomOrthogonalMatrix(r, inputDim),
		imagProj:         randomOrthogonalMatrix(r, inputDim),
		phaseBias:        randomUniformVector(r, hiddenDim, 0, 2*math.Pi),
		modeCos:          make([][]float32, defaultModes),
		modeSin:          make([][]float32, defaultModes),
		publicMix:        randomGaussianMatrix(r, defaultModes, publicDim),
	}
	for mode := 0; mode < m.modes; mode++ {
		phaseShift := randomUniformVector(r, hiddenDim, -math.Pi, math.Pi)
		m.modeCos[mode] = make([]float32, hiddenDim)
		m.modeSin[mode] = make([]float32, hiddenDim)
		for i := 0; i < hiddenDim; i++ {
			m.modeCos[mode][i] = float32(math.Cos(float64(phaseShift[i])))
			m.modeSin[mode][i] = float32(math.Sin(float64(phaseShift[i])))
		}
	}
	return m
}

// EncodeDoc encodes one document embedding into its stored KPT state.
func (m *Method) EncodeDoc(vec []float32) State {
	return m.encode(vec, m.docTopK, m.routeScale)
}

// EncodeQuery encodes one query embedding in the same keyed KPT state.
func (m *Method) EncodeQuery(vec []float32) State {
	return m.encode(vec, m.queryTopK, m.routeScale*m.collapseGain)
}

// ScoreDocAgainstQuery scores one stored doc state against an in-memory query state.
func (m *Method) ScoreDocAgainstQuery(doc, query State) SearchHit {
	score, coherence, baseOverlap, modeSupport, energy := m.scorePair(doc, query)
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
	score, _, _, _, _ := m.scorePair(a, b)
	return score
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

	type neighbor struct {
		index int
		score float64
	}

	for i := range docs {
		neighbors := make([]neighbor, 0, len(docs)-1)
		for j := range docs {
			if i == j {
				continue
			}
			score := m.ScoreDocs(docs[i], docs[j])
			if score < minScore {
				continue
			}
			neighbors = append(neighbors, neighbor{index: j, score: score})
		}
		sort.Slice(neighbors, func(a, b int) bool {
			return neighbors[a].score > neighbors[b].score
		})
		if len(neighbors) > k {
			neighbors = neighbors[:k]
		}
		for _, nb := range neighbors {
			g.SetWeightedEdge(g.NewWeightedEdge(simple.Node(i), simple.Node(nb.index), nb.score))
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

func (m *Method) encode(vec []float32, topK int, scale float64) State {
	mustDim(vec, m.inputDim)

	y := normalize(vec)
	carrierReal := project(y, m.realProj)
	carrierImag := project(y, m.imagProj)
	baseWaveReal := make([]float32, m.hiddenDim)
	baseWaveImag := make([]float32, m.hiddenDim)
	for i := 0; i < m.hiddenDim; i++ {
		phase := m.phaseScale*float64(carrierReal[i]) + float64(m.phaseBias[i])
		amp := math.Sqrt(math.Max(1e-6, 1.0+m.envelopeGain*math.Tanh(float64(carrierImag[i]))))
		baseWaveReal[i] = float32(amp * math.Cos(phase))
		baseWaveImag[i] = float32(amp * math.Sin(phase))
	}
	baseWaveReal, baseWaveImag = normalizeComplex(baseWaveReal, baseWaveImag)

	carrierEnergy := make([]float32, m.hiddenDim)
	for i := 0; i < m.hiddenDim; i++ {
		carrierEnergy[i] = baseWaveReal[i]*baseWaveReal[i] + baseWaveImag[i]*baseWaveImag[i]
	}

	routeLogits := make([]float32, m.modes)
	for mode := 0; mode < m.modes; mode++ {
		routeLogits[mode] = dot32(carrierEnergy, m.routeBank[mode]) + m.routeBias[mode]
		routeLogits[mode] /= float32(math.Sqrt(math.Max(1.0, float64(m.hiddenDim))))
	}

	sparseWeight := topkSoftAssign(routeLogits, topK, m.routeTemperature, scale)
	modeWeight := make([]float32, m.modes)
	baseFloor := float32(m.decoyFloor / float64(m.modes))
	for mode := 0; mode < m.modes; mode++ {
		modeWeight[mode] = baseFloor + float32(1.0-m.decoyFloor)*sparseWeight[mode]
	}
	modeWeight = normalizeL1(modeWeight)

	waveReal := make([]float32, m.modes*m.hiddenDim)
	waveImag := make([]float32, m.modes*m.hiddenDim)
	for mode := 0; mode < m.modes; mode++ {
		modeScale := float32(math.Sqrt(math.Max(float64(modeWeight[mode]), 1e-6)))
		offset := mode * m.hiddenDim
		for i := 0; i < m.hiddenDim; i++ {
			rotatedReal := baseWaveReal[i]*m.modeCos[mode][i] - baseWaveImag[i]*m.modeSin[mode][i]
			rotatedImag := baseWaveReal[i]*m.modeSin[mode][i] + baseWaveImag[i]*m.modeCos[mode][i]
			waveReal[offset+i] = modeScale * rotatedReal
			waveImag[offset+i] = modeScale * rotatedImag
		}
	}
	waveReal, waveImag = normalizeComplex(waveReal, waveImag)

	modeEnergy := make([]float32, m.modes)
	publicInput := make([]float32, m.modes)
	for mode := 0; mode < m.modes; mode++ {
		positive := routeLogits[mode]
		if positive < 0 {
			positive = 0
		}
		modeEnergy[mode] = modeWeight[mode] * positive
		publicInput[mode] = float32(math.Tanh(float64(modeWeight[mode] + 0.35*modeEnergy[mode])))
	}

	public := make([]float32, m.publicDim)
	for mode := 0; mode < m.modes; mode++ {
		for j := 0; j < m.publicDim; j++ {
			public[j] += publicInput[mode] * m.publicMix[mode][j]
		}
	}
	public = normalize(public)
	public = m.maskPublicObservation(public)

	state := State{
		Public:       public,
		BaseWaveReal: append([]float32(nil), baseWaveReal...),
		BaseWaveImag: append([]float32(nil), baseWaveImag...),
		WaveReal:     append([]float32(nil), waveReal...),
		WaveImag:     append([]float32(nil), waveImag...),
		ModeWeight:   append([]float32(nil), modeWeight...),
		ModeEnergy:   append([]float32(nil), modeEnergy...),
	}
	return m.scrambleWaves(state)
}

func (m *Method) scorePair(doc, query State) (float64, float64, float64, float64, float64) {
	mustStoredShape(doc)
	mustStoredShape(query)

	docClear := m.unscrambleWaves(doc)
	queryClear := m.unscrambleWaves(query)
	baseOverlap := dot(docClear.BaseWaveReal, queryClear.BaseWaveReal) + dot(docClear.BaseWaveImag, queryClear.BaseWaveImag)

	coherenceTotal := 0.0
	energyTotal := 0.0
	modeSupport := dot(doc.ModeWeight, query.ModeWeight)
	for mode := 0; mode < m.modes; mode++ {
		offset := mode * m.hiddenDim
		componentOverlap := dot(docClear.WaveReal[offset:offset+m.hiddenDim], queryClear.WaveReal[offset:offset+m.hiddenDim])
		componentOverlap += dot(docClear.WaveImag[offset:offset+m.hiddenDim], queryClear.WaveImag[offset:offset+m.hiddenDim])

		modeJoint := float64(doc.ModeWeight[mode] * query.ModeWeight[mode])
		if modeJoint < 0 {
			modeJoint = 0
		}
		modeGate := math.Sqrt(modeJoint)

		energyJoint := float64(doc.ModeEnergy[mode] * query.ModeEnergy[mode])
		if energyJoint < 0 {
			energyJoint = 0
		}
		energyGate := math.Sqrt(energyJoint)

		coherenceTotal += (componentOverlap * componentOverlap) * (0.30 + 0.70*modeGate)
		energyTotal += energyGate
	}

	coherence := coherenceTotal / float64(m.modes)
	energy := energyTotal / float64(m.modes)
	superposed := 0.80*coherence + 0.10*energy + 0.10*modeSupport
	score := m.coherenceWeight*superposed + (1.0-m.coherenceWeight)*(baseOverlap*baseOverlap)
	return score, coherence, baseOverlap, modeSupport, energy
}

func (m *Method) scrambleWaves(state State) State {
	wavePerm, waveSigns, _ := m.scrambleFromModeWeight(state.ModeWeight, "kpt-scramble-wave", len(state.WaveReal))
	basePerm, baseSigns, _ := m.scrambleFromModeWeight(state.ModeWeight, "kpt-scramble-base", len(state.BaseWaveReal))
	state.WaveReal = applyScramble(state.WaveReal, wavePerm, waveSigns)
	state.WaveImag = applyScramble(state.WaveImag, wavePerm, waveSigns)
	state.BaseWaveReal = applyScramble(state.BaseWaveReal, basePerm, baseSigns)
	state.BaseWaveImag = applyScramble(state.BaseWaveImag, basePerm, baseSigns)
	return state
}

func (m *Method) unscrambleWaves(state State) State {
	wavePerm, waveSigns, waveInverse := m.scrambleFromModeWeight(state.ModeWeight, "kpt-scramble-wave", len(state.WaveReal))
	basePerm, baseSigns, baseInverse := m.scrambleFromModeWeight(state.ModeWeight, "kpt-scramble-base", len(state.BaseWaveReal))
	_ = wavePerm
	_ = basePerm
	state.WaveReal = undoScramble(state.WaveReal, waveSigns, waveInverse)
	state.WaveImag = undoScramble(state.WaveImag, waveSigns, waveInverse)
	state.BaseWaveReal = undoScramble(state.BaseWaveReal, baseSigns, baseInverse)
	state.BaseWaveImag = undoScramble(state.BaseWaveImag, baseSigns, baseInverse)
	return state
}

func (m *Method) scrambleFromModeWeight(modeWeight []float32, salt string, size int) ([]int, []float32, []int) {
	seed := scrambleSeed(m.key, modeWeight, salt)
	r := rand.New(rand.NewSource(seed))
	perm := r.Perm(size)
	signs := make([]float32, size)
	for i := 0; i < size; i++ {
		signs[i] = 1.0
		if r.Intn(2) == 0 {
			signs[i] = -1.0
		}
	}
	inverse := make([]int, size)
	for idx, src := range perm {
		inverse[src] = idx
	}
	return perm, signs, inverse
}

func (m *Method) maskPublicObservation(public []float32) []float32 {
	if len(public) == 0 {
		return nil
	}
	mixRNG := rand.New(rand.NewSource(methodSeed(kptMethodName+"_public", m.key, len(public), 0)))
	mix := randomOrthogonalMatrix(mixRNG, len(public))
	mixed := project(public, mix)

	driftStd := 0.03 * m.publicMask
	if driftStd < 1e-4 {
		driftStd = 1e-4
	}
	scale := 1.0 + 1.8*m.publicMask
	for i := range mixed {
		mixed[i] += float32(mixRNG.NormFloat64() * driftStd)
		mixed[i] = float32(math.Tanh(float64(mixed[i]) * scale))
	}

	if m.publicChunk > 1 {
		useDim := (len(mixed) / m.publicChunk) * m.publicChunk
		if useDim > 0 {
			pooled := make([]float32, 0, useDim/m.publicChunk+len(mixed)-useDim)
			for offset := 0; offset < useDim; offset += m.publicChunk {
				total := float32(0)
				for _, value := range mixed[offset : offset+m.publicChunk] {
					total += value
				}
				pooled = append(pooled, total/float32(m.publicChunk))
			}
			pooled = append(pooled, mixed[useDim:]...)
			mixed = pooled
		}
	}

	return normalize(mixed)
}

func methodSeed(methodName, secretKey string, dim, offset int) int64 {
	info := fmt.Sprintf("%s|%d|%d", methodName, dim, offset)
	material, err := hkdf.Key(sha512.New, []byte(secretKey), []byte("kpt-v1"), info, 64)
	if err != nil {
		panic(fmt.Sprintf("secureindex: hkdf method seed failed: %v", err))
	}
	return int64(binary.BigEndian.Uint64(material[:8]))
}

func scrambleSeed(secretKey string, modeWeight []float32, salt string) int64 {
	ikm := append([]byte(secretKey), encodeFloat32Bytes(modeWeight)...)
	material, err := hkdf.Key(sha512.New, ikm, []byte(salt), "", 64)
	if err != nil {
		panic(fmt.Sprintf("secureindex: hkdf scramble seed failed: %v", err))
	}
	return int64(binary.BigEndian.Uint64(material[:8]))
}

func encodeFloat32Bytes(values []float32) []byte {
	buf := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(value))
	}
	return buf
}

func randomOrthogonalMatrix(r *rand.Rand, dim int) [][]float32 {
	data := make([]float64, dim*dim)
	for i := range data {
		data[i] = r.NormFloat64()
	}
	input := mat.NewDense(dim, dim, data)
	var qr mat.QR
	qr.Factorize(input)
	var q mat.Dense
	qr.QTo(&q)
	rows, cols := q.Dims()
	out := make([][]float32, rows)
	for i := 0; i < rows; i++ {
		row := make([]float32, cols)
		for j := 0; j < cols; j++ {
			row[j] = float32(q.At(i, j))
		}
		out[i] = row
	}
	return out
}

func randomGaussianMatrix(r *rand.Rand, rows, cols int) [][]float32 {
	out := make([][]float32, rows)
	for i := 0; i < rows; i++ {
		row := make([]float32, cols)
		for j := 0; j < cols; j++ {
			row[j] = float32(r.NormFloat64())
		}
		out[i] = row
	}
	return out
}

func randomMatrix(r *rand.Rand, rows, cols int) [][]float32 {
	out := randomGaussianMatrix(r, rows, cols)
	for i := range out {
		out[i] = normalize(out[i])
	}
	return out
}

func randomUniformVector(r *rand.Rand, dim int, lo, hi float64) []float32 {
	out := make([]float32, dim)
	for i := range out {
		out[i] = float32(lo + r.Float64()*(hi-lo))
	}
	return out
}

func project(vec []float32, matRows [][]float32) []float32 {
	if len(matRows) == 0 {
		return nil
	}
	out := make([]float32, len(matRows[0]))
	for i, weight := range vec {
		row := matRows[i]
		for j := range out {
			out[j] += weight * row[j]
		}
	}
	return out
}

func topkSoftAssign(logits []float32, topK int, temperature float64, scale float64) []float32 {
	if len(logits) == 0 {
		return nil
	}
	if topK <= 0 || topK > len(logits) {
		topK = len(logits)
	}
	type pair struct {
		index int
		value float32
	}
	pairs := make([]pair, len(logits))
	for i, value := range logits {
		pairs[i] = pair{index: i, value: float32(float64(value) * scale)}
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].value > pairs[j].value
	})
	pairs = pairs[:topK]

	maxValue := pairs[0].value
	out := make([]float32, len(logits))
	total := float64(0)
	denom := temperature
	if denom < 1e-4 {
		denom = 1e-4
	}
	for _, pair := range pairs {
		weight := math.Exp(float64(pair.value-maxValue) / denom)
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

func normalize(values []float32) []float32 {
	out := append([]float32(nil), values...)
	sum := 0.0
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

func normalizeComplex(realPart, imagPart []float32) ([]float32, []float32) {
	realOut := append([]float32(nil), realPart...)
	imagOut := append([]float32(nil), imagPart...)
	sum := 0.0
	for i := range realOut {
		sum += float64(realOut[i]*realOut[i] + imagOut[i]*imagOut[i])
	}
	if sum == 0 {
		return realOut, imagOut
	}
	norm := float32(math.Sqrt(sum))
	for i := range realOut {
		realOut[i] /= norm
		imagOut[i] /= norm
	}
	return realOut, imagOut
}

func normalizeL1(values []float32) []float32 {
	out := append([]float32(nil), values...)
	total := float32(0)
	for _, value := range out {
		total += value
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

func undoScramble(values []float32, signs []float32, inverse []int) []float32 {
	out := make([]float32, len(values))
	for src, idx := range inverse {
		out[src] = values[idx] * signs[idx]
	}
	return out
}

func dot(a, b []float32) float64 {
	limit := len(a)
	if len(b) < limit {
		limit = len(b)
	}
	total := 0.0
	for i := 0; i < limit; i++ {
		total += float64(a[i] * b[i])
	}
	return total
}

func dot32(a, b []float32) float32 {
	limit := len(a)
	if len(b) < limit {
		limit = len(b)
	}
	total := float32(0)
	for i := 0; i < limit; i++ {
		total += a[i] * b[i]
	}
	return total
}

func mustDim(vec []float32, want int) {
	if len(vec) != want {
		panic(fmt.Sprintf("secureindex: got dim %d, want %d", len(vec), want))
	}
}

func mustStoredShape(state State) {
	if len(state.BaseWaveReal) == 0 || len(state.BaseWaveImag) == 0 {
		panic("secureindex: state is missing base waves")
	}
	if len(state.WaveReal) == 0 || len(state.WaveImag) == 0 {
		panic("secureindex: state is missing mode waves")
	}
	if len(state.ModeWeight) == 0 || len(state.ModeEnergy) == 0 {
		panic("secureindex: state is missing routing state")
	}
}
