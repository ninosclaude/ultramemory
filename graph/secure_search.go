package graph

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"sort"
	"strings"

	"github.com/sharpner/ultramemory/llm"
	"github.com/sharpner/ultramemory/secureindex"
	"github.com/sharpner/ultramemory/store"
)

type secureEpisodeHit struct {
	row   store.SecureEpisodeIndexRow
	score float64
}

type secureRankedEntity struct {
	entity store.Entity
	score  float64
}

type secureRankedEdge struct {
	edge  store.Edge
	score float64
}

type secureRankedEpisode struct {
	row   store.SecureEpisodeIndexRow
	score float64
}

type secureRankedResult struct {
	result SearchResult
	score  float64
}

// SecureSearch reuses the old hybrid search roles on top of KPT episodes and
// encrypted graph artifacts. The only thing that changes is the at-rest layer.
func SecureSearch(ctx context.Context, db *store.DB, embedder llm.Embedder, personalKey, groupID, methodName, query string, limit int) ([]SearchResult, error) {
	if limit <= 0 {
		limit = 10
	}

	rows, err := db.AllSecureEpisodes(ctx, groupID, methodName)
	if err != nil {
		return nil, fmt.Errorf("load secure episodes: %w", err)
	}
	if len(rows) == 0 {
		return nil, nil
	}

	decryptedRows, err := decryptSecureEpisodes(rows, personalKey)
	if err != nil {
		return nil, err
	}

	allEpisodeRows := make(map[string]store.SecureEpisodeIndexRow, len(decryptedRows))
	for _, row := range decryptedRows {
		allEpisodeRows[row.EpisodeUUID] = row
	}

	entities, err := db.AllEntities(ctx, groupID)
	if err != nil {
		return nil, fmt.Errorf("load secure entities: %w", err)
	}
	decryptedEntities := decryptEntities(entities, personalKey)

	edges, err := db.AllEdges(ctx, groupID)
	if err != nil {
		return nil, fmt.Errorf("load secure edges: %w", err)
	}
	decryptedEdges := decryptEdges(edges, personalKey)

	queryTokens := tokenizeSecure(query)
	entityLex := secureRankEntities(queryTokens, decryptedEntities, limit*3)
	edgeLex := secureRankEdges(queryTokens, decryptedEdges, limit*3)
	episodeLex := secureRankEpisodes(queryTokens, decryptedRows, limit*3)

	var queryEmb []float32
	queryEmb, err = embedder.Embed(ctx, query)
	if err != nil {
		slog.Warn("secure query embedding failed, falling back to lexical-only search", "err", err)
	}

	episodeKPT := secureRankKPTEpisodes(personalKey, queryEmb, decryptedRows, limit*3)
	episodeScore := make(map[string]float64, len(episodeKPT))
	for _, hit := range episodeKPT {
		episodeScore[hit.row.EpisodeUUID] = hit.score
	}

	episodeSeedUUIDs := secureTopEpisodeUUIDs(episodeLex, episodeKPT, 5)
	topEpisodeEntities, err := db.EntitiesForEpisodes(ctx, episodeSeedUUIDs, groupID)
	if err != nil {
		return nil, fmt.Errorf("load secure episode entities: %w", err)
	}
	topEpisodeEntities = decryptEntities(topEpisodeEntities, personalKey)

	seeds := secureMAGMASeeds(entityLex, topEpisodeEntities, 5)
	var magmaRanked []ActivatedNode
	if len(seeds) > 0 {
		magmaRanked, err = SpreadMAGMA(ctx, db, seeds, query, queryEmb, groupID, DefaultMAGMAConfig())
		if err != nil {
			slog.Warn("secure MAGMA failed", "err", err)
		}
	}

	communityMap, err := db.CommunityMap(ctx, groupID)
	if err != nil {
		return nil, fmt.Errorf("load community map: %w", err)
	}
	seedCommunities := secureSeedCommunities(seeds, magmaRanked, communityMap)

	rrf := secureRRF(entityLex, edgeLex, episodeLex, episodeKPT, magmaRanked)
	secureApplyCommunityBoost(rrf, decryptedEdges, communityMap, seedCommunities)

	results, err := secureBuildResults(ctx, db, personalKey, groupID, rrf, decryptedEdges, allEpisodeRows, seedCommunities, limit)
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, nil
	}

	results = secureApplyRelevanceCutoff(results)
	results = secureApplyUncertaintyGate(results)
	if len(results) == 0 {
		return nil, nil
	}

	secureApplyTemporalDecay(results)
	sort.Slice(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	if len(results) > limit {
		results = results[:limit]
	}

	return results, nil
}

func decryptSecureEpisodes(rows []store.SecureEpisodeIndexRow, personalKey string) ([]store.SecureEpisodeIndexRow, error) {
	out := make([]store.SecureEpisodeIndexRow, 0, len(rows))
	for _, row := range rows {
		content, err := secureindex.DecryptString(personalKey, "episode-content", row.Content)
		if err != nil {
			row.Content = "<entschlüsselung fehlgeschlagen>"
		}
		if err == nil {
			row.Content = content
		}
		source, err := secureindex.DecryptString(personalKey, "episode-source", row.Source)
		if err != nil {
			row.Source = "<locked>"
		}
		if err == nil {
			row.Source = source
		}
		out = append(out, row)
	}
	return out, nil
}

func secureStates(rows []store.SecureEpisodeIndexRow) []secureindex.State {
	states := make([]secureindex.State, 0, len(rows))
	for _, row := range rows {
		states = append(states, secureindex.State{
			Public:       row.Public,
			BaseWaveReal: row.BaseWaveReal,
			BaseWaveImag: row.BaseWaveImag,
			WaveReal:     row.WaveReal,
			WaveImag:     row.WaveImag,
			ModeWeight:   row.ModeWeight,
			ModeEnergy:   row.ModeEnergy,
		})
	}
	return states
}

func decryptEntities(entities []store.Entity, personalKey string) []store.Entity {
	out := make([]store.Entity, 0, len(entities))
	for _, entity := range entities {
		name, err := secureindex.DecryptString(personalKey, "entity-name", entity.Name)
		if err != nil {
			entity.Name = "<locked>"
		}
		if err == nil {
			entity.Name = name
		}
		description, err := secureindex.DecryptString(personalKey, "entity-description", entity.Description)
		if err != nil {
			entity.Description = "<locked>"
		}
		if err == nil {
			entity.Description = description
		}
		out = append(out, entity)
	}
	return out
}

func decryptEdges(edges []store.Edge, personalKey string) []store.Edge {
	out := make([]store.Edge, 0, len(edges))
	for _, edge := range edges {
		name, err := secureindex.DecryptString(personalKey, "edge-name", edge.Name)
		if err != nil {
			edge.Name = "<locked>"
		}
		if err == nil {
			edge.Name = name
		}
		fact, err := secureindex.DecryptString(personalKey, "edge-fact", edge.Fact)
		if err != nil {
			edge.Fact = "<locked>"
		}
		if err == nil {
			edge.Fact = fact
		}
		out = append(out, edge)
	}
	return out
}

func tokenizeSecure(text string) []string {
	raw := strings.Fields(strings.ToLower(text))
	if len(raw) == 0 {
		return nil
	}
	seen := make(map[string]bool, len(raw))
	var out []string
	for _, token := range raw {
		token = strings.Trim(token, ".,:;!?()[]{}\"'")
		if token == "" {
			continue
		}
		if seen[token] {
			continue
		}
		seen[token] = true
		out = append(out, token)
	}
	return out
}

func secureLexicalScore(queryTokens []string, parts ...string) float64 {
	if len(queryTokens) == 0 {
		return 0
	}
	targetTokens := tokenizeSecure(strings.Join(parts, " "))
	if len(targetTokens) == 0 {
		return 0
	}
	targetSet := make(map[string]bool, len(targetTokens))
	for _, token := range targetTokens {
		targetSet[token] = true
	}
	matches := 0
	for _, token := range queryTokens {
		if targetSet[token] {
			matches++
			continue
		}
		for candidate := range targetSet {
			if strings.Contains(candidate, token) {
				matches++
				break
			}
			if strings.Contains(token, candidate) {
				matches++
				break
			}
		}
	}
	if matches == 0 {
		return 0
	}
	return float64(matches) / float64(max(len(queryTokens), len(targetTokens)))
}

func secureRankEntities(queryTokens []string, entities []store.Entity, limit int) []secureRankedEntity {
	ranked := make([]secureRankedEntity, 0, len(entities))
	for _, entity := range entities {
		score := secureLexicalScore(queryTokens, entity.Name, entity.Description)
		if score <= 0 {
			continue
		}
		ranked = append(ranked, secureRankedEntity{entity: entity, score: score})
	}
	sort.Slice(ranked, func(i, j int) bool { return ranked[i].score > ranked[j].score })
	if len(ranked) > limit {
		ranked = ranked[:limit]
	}
	return ranked
}

func secureRankEdges(queryTokens []string, edges []store.Edge, limit int) []secureRankedEdge {
	ranked := make([]secureRankedEdge, 0, len(edges))
	for _, edge := range edges {
		score := secureLexicalScore(queryTokens, edge.Name, edge.Fact)
		if score <= 0 {
			continue
		}
		ranked = append(ranked, secureRankedEdge{edge: edge, score: score})
	}
	sort.Slice(ranked, func(i, j int) bool { return ranked[i].score > ranked[j].score })
	if len(ranked) > limit {
		ranked = ranked[:limit]
	}
	return ranked
}

func secureRankEpisodes(queryTokens []string, rows []store.SecureEpisodeIndexRow, limit int) []secureRankedEpisode {
	ranked := make([]secureRankedEpisode, 0, len(rows))
	for _, row := range rows {
		score := secureLexicalScore(queryTokens, row.Source, row.Content)
		if score <= 0 {
			continue
		}
		ranked = append(ranked, secureRankedEpisode{row: row, score: score})
	}
	sort.Slice(ranked, func(i, j int) bool { return ranked[i].score > ranked[j].score })
	if len(ranked) > limit {
		ranked = ranked[:limit]
	}
	return ranked
}

func secureRankKPTEpisodes(personalKey string, queryEmb []float32, rows []store.SecureEpisodeIndexRow, limit int) []secureEpisodeHit {
	if len(queryEmb) == 0 {
		return nil
	}
	method := secureindex.NewMethod(personalKey, len(queryEmb))
	queryState := method.EncodeQuery(queryEmb)
	searchHits := method.Search(secureStates(rows), queryState, min(limit, len(rows)))
	hits := make([]secureEpisodeHit, 0, len(searchHits))
	for _, hit := range searchHits {
		row := rows[hit.Index]
		hits = append(hits, secureEpisodeHit{row: row, score: hit.Score})
	}
	return hits
}

func secureTopEpisodeUUIDs(lex []secureRankedEpisode, kpt []secureEpisodeHit, limit int) []string {
	seen := make(map[string]bool)
	var out []string
	for _, item := range lex {
		if seen[item.row.EpisodeUUID] {
			continue
		}
		seen[item.row.EpisodeUUID] = true
		out = append(out, item.row.EpisodeUUID)
		if len(out) >= limit {
			return out
		}
	}
	for _, item := range kpt {
		if seen[item.row.EpisodeUUID] {
			continue
		}
		seen[item.row.EpisodeUUID] = true
		out = append(out, item.row.EpisodeUUID)
		if len(out) >= limit {
			return out
		}
	}
	return out
}

func secureMAGMASeeds(entityLex []secureRankedEntity, linked []store.Entity, limit int) []ActivatedNode {
	seen := make(map[string]bool)
	var out []ActivatedNode
	for _, item := range entityLex {
		if seen[item.entity.UUID] {
			continue
		}
		seen[item.entity.UUID] = true
		out = append(out, ActivatedNode{
			UUID:       item.entity.UUID,
			Name:       item.entity.Name,
			EntityType: item.entity.EntityType,
		})
		if len(out) >= limit {
			return out
		}
	}
	for _, entity := range linked {
		if seen[entity.UUID] {
			continue
		}
		seen[entity.UUID] = true
		out = append(out, ActivatedNode{
			UUID:       entity.UUID,
			Name:       entity.Name,
			EntityType: entity.EntityType,
		})
		if len(out) >= limit {
			return out
		}
	}
	return out
}

func secureSeedCommunities(seeds []ActivatedNode, magma []ActivatedNode, communityMap map[string]int) map[int]bool {
	out := make(map[int]bool)
	for _, seed := range seeds {
		cid, ok := communityMap[seed.UUID]
		if !ok {
			continue
		}
		out[cid] = true
	}
	for i, node := range magma {
		if i >= 5 {
			break
		}
		cid, ok := communityMap[node.UUID]
		if !ok {
			continue
		}
		out[cid] = true
	}
	return out
}

func secureRRF(entityLex []secureRankedEntity, edgeLex []secureRankedEdge, episodeLex []secureRankedEpisode, episodeKPT []secureEpisodeHit, magma []ActivatedNode) map[string]float64 {
	const k = 1
	rrf := map[string]float64{}

	for rank, item := range entityLex {
		rrf["ent:"+item.entity.UUID] += 1.0 / float64(k+rank+1)
	}
	for rank, item := range edgeLex {
		rrf["edg:"+item.edge.UUID] += 1.0 / float64(k+rank+1)
	}
	for rank, item := range episodeLex {
		rrf["ep:"+item.row.EpisodeUUID] += 1.5 / float64(k+rank+1)
	}
	for rank, item := range episodeKPT {
		rrf["ep:"+item.row.EpisodeUUID] += 1.5 / float64(k+rank+1)
	}
	for rank, item := range magma {
		rrf["ent:"+item.UUID] += 1.0 / float64(k+rank+1)
	}

	return rrf
}

func secureApplyCommunityBoost(rrf map[string]float64, edges []store.Edge, communityMap map[string]int, seedCommunities map[int]bool) {
	if len(seedCommunities) == 0 {
		return
	}
	for _, edge := range edges {
		srcCommunity, ok := communityMap[edge.SourceUUID]
		if ok && seedCommunities[srcCommunity] {
			rrf["edg:"+edge.UUID] += 0.15
			continue
		}
		tgtCommunity, ok := communityMap[edge.TargetUUID]
		if ok && seedCommunities[tgtCommunity] {
			rrf["edg:"+edge.UUID] += 0.15
		}
	}
	for cid := range seedCommunities {
		rrf[fmt.Sprintf("com:%d", cid)] += 0.6
	}
}

func secureBuildResults(ctx context.Context, db *store.DB, personalKey, groupID string, rrf map[string]float64, edges []store.Edge, episodes map[string]store.SecureEpisodeIndexRow, seedCommunities map[int]bool, limit int) ([]SearchResult, error) {
	edgesByUUID := make(map[string]store.Edge, len(edges))
	for _, edge := range edges {
		edgesByUUID[edge.UUID] = edge
	}

	communities, err := db.ListSecureCommunities(ctx, groupID, personalKey)
	if err != nil {
		return nil, fmt.Errorf("list secure communities: %w", err)
	}
	communityByID := make(map[int]store.CommunitySummary, len(communities))
	for _, community := range communities {
		communityByID[community.CommunityID] = community
	}

	type entry struct {
		key   string
		score float64
	}
	entries := make([]entry, 0, len(rrf))
	for key, score := range rrf {
		entries = append(entries, entry{key: key, score: score})
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].score > entries[j].score })

	results := make([]SearchResult, 0, limit)
	for _, item := range entries {
		if len(results) >= limit*3 {
			break
		}
		if strings.HasPrefix(item.key, "ent:") {
			continue
		}
		if strings.HasPrefix(item.key, "edg:") {
			uid := strings.TrimPrefix(item.key, "edg:")
			edge, ok := edgesByUUID[uid]
			if !ok {
				continue
			}
			validAt := ""
			if edge.ValidAt != nil {
				validAt = *edge.ValidAt
			}
			results = append(results, SearchResult{
				Type:    "edge",
				UUID:    edge.UUID,
				Title:   edge.Name,
				Body:    edge.Fact,
				Score:   item.score,
				Source:  secureFirstEdgeSource(edge, episodes),
				ValidAt: validAt,
			})
			continue
		}
		if strings.HasPrefix(item.key, "ep:") {
			uid := strings.TrimPrefix(item.key, "ep:")
			row, ok := episodes[uid]
			if !ok {
				continue
			}
			results = append(results, SearchResult{
				Type:   "episode",
				UUID:   row.EpisodeUUID,
				Title:  row.Source,
				Body:   row.Content,
				Score:  item.score,
				Source: row.Source,
			})
			continue
		}
		if !strings.HasPrefix(item.key, "com:") {
			continue
		}
		var cid int
		_, scanErr := fmt.Sscanf(strings.TrimPrefix(item.key, "com:"), "%d", &cid)
		if scanErr != nil {
			continue
		}
		if !seedCommunities[cid] {
			continue
		}
		community, ok := communityByID[cid]
		if !ok {
			continue
		}
		if community.Report == "" {
			continue
		}
		results = append(results, SearchResult{
			Type:  "community",
			UUID:  fmt.Sprintf("community:%d", cid),
			Title: fmt.Sprintf("community-%d", cid),
			Body:  community.Report,
			Score: item.score,
		})
	}

	return secureDedupResults(results), nil
}

func secureDedupResults(results []SearchResult) []SearchResult {
	seen := make(map[string]bool)
	out := make([]SearchResult, 0, len(results))
	for _, result := range results {
		key := result.Type + ":" + result.UUID
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, result)
	}
	return out
}

func secureApplyRelevanceCutoff(results []SearchResult) []SearchResult {
	if len(results) <= 1 {
		return results
	}
	topScore := results[0].Score
	cutoff := topScore * 0.15
	for i, result := range results[1:] {
		if result.Score >= cutoff {
			continue
		}
		return results[:i+1]
	}
	return results
}

func secureApplyUncertaintyGate(results []SearchResult) []SearchResult {
	if len(results) == 0 {
		return nil
	}
	const uncertaintyGate = 0.1
	if results[0].Score >= uncertaintyGate {
		return results
	}
	return nil
}

func secureApplyTemporalDecay(results []SearchResult) {
	maxSess := 0
	for _, result := range results {
		sess := sessionFromSource(result.Source)
		if sess <= maxSess {
			continue
		}
		maxSess = sess
	}
	if maxSess <= 1 {
		return
	}
	const lambdaT = 0.3
	for i := range results {
		sess := sessionFromSource(results[i].Source)
		if sess <= 0 {
			continue
		}
		age := float64(maxSess-sess) / float64(maxSess)
		results[i].Score *= math.Exp(-lambdaT * age)
	}
}

func secureFirstEdgeSource(edge store.Edge, episodes map[string]store.SecureEpisodeIndexRow) string {
	if edge.Episodes == "" {
		return ""
	}
	var episodeUUIDs []string
	if err := json.Unmarshal([]byte(edge.Episodes), &episodeUUIDs); err != nil {
		return ""
	}
	for _, episodeUUID := range episodeUUIDs {
		row, ok := episodes[episodeUUID]
		if !ok {
			continue
		}
		return row.Source
	}
	return ""
}
