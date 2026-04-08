package store

import (
	"context"
	"database/sql"
	"fmt"
	"math"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/sharpner/ultramemory/secureindex"
	"gonum.org/v1/gonum/graph/community"
	"gonum.org/v1/gonum/graph/simple"
)

const (
	secureCurvatureK        = 10
	secureCurvatureMinScore = 0.45
)

// SecureResolveResult summarises KPT-native duplicate resolution on episodes.
type SecureResolveResult struct {
	ClustersFound  int `json:"clusters_found"`
	EpisodesMerged int `json:"episodes_merged"`
}

// ResolveSecureEntities merges near-duplicate encrypted entities on the secure graph path.
func (d *DB) ResolveSecureEntities(ctx context.Context, groupID, personalKey string, cfg ResolveConfig) (ResolveResult, error) {
	entities, err := d.AllEntities(ctx, groupID)
	if err != nil {
		return ResolveResult{}, fmt.Errorf("load secure entities: %w", err)
	}
	if len(entities) == 0 {
		return ResolveResult{}, nil
	}

	decrypted := make([]Entity, 0, len(entities))
	for _, entity := range entities {
		name, err := secureindex.DecryptString(personalKey, "entity-name", entity.Name)
		if err != nil {
			return ResolveResult{}, fmt.Errorf("decrypt entity %s name: %w", entity.UUID, err)
		}
		entity.Name = name
		description, err := secureindex.DecryptString(personalKey, "entity-description", entity.Description)
		if err != nil {
			return ResolveResult{}, fmt.Errorf("decrypt entity %s description: %w", entity.UUID, err)
		}
		entity.Description = description
		decrypted = append(decrypted, entity)
	}

	byType := make(map[string][]Entity)
	for _, entity := range decrypted {
		byType[entity.EntityType] = append(byType[entity.EntityType], entity)
	}

	uuidToIdx := make(map[string]int, len(decrypted))
	for i, entity := range decrypted {
		uuidToIdx[entity.UUID] = i
	}
	uf := newUnionFind(len(decrypted))
	for _, group := range byType {
		for i := 0; i < len(group); i++ {
			for j := i + 1; j < len(group); j++ {
				nameSim := tokenJaccard(group[i].Name, group[j].Name)
				if nameSim < 0.5 {
					continue
				}
				descSim := tokenJaccard(group[i].Description, group[j].Description)
				if nameSim < cfg.Threshold && descSim < cfg.Threshold {
					continue
				}
				uf.union(uuidToIdx[group[i].UUID], uuidToIdx[group[j].UUID])
			}
		}
	}

	rootToMembers := make(map[int][]int)
	for i := range decrypted {
		root := uf.find(i)
		rootToMembers[root] = append(rootToMembers[root], i)
	}

	var mergeClusters [][]Entity
	for _, members := range rootToMembers {
		if len(members) < 2 {
			continue
		}
		cluster := make([]Entity, 0, len(members))
		for _, idx := range members {
			cluster = append(cluster, decrypted[idx])
		}
		mergeClusters = append(mergeClusters, cluster)
	}

	result := ResolveResult{ClustersFound: len(mergeClusters)}
	if cfg.DryRun {
		return result, nil
	}
	for _, cluster := range mergeClusters {
		canonical := pickCanonical(ctx, d, cluster, groupID)
		dupes := make([]Entity, 0, len(cluster)-1)
		for _, entity := range cluster {
			if entity.UUID == canonical.UUID {
				continue
			}
			dupes = append(dupes, entity)
		}
		edgesRetargeted, episodesRelinked, err := d.mergeCluster(ctx, groupID, canonical, dupes)
		if err != nil {
			return result, err
		}
		result.EntitiesMerged += len(dupes)
		result.EdgesRetargeted += edgesRetargeted
		result.EpisodesRelinked += episodesRelinked
	}
	return result, nil
}

// ResolveSecureEpisodes merges near-duplicate episodes inside the keyed KPT space.
func (d *DB) ResolveSecureEpisodes(ctx context.Context, groupID, methodName, personalKey string, threshold float64, dryRun bool) (SecureResolveResult, error) {
	rows, err := d.AllSecureEpisodes(ctx, groupID, methodName)
	if err != nil {
		return SecureResolveResult{}, fmt.Errorf("load secure episodes: %w", err)
	}
	if len(rows) == 0 {
		return SecureResolveResult{}, nil
	}

	method := secureindex.NewMethod(personalKey, len(rows[0].BaseWaveReal))
	states := make([]secureindex.State, 0, len(rows))
	decrypted := make([]SecureEpisodeIndexRow, 0, len(rows))
	for _, row := range rows {
		content, err := secureindex.DecryptString(personalKey, "episode-content", row.Content)
		if err != nil {
			return SecureResolveResult{}, fmt.Errorf("decrypt episode %s content: %w", row.EpisodeUUID, err)
		}
		source, err := secureindex.DecryptString(personalKey, "episode-source", row.Source)
		if err != nil {
			return SecureResolveResult{}, fmt.Errorf("decrypt episode %s source: %w", row.EpisodeUUID, err)
		}
		row.Content = content
		row.Source = source
		decrypted = append(decrypted, row)
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

	uf := newUnionFind(len(rows))
	for i := 0; i < len(rows); i++ {
		for j := i + 1; j < len(rows); j++ {
			score := method.ScoreDocs(states[i], states[j])
			nameSim := tokenJaccard(decrypted[i].Content, decrypted[j].Content)
			if score < threshold && nameSim < threshold {
				continue
			}
			uf.union(i, j)
		}
	}

	rootToMembers := make(map[int][]int)
	for i := range rows {
		root := uf.find(i)
		rootToMembers[root] = append(rootToMembers[root], i)
	}

	var clusters [][]SecureEpisodeIndexRow
	for _, members := range rootToMembers {
		if len(members) < 2 {
			continue
		}
		cluster := make([]SecureEpisodeIndexRow, 0, len(members))
		for _, idx := range members {
			cluster = append(cluster, decrypted[idx])
		}
		clusters = append(clusters, cluster)
	}

	result := SecureResolveResult{ClustersFound: len(clusters)}
	if len(clusters) == 0 {
		return result, nil
	}

	for _, cluster := range clusters {
		result.EpisodesMerged += len(cluster) - 1
	}
	if dryRun {
		return result, nil
	}

	for _, cluster := range clusters {
		canonical := pickSecureCanonical(cluster)
		dupes := make([]SecureEpisodeIndexRow, 0, len(cluster)-1)
		for _, row := range cluster {
			if row.EpisodeUUID == canonical.EpisodeUUID {
				continue
			}
			dupes = append(dupes, row)
		}
		if err := d.mergeSecureCluster(ctx, groupID, methodName, dupes); err != nil {
			return result, err
		}
	}

	return result, nil
}

// ComputeSecureCurvatures computes Ollivier-Ricci curvature on the keyed episode graph.
func (d *DB) ComputeSecureCurvatures(ctx context.Context, groupID, methodName, personalKey string) ([]EdgeCurvature, CurvatureStats, error) {
	start := time.Now()
	rows, err := d.AllSecureEpisodes(ctx, groupID, methodName)
	if err != nil {
		return nil, CurvatureStats{}, fmt.Errorf("load secure episodes: %w", err)
	}
	if len(rows) == 0 {
		return nil, CurvatureStats{}, nil
	}

	method := secureindex.NewMethod(personalKey, len(rows[0].BaseWaveReal))
	states := make([]secureindex.State, 0, len(rows))
	labels := make([]string, 0, len(rows))
	for _, row := range rows {
		content, err := secureindex.DecryptString(personalKey, "episode-content", row.Content)
		if err != nil {
			return nil, CurvatureStats{}, fmt.Errorf("decrypt episode %s content: %w", row.EpisodeUUID, err)
		}
		source, err := secureindex.DecryptString(personalKey, "episode-source", row.Source)
		if err != nil {
			return nil, CurvatureStats{}, fmt.Errorf("decrypt episode %s source: %w", row.EpisodeUUID, err)
		}
		label := filepath.Base(source)
		if label == "" || label == "." || label == "/" {
			label = secureLabel(content)
		}
		labels = append(labels, label)
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

	g := &adjGraph{neighbors: make(map[int64][]int64, len(states))}
	for i := range states {
		g.neighbors[int64(i)] = nil
	}

	type neighbor struct {
		index int
		score float64
	}

	edgeSet := map[edgePair]bool{}
	for i := range states {
		neighbors := make([]neighbor, 0, len(states)-1)
		for j := range states {
			if i == j {
				continue
			}
			score := method.ScoreDocs(states[i], states[j])
			if score < secureCurvatureMinScore {
				continue
			}
			neighbors = append(neighbors, neighbor{index: j, score: score})
		}
		slices.SortFunc(neighbors, func(a, b neighbor) int {
			if a.score > b.score {
				return -1
			}
			if a.score < b.score {
				return 1
			}
			return 0
		})
		if len(neighbors) > secureCurvatureK {
			neighbors = neighbors[:secureCurvatureK]
		}
		for _, nb := range neighbors {
			lo := int64(i)
			hi := int64(nb.index)
			if lo > hi {
				lo, hi = hi, lo
			}
			ep := edgePair{lo: lo, hi: hi}
			if edgeSet[ep] {
				continue
			}
			edgeSet[ep] = true
			g.neighbors[lo] = append(g.neighbors[lo], hi)
			g.neighbors[hi] = append(g.neighbors[hi], lo)
		}
	}
	g.sortNeighbors()

	edges := make([]edgePair, 0, len(edgeSet))
	for ep := range edgeSet {
		edges = append(edges, ep)
	}

	results := make([]EdgeCurvature, 0, len(edges))
	stats := CurvatureStats{
		TotalEdges: len(edges),
	}
	if len(edges) == 0 {
		return results, stats, nil
	}

	stats.Min = 1e9
	stats.Max = -1e9
	for _, ep := range edges {
		k := ollivierRicci(g, ep.lo, ep.hi)
		sourceIdx := int(ep.lo)
		targetIdx := int(ep.hi)
		results = append(results, EdgeCurvature{
			SourceUUID: rows[sourceIdx].EpisodeUUID,
			TargetUUID: rows[targetIdx].EpisodeUUID,
			SourceName: labels[sourceIdx],
			TargetName: labels[targetIdx],
			Curvature:  k,
		})
		stats.Mean += k
		if k < stats.Min {
			stats.Min = k
		}
		if k > stats.Max {
			stats.Max = k
		}
		switch {
		case k < -0.05:
			stats.Bridges++
		case k > 0.05:
			stats.Internal++
		default:
			stats.Flat++
		}
	}

	stats.Mean /= float64(len(results))
	stats.Elapsed = time.Since(start).Round(time.Millisecond).String()
	return results, stats, nil
}

// ComputeSecureGraphCurvatures computes ORC on the encrypted entity graph and decrypts labels for display.
func (d *DB) ComputeSecureGraphCurvatures(ctx context.Context, groupID, personalKey string) ([]EdgeCurvature, CurvatureStats, error) {
	curvatures, stats, err := d.ComputeCurvatures(ctx, groupID, 0)
	if err != nil {
		return nil, CurvatureStats{}, err
	}
	entities, err := d.AllEntities(ctx, groupID)
	if err != nil {
		return nil, CurvatureStats{}, fmt.Errorf("load secure entities: %w", err)
	}
	names := make(map[string]string, len(entities))
	for _, entity := range entities {
		name, err := secureindex.DecryptString(personalKey, "entity-name", entity.Name)
		if err != nil {
			names[entity.UUID] = "<locked>"
			continue
		}
		names[entity.UUID] = name
	}
	for i := range curvatures {
		curvatures[i].SourceName = names[curvatures[i].SourceUUID]
		curvatures[i].TargetName = names[curvatures[i].TargetUUID]
	}
	return curvatures, stats, nil
}

// SecureRicciCommunities runs ORC-weighted Louvain on the encrypted entity graph.
func (d *DB) SecureRicciCommunities(ctx context.Context, groupID string, resolution float64) (CommunityResult, error) {
	if resolution <= 0 {
		resolution = 1.0
	}

	rows, err := d.sql.QueryContext(ctx,
		`SELECT uuid FROM entities WHERE group_id = ?`, groupID)
	if err != nil {
		return CommunityResult{}, fmt.Errorf("load entities: %w", err)
	}
	defer rows.Close() //nolint:errcheck

	uuidToID := map[string]int64{}
	idToUUID := map[int64]string{}
	var nextID int64
	for rows.Next() {
		var uuid string
		if err := rows.Scan(&uuid); err != nil {
			return CommunityResult{}, err
		}
		uuidToID[uuid] = nextID
		idToUUID[nextID] = uuid
		nextID++
	}
	if err := rows.Err(); err != nil {
		return CommunityResult{}, err
	}
	if nextID < 2 {
		return CommunityResult{Entities: int(nextID)}, nil
	}

	curvatures, _, err := d.ComputeCurvatures(ctx, groupID, 0)
	if err != nil {
		return CommunityResult{}, err
	}

	louvainGraph := simple.NewWeightedUndirectedGraph(0, 0)
	for id := int64(0); id < nextID; id++ {
		louvainGraph.AddNode(simple.Node(id))
	}

	minCurvature := math.Inf(1)
	for _, edge := range curvatures {
		if edge.Curvature < minCurvature {
			minCurvature = edge.Curvature
		}
	}
	offset := 0.01
	if len(curvatures) > 0 {
		offset = -minCurvature + 0.01
	}

	for _, edge := range curvatures {
		sourceID, ok := uuidToID[edge.SourceUUID]
		if !ok {
			continue
		}
		targetID, ok := uuidToID[edge.TargetUUID]
		if !ok {
			continue
		}
		weight := edge.Curvature + offset
		if weight < 0.01 {
			weight = 0.01
		}
		louvainGraph.SetWeightedEdge(louvainGraph.NewWeightedEdge(simple.Node(sourceID), simple.Node(targetID), weight))
	}

	reduced := community.Modularize(louvainGraph, resolution, nil)
	partitions := reduced.Communities()
	communityMap := make(map[int64][]string, len(partitions))
	for communityID, members := range partitions {
		uuidList := make([]string, 0, len(members))
		for _, member := range members {
			uuidList = append(uuidList, idToUUID[member.ID()])
		}
		communityMap[int64(communityID)] = uuidList
	}
	if err := d.WriteCommunityIDs(ctx, groupID, communityMap); err != nil {
		return CommunityResult{}, err
	}

	return CommunityResult{
		Communities: len(partitions),
		Entities:    int(nextID),
	}, nil
}

// TopSecureBridges returns persisted bridge edges with decrypted labels.
func (d *DB) TopSecureBridges(ctx context.Context, groupID string, n int, personalKey string) ([]EdgeCurvature, error) {
	bridges, err := d.TopBridges(ctx, groupID, n)
	if err != nil {
		return nil, err
	}
	entities, err := d.AllEntities(ctx, groupID)
	if err != nil {
		return nil, fmt.Errorf("load secure entities: %w", err)
	}
	names := make(map[string]string, len(entities))
	for _, entity := range entities {
		name, err := secureindex.DecryptString(personalKey, "entity-name", entity.Name)
		if err != nil {
			names[entity.UUID] = "<locked>"
			continue
		}
		names[entity.UUID] = name
	}
	for i := range bridges {
		bridges[i].SourceName = names[bridges[i].SourceUUID]
		bridges[i].TargetName = names[bridges[i].TargetUUID]
	}
	return bridges, nil
}

func (d *DB) mergeSecureCluster(ctx context.Context, groupID, methodName string, dupes []SecureEpisodeIndexRow) error {
	if len(dupes) == 0 {
		return nil
	}

	dupeUUIDs := make([]string, 0, len(dupes))
	for _, row := range dupes {
		dupeUUIDs = append(dupeUUIDs, row.EpisodeUUID)
	}
	ph := placeholders(len(dupeUUIDs))
	args := stringsToAny(dupeUUIDs)

	tx, err := d.sql.BeginTx(ctx, &sql.TxOptions{})
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck

	if _, err := tx.ExecContext(ctx,
		`DELETE FROM secure_episode_index WHERE episode_uuid IN (`+ph+`) AND group_id = ? AND method = ?`,
		append(append(args, groupID), methodName)...,
	); err != nil {
		return fmt.Errorf("delete secure rows: %w", err)
	}

	if _, err := tx.ExecContext(ctx,
		`DELETE FROM episodes_fts WHERE uuid IN (`+ph+`)`,
		args...,
	); err != nil {
		return fmt.Errorf("delete secure fts rows: %w", err)
	}

	if _, err := tx.ExecContext(ctx,
		`DELETE FROM episodes WHERE uuid IN (`+ph+`) AND group_id = ?`,
		append(args, groupID)...,
	); err != nil {
		return fmt.Errorf("delete secure episodes: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit secure merge: %w", err)
	}
	return nil
}

func pickSecureCanonical(cluster []SecureEpisodeIndexRow) SecureEpisodeIndexRow {
	best := cluster[0]
	for _, row := range cluster[1:] {
		if len(strings.TrimSpace(row.Content)) > len(strings.TrimSpace(best.Content)) {
			best = row
			continue
		}
		if len(strings.TrimSpace(row.Content)) == len(strings.TrimSpace(best.Content)) && row.EpisodeUUID < best.EpisodeUUID {
			best = row
		}
	}
	return best
}

func secureLabel(content string) string {
	value := strings.TrimSpace(strings.ReplaceAll(content, "\n", " "))
	if len(value) <= 24 {
		return value
	}
	return value[:21] + "..."
}
