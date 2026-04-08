package store

import (
	"context"
	"database/sql"
	"fmt"
)

// Edge is a typed relationship between two entities.
type Edge struct {
	UUID       string
	SourceUUID string
	TargetUUID string
	Name       string // relation type e.g. WORKS_AT
	LookupKey  string
	Fact       string // natural language fact
	GroupID    string
	ValidAt    *string
	InvalidAt  *string
	Episodes   string // JSON array of episode UUIDs
	Embedding  []float32
}

// UpsertEdge inserts or updates an edge.
// Deduplicates by (source_uuid, target_uuid, name, group_id).
func (d *DB) UpsertEdge(ctx context.Context, e Edge) error {
	var existing string
	query := `SELECT uuid FROM edges
		 WHERE source_uuid = ? AND target_uuid = ? AND name = ? AND group_id = ?
		 LIMIT 1`
	args := []any{e.SourceUUID, e.TargetUUID, e.Name, e.GroupID}
	if e.LookupKey != "" {
		query = `SELECT uuid FROM edges
		 WHERE group_id = ? AND lookup_key = ?
		 LIMIT 1`
		args = []any{e.GroupID, e.LookupKey}
	}
	err := d.sql.QueryRowContext(ctx, query, args...).Scan(&existing)
	if err != nil && err != sql.ErrNoRows {
		return fmt.Errorf("lookup edge: %w", err)
	}

	var embBlob []byte
	if len(e.Embedding) > 0 {
		embBlob = EncodeEmbedding(e.Embedding)
	}

	if existing != "" {
		if _, err = d.sql.ExecContext(ctx,
			`UPDATE edges SET fact = ?, embedding = ?, lookup_key = CASE WHEN ? != '' THEN ? ELSE lookup_key END WHERE uuid = ?`,
			e.Fact, embBlob, e.LookupKey, e.LookupKey, existing,
		); err != nil {
			return err
		}
		// Keep FTS in sync with updated fact.
		if _, err := d.sql.ExecContext(ctx,
			`DELETE FROM edges_fts WHERE uuid = ?`, existing,
		); err != nil {
			return err
		}
		if e.LookupKey != "" {
			return nil
		}
		_, err = d.sql.ExecContext(ctx,
			`INSERT INTO edges_fts (uuid, fact) VALUES (?, ?)`, existing, e.Fact,
		)
		return err
	}

	_, err = d.sql.ExecContext(ctx, `
		INSERT INTO edges (uuid, source_uuid, target_uuid, name, lookup_key, fact, group_id, valid_at, invalid_at, episodes, embedding)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		e.UUID, e.SourceUUID, e.TargetUUID, e.Name, e.LookupKey, e.Fact,
		e.GroupID, e.ValidAt, e.InvalidAt, e.Episodes, embBlob,
	)
	if err != nil {
		return fmt.Errorf("insert edge: %w", err)
	}

	if _, err := d.sql.ExecContext(ctx,
		`DELETE FROM edges_fts WHERE uuid = ?`, e.UUID,
	); err != nil {
		return err
	}
	if e.LookupKey != "" {
		return nil
	}
	_, err = d.sql.ExecContext(ctx,
		`INSERT INTO edges_fts (uuid, fact) VALUES (?, ?)`, e.UUID, e.Fact,
	)
	return err
}

// CountEdges returns the total edge count for a group.
func (d *DB) CountEdges(ctx context.Context, groupID string) (int, error) {
	var n int
	err := d.sql.QueryRowContext(ctx,
		`SELECT count(*) FROM edges WHERE group_id = ?`, groupID,
	).Scan(&n)
	return n, err
}

// SearchEdgesFTS performs fulltext search over edge facts.
func (d *DB) SearchEdgesFTS(ctx context.Context, query, groupID string, limit int) ([]Edge, error) {
	fq := fts5Query(query)
	if fq == "" {
		return nil, nil
	}
	rows, err := d.sql.QueryContext(ctx, `
		SELECT e.uuid, e.source_uuid, e.target_uuid, e.name, e.lookup_key, e.fact, e.valid_at, e.embedding
		FROM edges_fts f
		JOIN edges e ON e.uuid = f.uuid
		WHERE edges_fts MATCH ? AND e.group_id = ?
		ORDER BY rank
		LIMIT ?`,
		fq, groupID, limit,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close() //nolint:errcheck

	var out []Edge
	for rows.Next() {
		var e Edge
		var blob []byte
		if err := rows.Scan(&e.UUID, &e.SourceUUID, &e.TargetUUID, &e.Name, &e.LookupKey, &e.Fact, &e.ValidAt, &blob); err != nil {
			return nil, err
		}
		e.GroupID = groupID
		e.Embedding = DecodeEmbedding(blob)
		out = append(out, e)
	}
	return out, rows.Err()
}

// EdgesForEntities returns all edges where source_uuid or target_uuid is in uuids.
// Used by MAGMA neighborhood expansion: activated entities → adjacent edge facts.
func (d *DB) EdgesForEntities(ctx context.Context, uuids []string, groupID string) ([]Edge, error) {
	if len(uuids) == 0 {
		return nil, nil
	}
	ph := placeholders(len(uuids))
	args := make([]any, 0, 2*len(uuids)+1)
	for _, u := range uuids {
		args = append(args, u)
	}
	for _, u := range uuids {
		args = append(args, u)
	}
	args = append(args, groupID)
	rows, err := d.sql.QueryContext(ctx, `
		SELECT uuid, source_uuid, target_uuid, name, lookup_key, fact, valid_at
		FROM edges
		WHERE (source_uuid IN (`+ph+`) OR target_uuid IN (`+ph+`))
		  AND group_id = ?`, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close() //nolint:errcheck

	var out []Edge
	for rows.Next() {
		var e Edge
		if err := rows.Scan(&e.UUID, &e.SourceUUID, &e.TargetUUID, &e.Name, &e.LookupKey, &e.Fact, &e.ValidAt); err != nil {
			return nil, err
		}
		e.GroupID = groupID
		out = append(out, e)
	}
	return out, rows.Err()
}

// AllEdgesWithEmbeddings loads all edges with embeddings for vector search.
func (d *DB) AllEdgesWithEmbeddings(ctx context.Context, groupID string) ([]Edge, error) {
	rows, err := d.sql.QueryContext(ctx,
		`SELECT uuid, source_uuid, target_uuid, name, lookup_key, fact, valid_at, embedding
		 FROM edges
		 WHERE group_id = ? AND embedding IS NOT NULL`,
		groupID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close() //nolint:errcheck

	var out []Edge
	for rows.Next() {
		var e Edge
		var blob []byte
		if err := rows.Scan(&e.UUID, &e.SourceUUID, &e.TargetUUID, &e.Name, &e.LookupKey, &e.Fact, &e.ValidAt, &blob); err != nil {
			return nil, err
		}
		e.GroupID = groupID
		e.Embedding = DecodeEmbedding(blob)
		out = append(out, e)
	}
	return out, rows.Err()
}

// AllEdges loads all edges for one group, regardless of embedding presence.
func (d *DB) AllEdges(ctx context.Context, groupID string) ([]Edge, error) {
	rows, err := d.sql.QueryContext(ctx, `
		SELECT uuid, source_uuid, target_uuid, name, lookup_key, fact, valid_at, invalid_at, episodes, embedding
		FROM edges
		WHERE group_id = ?
		ORDER BY created_at ASC, uuid ASC`,
		groupID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close() //nolint:errcheck

	var out []Edge
	for rows.Next() {
		var e Edge
		var blob []byte
		if err := rows.Scan(
			&e.UUID,
			&e.SourceUUID,
			&e.TargetUUID,
			&e.Name,
			&e.LookupKey,
			&e.Fact,
			&e.ValidAt,
			&e.InvalidAt,
			&e.Episodes,
			&blob,
		); err != nil {
			return nil, err
		}
		e.GroupID = groupID
		e.Embedding = DecodeEmbedding(blob)
		out = append(out, e)
	}
	return out, rows.Err()
}
