package store

import (
	"context"
	"database/sql"
	"fmt"
)

// Entity is a named node in the knowledge graph.
type Entity struct {
	UUID        string
	Name        string
	LookupKey   string
	EntityType  string
	GroupID     string
	Embedding   []float32
	Description string
	CommunityID int
}

// UpsertEntity inserts or merges an entity by name+group (case-insensitive).
// Returns the canonical UUID that should be used.
func (d *DB) UpsertEntity(ctx context.Context, e Entity) (string, error) {
	var existing string
	query := `SELECT uuid FROM entities
		 WHERE group_id = ? AND lower(name) = lower(?)
		 LIMIT 1`
	args := []any{e.GroupID, e.Name}
	if e.LookupKey != "" {
		query = `SELECT uuid FROM entities
		 WHERE group_id = ? AND lookup_key = ?
		 LIMIT 1`
		args = []any{e.GroupID, e.LookupKey}
	}
	err := d.sql.QueryRowContext(ctx, query, args...).Scan(&existing)
	if err != nil && err != sql.ErrNoRows {
		return "", fmt.Errorf("lookup entity: %w", err)
	}
	if existing != "" {
		if len(e.Embedding) > 0 {
			_, err = d.sql.ExecContext(ctx,
				`UPDATE entities SET embedding = ?, description = ?, lookup_key = CASE WHEN ? != '' THEN ? ELSE lookup_key END WHERE uuid = ?`,
				EncodeEmbedding(e.Embedding), e.Description, e.LookupKey, e.LookupKey, existing,
			)
			return existing, err
		}
		_, err = d.sql.ExecContext(ctx,
			`UPDATE entities SET description = ?, lookup_key = CASE WHEN ? != '' THEN ? ELSE lookup_key END WHERE uuid = ?`,
			e.Description, e.LookupKey, e.LookupKey, existing,
		)
		return existing, err
	}

	var embBlob []byte
	if len(e.Embedding) > 0 {
		embBlob = EncodeEmbedding(e.Embedding)
	}
	_, err = d.sql.ExecContext(ctx, `
		INSERT INTO entities (uuid, name, lookup_key, entity_type, group_id, embedding, description)
		VALUES (?, ?, ?, ?, ?, ?, ?)`,
		e.UUID, e.Name, e.LookupKey, e.EntityType, e.GroupID, embBlob, e.Description,
	)
	if err != nil {
		return "", fmt.Errorf("insert entity: %w", err)
	}
	if _, err := d.sql.ExecContext(ctx,
		`DELETE FROM entities_fts WHERE uuid = ?`, e.UUID,
	); err != nil {
		return "", err
	}
	if e.LookupKey != "" {
		return e.UUID, nil
	}
	_, err = d.sql.ExecContext(ctx,
		`INSERT INTO entities_fts (uuid, name) VALUES (?, ?)`, e.UUID, e.Name,
	)
	return e.UUID, err
}

// LinkEntityEpisode creates the many-to-many association.
func (d *DB) LinkEntityEpisode(ctx context.Context, entityUUID, episodeUUID string) error {
	_, err := d.sql.ExecContext(ctx,
		`INSERT OR IGNORE INTO entity_episodes (entity_uuid, episode_uuid) VALUES (?, ?)`,
		entityUUID, episodeUUID,
	)
	return err
}

// CountEntities returns the total entity count for a group.
func (d *DB) CountEntities(ctx context.Context, groupID string) (int, error) {
	var n int
	err := d.sql.QueryRowContext(ctx,
		`SELECT count(*) FROM entities WHERE group_id = ?`, groupID,
	).Scan(&n)
	return n, err
}

// AllEntitiesWithEmbeddings loads all entities that have embeddings for vector search.
func (d *DB) AllEntitiesWithEmbeddings(ctx context.Context, groupID string) ([]Entity, error) {
	rows, err := d.sql.QueryContext(ctx,
		`SELECT uuid, name, lookup_key, entity_type, embedding, description, community_id
		 FROM entities
		 WHERE group_id = ? AND embedding IS NOT NULL`,
		groupID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close() //nolint:errcheck

	var out []Entity
	for rows.Next() {
		var e Entity
		var blob []byte
		if err := rows.Scan(&e.UUID, &e.Name, &e.LookupKey, &e.EntityType, &blob, &e.Description, &e.CommunityID); err != nil {
			return nil, err
		}
		e.GroupID = groupID
		e.Embedding = DecodeEmbedding(blob)
		out = append(out, e)
	}
	return out, rows.Err()
}

// EntitiesForEpisodes returns entities linked to any of the given episode UUIDs
// via the entity_episodes join table. Used for episode→entity MAGMA seed expansion:
// FTS episode hits often link to entities not directly matched by entity FTS.
func (d *DB) EntitiesForEpisodes(ctx context.Context, episodeUUIDs []string, groupID string) ([]Entity, error) {
	if len(episodeUUIDs) == 0 {
		return nil, nil
	}
	ph := placeholders(len(episodeUUIDs))
	args := make([]any, 0, len(episodeUUIDs)+1)
	for _, u := range episodeUUIDs {
		args = append(args, u)
	}
	args = append(args, groupID)
	rows, err := d.sql.QueryContext(ctx, `
		SELECT DISTINCT e.uuid, e.name, e.lookup_key, e.entity_type, e.embedding, e.description, e.community_id
		FROM entities e
		JOIN entity_episodes ee ON ee.entity_uuid = e.uuid
		WHERE ee.episode_uuid IN (`+ph+`) AND e.group_id = ?`, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close() //nolint:errcheck

	var out []Entity
	for rows.Next() {
		var e Entity
		var blob []byte
		if err := rows.Scan(&e.UUID, &e.Name, &e.LookupKey, &e.EntityType, &blob, &e.Description, &e.CommunityID); err != nil {
			return nil, err
		}
		e.GroupID = groupID
		e.Embedding = DecodeEmbedding(blob)
		out = append(out, e)
	}
	return out, rows.Err()
}

// SearchEntitiesFTS performs fulltext search on entity names.
func (d *DB) SearchEntitiesFTS(ctx context.Context, query, groupID string, limit int) ([]Entity, error) {
	fq := fts5Query(query)
	if fq == "" {
		return nil, nil
	}
	rows, err := d.sql.QueryContext(ctx, `
		SELECT e.uuid, e.name, e.lookup_key, e.entity_type, e.embedding, e.description, e.community_id
		FROM entities_fts f
		JOIN entities e ON e.uuid = f.uuid
		WHERE entities_fts MATCH ? AND e.group_id = ?
		ORDER BY rank
		LIMIT ?`,
		fq, groupID, limit,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close() //nolint:errcheck

	var out []Entity
	for rows.Next() {
		var e Entity
		var blob []byte
		if err := rows.Scan(&e.UUID, &e.Name, &e.LookupKey, &e.EntityType, &blob, &e.Description, &e.CommunityID); err != nil {
			return nil, err
		}
		e.GroupID = groupID
		e.Embedding = DecodeEmbedding(blob)
		out = append(out, e)
	}
	return out, rows.Err()
}

// AllEntities loads all entities for one group, regardless of embedding presence.
func (d *DB) AllEntities(ctx context.Context, groupID string) ([]Entity, error) {
	rows, err := d.sql.QueryContext(ctx, `
		SELECT uuid, name, lookup_key, entity_type, embedding, description, community_id
		FROM entities
		WHERE group_id = ?
		ORDER BY created_at ASC, uuid ASC`,
		groupID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close() //nolint:errcheck

	var out []Entity
	for rows.Next() {
		var e Entity
		var blob []byte
		if err := rows.Scan(&e.UUID, &e.Name, &e.LookupKey, &e.EntityType, &blob, &e.Description, &e.CommunityID); err != nil {
			return nil, err
		}
		e.GroupID = groupID
		e.Embedding = DecodeEmbedding(blob)
		out = append(out, e)
	}
	return out, rows.Err()
}
