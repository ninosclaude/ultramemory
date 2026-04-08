// Package store handles all SQLite persistence for memory-local.
// Uses modernc.org/sqlite (pure Go, no CGO) with WAL mode.
package store

import (
	"context"
	"database/sql"
	"encoding/binary"
	"fmt"
	"math"
	"net/url"
	"strings"
	"unicode"

	_ "modernc.org/sqlite"
)

// PRAGMAs are set via connection string so they apply to every pool connection.
// Schema only contains DDL.
const schema = `

CREATE TABLE IF NOT EXISTS db_meta (
	key   TEXT PRIMARY KEY,
	value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS episodes (
	uuid        TEXT PRIMARY KEY,
	content     TEXT NOT NULL,
	group_id    TEXT NOT NULL DEFAULT 'default',
	source      TEXT NOT NULL DEFAULT '',
	created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
	embedding   BLOB
);

CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
	content,
	uuid UNINDEXED,
	tokenize = 'porter unicode61'
);

CREATE TABLE IF NOT EXISTS entities (
	uuid        TEXT PRIMARY KEY,
	name        TEXT NOT NULL,
	lookup_key  TEXT NOT NULL DEFAULT '',
	entity_type TEXT NOT NULL,
	group_id    TEXT NOT NULL DEFAULT 'default',
	created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
	embedding    BLOB,
	description  TEXT NOT NULL DEFAULT '',
	community_id INTEGER NOT NULL DEFAULT -1
);

CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
	name,
	uuid UNINDEXED,
	tokenize = 'porter unicode61'
);

CREATE TABLE IF NOT EXISTS edges (
	uuid        TEXT PRIMARY KEY,
	source_uuid TEXT NOT NULL,
	target_uuid TEXT NOT NULL,
	name        TEXT NOT NULL,
	lookup_key  TEXT NOT NULL DEFAULT '',
	fact        TEXT NOT NULL,
	group_id    TEXT NOT NULL DEFAULT 'default',
	valid_at    TEXT,
	invalid_at  TEXT,
	episodes    TEXT NOT NULL DEFAULT '[]',
	created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
	embedding   BLOB,
	FOREIGN KEY (source_uuid) REFERENCES entities(uuid),
	FOREIGN KEY (target_uuid) REFERENCES entities(uuid)
);

CREATE VIRTUAL TABLE IF NOT EXISTS edges_fts USING fts5(
	fact,
	uuid UNINDEXED,
	tokenize = 'porter unicode61'
);

CREATE TABLE IF NOT EXISTS entity_episodes (
	entity_uuid  TEXT NOT NULL,
	episode_uuid TEXT NOT NULL,
	PRIMARY KEY (entity_uuid, episode_uuid),
	FOREIGN KEY (entity_uuid)  REFERENCES entities(uuid),
	FOREIGN KEY (episode_uuid) REFERENCES episodes(uuid)
);

CREATE TABLE IF NOT EXISTS jobs (
	id         INTEGER PRIMARY KEY AUTOINCREMENT,
	type       TEXT NOT NULL,
	payload    TEXT NOT NULL,
	status     TEXT NOT NULL DEFAULT 'pending',
	attempts   INTEGER NOT NULL DEFAULT 0,
	max_attempts INTEGER NOT NULL DEFAULT 3,
	error      TEXT,
	not_before DATETIME,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_jobs_pending
	ON jobs(status, not_before, created_at ASC)
	WHERE status = 'pending';

CREATE INDEX IF NOT EXISTS idx_entities_group
	ON entities(group_id, name);

CREATE INDEX IF NOT EXISTS idx_entities_lookup
	ON entities(group_id, lookup_key);

-- Composite indexes for GetNeighbors (MAGMA traversal hot path).
-- Filters always combine source_uuid/target_uuid with group_id;
-- composite indexes let SQLite seek directly without a post-filter scan.
CREATE INDEX IF NOT EXISTS idx_edges_src_grp
	ON edges(source_uuid, group_id);

CREATE INDEX IF NOT EXISTS idx_edges_tgt_grp
	ON edges(target_uuid, group_id);

CREATE INDEX IF NOT EXISTS idx_edges_group
	ON edges(group_id);

CREATE INDEX IF NOT EXISTS idx_edges_lookup
	ON edges(group_id, lookup_key);

CREATE TABLE IF NOT EXISTS community_reports (
	community_id INTEGER NOT NULL,
	group_id     TEXT NOT NULL,
	report       TEXT NOT NULL DEFAULT '',
	PRIMARY KEY (community_id, group_id)
);

CREATE TABLE IF NOT EXISTS secure_episode_index (
	episode_uuid  TEXT NOT NULL,
	group_id      TEXT NOT NULL,
	method        TEXT NOT NULL,
	public_layer  BLOB NOT NULL,
	base_vec      BLOB NOT NULL,
	base_wave_imag BLOB NOT NULL,
	wave_real     BLOB NOT NULL,
	wave_imag     BLOB NOT NULL,
	mode_weight   BLOB NOT NULL,
	mode_energy   BLOB NOT NULL,
	key_probe     BLOB,
	created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (episode_uuid, group_id, method),
	FOREIGN KEY (episode_uuid) REFERENCES episodes(uuid)
);

CREATE INDEX IF NOT EXISTS idx_secure_episode_index_group
	ON secure_episode_index(group_id, method);
`

// migrations runs after schema init to add columns to existing databases.
const migrations = `
ALTER TABLE entities ADD COLUMN community_id INTEGER NOT NULL DEFAULT -1;
ALTER TABLE entities ADD COLUMN description TEXT NOT NULL DEFAULT '';
ALTER TABLE entities ADD COLUMN lookup_key TEXT NOT NULL DEFAULT '';
CREATE INDEX IF NOT EXISTS idx_entities_community ON entities(group_id, community_id);
CREATE INDEX IF NOT EXISTS idx_entities_lookup ON entities(group_id, lookup_key);
CREATE TABLE IF NOT EXISTS edge_curvatures (
	source_uuid TEXT NOT NULL,
	target_uuid TEXT NOT NULL,
	group_id    TEXT NOT NULL,
	curvature   REAL NOT NULL,
	PRIMARY KEY (source_uuid, target_uuid, group_id)
);
CREATE INDEX IF NOT EXISTS idx_edge_curvatures_group ON edge_curvatures(group_id, curvature);
CREATE TABLE IF NOT EXISTS mutual_knn_edges (
	source_uuid TEXT NOT NULL,
	target_uuid TEXT NOT NULL,
	group_id   TEXT NOT NULL,
	PRIMARY KEY (source_uuid, target_uuid, group_id)
);
CREATE INDEX IF NOT EXISTS idx_mknn_group ON mutual_knn_edges(group_id);
ALTER TABLE jobs ADD COLUMN not_before DATETIME;
ALTER TABLE edges ADD COLUMN lookup_key TEXT NOT NULL DEFAULT '';
CREATE INDEX IF NOT EXISTS idx_edges_lookup ON edges(group_id, lookup_key);
CREATE TABLE IF NOT EXISTS secure_episode_index (
	episode_uuid  TEXT NOT NULL,
	group_id      TEXT NOT NULL,
	method        TEXT NOT NULL,
	public_layer  BLOB NOT NULL,
	base_vec      BLOB NOT NULL,
	base_wave_imag BLOB NOT NULL DEFAULT X'',
	wave_real     BLOB NOT NULL,
	wave_imag     BLOB NOT NULL,
	mode_weight   BLOB NOT NULL,
	mode_energy   BLOB NOT NULL,
	key_probe     BLOB,
	created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
	PRIMARY KEY (episode_uuid, group_id, method)
);
CREATE INDEX IF NOT EXISTS idx_secure_episode_index_group ON secure_episode_index(group_id, method);
ALTER TABLE secure_episode_index ADD COLUMN base_wave_imag BLOB NOT NULL DEFAULT X'';
ALTER TABLE secure_episode_index ADD COLUMN key_probe BLOB;
`

// DB wraps a SQLite database connection.
type DB struct {
	sql *sql.DB
}

// Open opens (or creates) the SQLite database at path.
// PRAGMAs are set via connection string so every pooled connection inherits them.
func Open(path string) (*DB, error) {
	dsn := buildDSN(path)
	conn, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("open sqlite: %w", err)
	}
	conn.SetMaxOpenConns(4)
	conn.SetMaxIdleConns(2)

	if err := ensureDBFormat(conn); err != nil {
		_ = conn.Close()
		return nil, err
	}
	if _, err := conn.ExecContext(context.Background(), schema); err != nil {
		_ = conn.Close()
		return nil, fmt.Errorf("apply schema: %w", err)
	}
	if err := runMigrations(conn); err != nil {
		_ = conn.Close()
		return nil, err
	}
	return &DB{sql: conn}, nil
}

// buildDSN constructs a modernc.org/sqlite connection string with PRAGMAs
// that apply to every connection in the pool, not just the first.
// Path is escaped to handle spaces, '?', '#' etc.
func buildDSN(path string) string {
	params := url.Values{}
	params.Add("_pragma", "journal_mode(WAL)")
	params.Add("_pragma", "busy_timeout(5000)")
	params.Add("_pragma", "synchronous(NORMAL)")
	params.Add("_pragma", "foreign_keys(ON)")
	return "file:" + url.PathEscape(path) + "?" + params.Encode()
}

// runMigrations executes each migration statement individually.
// Expected errors (duplicate column, already-existing index) are ignored;
// all other errors are fatal.
func runMigrations(conn *sql.DB) error {
	for _, m := range strings.Split(migrations, ";") {
		m = strings.TrimSpace(m)
		if m == "" {
			continue
		}
		_, err := conn.ExecContext(context.Background(), m)
		if err == nil {
			continue
		}
		msg := err.Error()
		if strings.Contains(msg, "duplicate column") || strings.Contains(msg, "already exists") {
			continue
		}
		return fmt.Errorf("migration %q: %w", m, err)
	}
	return nil
}

func ensureDBFormat(conn *sql.DB) error {
	legacy, err := hasLegacySchema(conn)
	if err != nil {
		return err
	}

	if _, err := conn.ExecContext(context.Background(), `CREATE TABLE IF NOT EXISTS db_meta (
		key   TEXT PRIMARY KEY,
		value TEXT NOT NULL
	)`); err != nil {
		return fmt.Errorf("ensure db_meta: %w", err)
	}

	var actual string
	err = conn.QueryRowContext(context.Background(),
		`SELECT value FROM db_meta WHERE key = ?`, DBFormatKey,
	).Scan(&actual)
	if err == nil {
		if actual == currentDBFormat {
			return nil
		}
		return &DBFormatMismatchError{
			Expected: currentDBFormat,
			Actual:   actual,
		}
	}
	if err != sql.ErrNoRows {
		return fmt.Errorf("read db format: %w", err)
	}

	stampedFormat := currentDBFormat
	if legacy {
		stampedFormat = DBFormatOllama
	}

	if err := stampDBFormat(conn, stampedFormat); err != nil {
		return err
	}

	if stampedFormat == currentDBFormat {
		return nil
	}

	return &DBFormatMismatchError{
		Expected: currentDBFormat,
		Actual:   stampedFormat,
	}
}

func hasLegacySchema(conn *sql.DB) (bool, error) {
	var table string
	err := conn.QueryRowContext(context.Background(), `
		SELECT name
		  FROM sqlite_master
		 WHERE type = 'table'
		   AND name NOT LIKE 'sqlite_%'
		   AND name != 'db_meta'
		 LIMIT 1`,
	).Scan(&table)
	if err == nil {
		return true, nil
	}
	if err == sql.ErrNoRows {
		return false, nil
	}
	return false, fmt.Errorf("inspect legacy schema: %w", err)
}

func stampDBFormat(conn *sql.DB, format string) error {
	if _, err := conn.ExecContext(context.Background(),
		`INSERT INTO db_meta (key, value) VALUES (?, ?)`, DBFormatKey, format,
	); err != nil {
		return fmt.Errorf("stamp db format: %w", err)
	}
	return nil
}

// Close closes the database.
func (d *DB) Close() error {
	return d.sql.Close()
}

// SQL returns the raw *sql.DB for advanced usage.
func (d *DB) SQL() *sql.DB {
	return d.sql
}

// EncodeEmbedding serialises []float32 → []byte (little-endian).
func EncodeEmbedding(v []float32) []byte {
	buf := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return buf
}

// DecodeEmbedding deserialises []byte → []float32.
func DecodeEmbedding(b []byte) []float32 {
	if len(b) == 0 {
		return nil
	}
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return v
}

// fts5Query converts a user query into an FTS5 match expression.
// Each word becomes a prefix term (word*) joined by OR.
// Possessives like "Caroline's" are split on apostrophe so "Caroline's" → "Caroline*" not "Carolines*".
// (FTS5 unicode61 tokenizer stores "caroline", not "carolines" — the 's' suffix causes a mismatch
// when the apostrophe is stripped rather than used as a split point.)
// Tokens shorter than 2 chars (possessive "s", etc.) are dropped.
func fts5Query(q string) string {
	// Split on whitespace first, then on apostrophes within each word.
	var rawTokens []string
	for _, word := range strings.Fields(q) {
		parts := strings.FieldsFunc(word, func(r rune) bool {
			return r == '\'' || r == '\u2019' || r == '`'
		})
		rawTokens = append(rawTokens, parts...)
	}

	terms := make([]string, 0, len(rawTokens))
	for _, w := range rawTokens {
		// Keep only letters, digits, and underscores — drop everything
		// else to avoid FTS5 syntax errors from special chars.
		w = strings.Map(func(r rune) rune {
			if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' {
				return r
			}
			return -1
		}, w)
		if len(w) < 2 && !isDigit(w) {
			continue // drop possessive "s", single letters — but keep digits like "4"
		}
		terms = append(terms, w+"*")
	}
	if len(terms) == 0 {
		return ""
	}
	// OR semantics: any matching term ranks higher; all terms present = highest score.
	return strings.Join(terms, " OR ")
}

func isDigit(s string) bool {
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return len(s) > 0
}

// CosineSimilarity computes cosine similarity between two vectors.
// Returns 0 if either vector is nil or lengths differ.
func CosineSimilarity(a, b []float32) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		ai, bi := float64(a[i]), float64(b[i])
		dot += ai * bi
		normA += ai * ai
		normB += bi * bi
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
