package brain

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	_ "modernc.org/sqlite"
)

func init() {
	RegisterBackend("sqlite", func(cfg Config) (BrainStore, error) {
		path := cfg.DBPath
		if path == "" {
			dir := cfg.DataDir
			if dir == "" {
				dir = DefaultDataDir()
			}
			if err := os.MkdirAll(dir, 0o755); err != nil {
				return nil, err
			}
			path = filepath.Join(dir, "brain.db")
		}
		return &SqliteStore{path: path}, nil
	})
}

const sqliteSchema = `
CREATE TABLE IF NOT EXISTS pages (
  slug         TEXT PRIMARY KEY,
  content      TEXT NOT NULL,
  frontmatter  TEXT,
  updated_at   TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(content, content='pages', content_rowid='rowid');
CREATE TRIGGER IF NOT EXISTS pages_ai AFTER INSERT ON pages BEGIN
  INSERT INTO pages_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS pages_ad AFTER DELETE ON pages BEGIN
  INSERT INTO pages_fts(pages_fts, rowid, content) VALUES('delete', old.rowid, old.content);
END;
CREATE TRIGGER IF NOT EXISTS pages_au AFTER UPDATE ON pages BEGIN
  INSERT INTO pages_fts(pages_fts, rowid, content) VALUES('delete', old.rowid, old.content);
  INSERT INTO pages_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TABLE IF NOT EXISTS edges (
  source    TEXT NOT NULL,
  target    TEXT NOT NULL,
  type      TEXT NOT NULL,
  evidence  TEXT,
  PRIMARY KEY (source, target, type)
);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
CREATE INDEX IF NOT EXISTS idx_edges_type   ON edges(type);

CREATE TABLE IF NOT EXISTS facts (
  id          TEXT PRIMARY KEY,
  subject     TEXT NOT NULL,
  predicate   TEXT NOT NULL,
  object      TEXT NOT NULL,
  source      TEXT,
  ts          TEXT NOT NULL,
  confidence  REAL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject);
CREATE INDEX IF NOT EXISTS idx_facts_ts      ON facts(ts);
`

// SqliteStore is the single-binary, zero-infra default. Pure-Go driver
// (modernc.org/sqlite) — no cgo, builds anywhere.
type SqliteStore struct {
	path string
	db   *sql.DB
}

func (s *SqliteStore) Init(ctx context.Context) error {
	db, err := sql.Open("sqlite", s.path)
	if err != nil {
		return err
	}
	if _, err := db.ExecContext(ctx, sqliteSchema); err != nil {
		db.Close()
		return err
	}
	s.db = db
	return nil
}

func (s *SqliteStore) UpsertPage(ctx context.Context, slug, content string, frontmatter map[string]any) error {
	fm, _ := json.Marshal(frontmatter)
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO pages (slug, content, frontmatter, updated_at)
		VALUES (?, ?, ?, ?)
		ON CONFLICT(slug) DO UPDATE SET
		  content    = excluded.content,
		  frontmatter= excluded.frontmatter,
		  updated_at = excluded.updated_at`,
		slug, content, string(fm), time.Now().UTC().Format(time.RFC3339Nano),
	)
	return err
}

func (s *SqliteStore) GetPage(ctx context.Context, slug string) (*Page, error) {
	row := s.db.QueryRowContext(ctx,
		`SELECT slug, content, updated_at FROM pages WHERE slug = ?`, slug)
	var p Page
	if err := row.Scan(&p.Slug, &p.Content, &p.UpdatedAt); err != nil {
		if err == sql.ErrNoRows {
			return nil, nil
		}
		return nil, err
	}
	return &p, nil
}

func (s *SqliteStore) UpsertEdges(ctx context.Context, source string, edges []Edge) error {
	if len(edges) == 0 {
		return nil
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	stmt, err := tx.PrepareContext(ctx, `
		INSERT INTO edges (source, target, type, evidence) VALUES (?, ?, ?, ?)
		ON CONFLICT(source, target, type) DO UPDATE SET evidence = excluded.evidence`)
	if err != nil {
		tx.Rollback()
		return err
	}
	defer stmt.Close()
	for _, e := range edges {
		if _, err := stmt.ExecContext(ctx, e.Source, e.Target, string(e.Type), nullIfEmpty(e.Evidence)); err != nil {
			tx.Rollback()
			return err
		}
	}
	return tx.Commit()
}

func (s *SqliteStore) EdgesFor(ctx context.Context, slug string, dir EdgeDir) ([]Edge, error) {
	var q string
	var args []any
	switch dir {
	case DirIn:
		q = `SELECT source, target, type, evidence FROM edges WHERE target = ?`
		args = []any{slug}
	case DirOut:
		q = `SELECT source, target, type, evidence FROM edges WHERE source = ?`
		args = []any{slug}
	default: // Both
		q = `SELECT source, target, type, evidence FROM edges WHERE source = ? OR target = ?`
		args = []any{slug, slug}
	}
	rows, err := s.db.QueryContext(ctx, q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Edge
	for rows.Next() {
		var e Edge
		var ev sql.NullString
		var typ string
		if err := rows.Scan(&e.Source, &e.Target, &typ, &ev); err != nil {
			return nil, err
		}
		e.Type = EdgeType(typ)
		if ev.Valid {
			e.Evidence = ev.String
		}
		out = append(out, e)
	}
	return out, rows.Err()
}

func (s *SqliteStore) UpsertFact(ctx context.Context, f Fact) error {
	id := f.ID
	if id == "" {
		id = fmt.Sprintf("%s::%s::%d", f.Subject, f.Predicate, time.Now().UnixNano())
	}
	ts := f.TS
	if ts == "" {
		ts = time.Now().UTC().Format(time.RFC3339Nano)
	}
	conf := f.Confidence
	if conf == 0 {
		conf = 1.0
	}
	_, err := s.db.ExecContext(ctx, `
		INSERT INTO facts (id, subject, predicate, object, source, ts, confidence)
		VALUES (?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(id) DO UPDATE SET
		  object     = excluded.object,
		  ts         = excluded.ts,
		  confidence = excluded.confidence`,
		id, f.Subject, f.Predicate, f.Object, nullIfEmpty(f.Source), ts, conf,
	)
	return err
}

func (s *SqliteStore) Recall(ctx context.Context, entity string, limit int, since string) ([]Fact, error) {
	if limit <= 0 {
		limit = 50
	}
	if since == "" {
		since = "1970-01-01"
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, subject, predicate, object, source, ts, confidence
		FROM facts
		WHERE subject = ? AND ts >= ?
		ORDER BY ts DESC
		LIMIT ?`,
		entity, since, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Fact
	for rows.Next() {
		var f Fact
		var src sql.NullString
		if err := rows.Scan(&f.ID, &f.Subject, &f.Predicate, &f.Object, &src, &f.TS, &f.Confidence); err != nil {
			return nil, err
		}
		if src.Valid {
			f.Source = src.String
		}
		out = append(out, f)
	}
	return out, rows.Err()
}

func (s *SqliteStore) HybridSearch(ctx context.Context, query string, topK int) ([]SearchHit, error) {
	if topK <= 0 {
		topK = 5
	}
	rows, err := s.db.QueryContext(ctx, `
		SELECT pages.slug AS slug,
		       snippet(pages_fts, 0, '<b>', '</b>', '…', 32) AS excerpt
		FROM pages_fts
		JOIN pages ON pages.rowid = pages_fts.rowid
		WHERE pages_fts MATCH ?
		ORDER BY rank
		LIMIT ?`,
		query, topK*2)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var hits []SearchHit
	i := 0
	for rows.Next() {
		var h SearchHit
		if err := rows.Scan(&h.Slug, &h.Excerpt); err != nil {
			return nil, err
		}
		h.Score = 1.0 / float64(60+i)
		h.Source = "keyword"
		hits = append(hits, h)
		i++
	}
	if len(hits) > topK {
		hits = hits[:topK]
	}
	return hits, rows.Err()
}

func (s *SqliteStore) Close() error {
	if s.db == nil {
		return nil
	}
	return s.db.Close()
}

func nullIfEmpty(s string) any {
	if s == "" {
		return nil
	}
	return s
}
