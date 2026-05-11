package brain

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// BrainStore is the pluggable persistence contract every Hanzo Brain
// backend implements — identical surface to the TS `BrainStore`
// interface, the Python `BaseVectorDB`, and the Rust `BrainStore`
// trait so a brain.db written by any runtime is readable by all
// other runtimes.
type BrainStore interface {
	Init(ctx context.Context) error
	UpsertPage(ctx context.Context, slug, content string, frontmatter map[string]any) error
	GetPage(ctx context.Context, slug string) (*Page, error)
	UpsertEdges(ctx context.Context, source string, edges []Edge) error
	EdgesFor(ctx context.Context, slug string, dir EdgeDir) ([]Edge, error)
	UpsertFact(ctx context.Context, fact Fact) error
	Recall(ctx context.Context, entity string, limit int, since string) ([]Fact, error)
	HybridSearch(ctx context.Context, query string, topK int) ([]SearchHit, error)
	Close() error
}

// EdgeDir filters edges_for queries by direction.
type EdgeDir int

const (
	DirBoth EdgeDir = iota
	DirIn
	DirOut
)

// Page is a single brain page (compiled truth + timeline).
type Page struct {
	Slug      string `json:"slug"`
	Content   string `json:"content"`
	UpdatedAt string `json:"updated_at"`
}

// Fact is a (subject, predicate, object) triple with provenance.
type Fact struct {
	ID         string  `json:"id,omitempty"`
	Subject    string  `json:"subject"`
	Predicate  string  `json:"predicate"`
	Object     string  `json:"object"`
	Source     string  `json:"source,omitempty"`
	TS         string  `json:"ts,omitempty"`
	Confidence float64 `json:"confidence,omitempty"`
}

// SearchHit is one row from hybrid search.
type SearchHit struct {
	Slug    string  `json:"slug"`
	Excerpt string  `json:"excerpt"`
	Score   float64 `json:"score"`
	Source  string  `json:"source"` // "vector" | "keyword" | "fused"
}

// Config selects + parameterises the chosen backend.
type Config struct {
	Backend         string // "sqlite" (default)
	DataDir         string // default ~/.hanzo/brain
	DBPath          string // explicit file; overrides DataDir
	EmbeddingModel  string
	EmbeddingAPIKey string
}

// BackendFactory builds a BrainStore for the given Config.
type BackendFactory func(cfg Config) (BrainStore, error)

var (
	backendsMu sync.RWMutex
	backends   = map[string]BackendFactory{}
)

// RegisterBackend wires a backend factory into the registry. Callers
// can override the built-in sqlite factory by re-registering "sqlite".
func RegisterBackend(name string, factory BackendFactory) {
	backendsMu.Lock()
	defer backendsMu.Unlock()
	backends[name] = factory
}

// ListBackends returns the names of every registered backend, in
// undefined order. Mirrors the TS `listBackends()` shape so callers
// can probe what's installed before opening.
func ListBackends() []string {
	backendsMu.RLock()
	defer backendsMu.RUnlock()
	out := make([]string, 0, len(backends))
	for k := range backends {
		out = append(out, k)
	}
	return out
}

// DefaultDataDir returns ~/.hanzo/brain — the canonical artifact root
// shared with hanzo-mcp, hanzo-dev, the bot, and the Python SDK.
func DefaultDataDir() string {
	if d := os.Getenv("HANZO_BRAIN_DIR"); d != "" {
		return d
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ".hanzo/brain"
	}
	return filepath.Join(home, ".hanzo", "brain")
}

// Open resolves the requested backend and returns an initialised
// BrainStore. Defaults to "sqlite" unless cfg.Backend or
// HANZO_BRAIN_BACKEND is set.
func Open(ctx context.Context, cfg Config) (BrainStore, error) {
	name := cfg.Backend
	if name == "" {
		name = os.Getenv("HANZO_BRAIN_BACKEND")
	}
	if name == "" {
		name = "sqlite"
	}
	backendsMu.RLock()
	factory, ok := backends[name]
	backendsMu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("brain: unknown backend %q. registered: %v", name, ListBackends())
	}
	store, err := factory(cfg)
	if err != nil {
		return nil, err
	}
	if err := store.Init(ctx); err != nil {
		return nil, err
	}
	return store, nil
}

// ErrNotFound is returned by GetPage when no row matches the slug.
var ErrNotFound = errors.New("brain: not found")
