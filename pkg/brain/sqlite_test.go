package brain

import (
	"context"
	"path/filepath"
	"testing"
)

func TestSqliteStore_EndToEnd(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	store, err := Open(ctx, Config{DBPath: filepath.Join(dir, "brain.db")})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { store.Close() })

	// Pages
	if err := store.UpsertPage(ctx, "people/alice", "Alice is the CEO of Acme. She founded Beta Co. She invested in Foobar.", map[string]any{"type": "person"}); err != nil {
		t.Fatalf("UpsertPage: %v", err)
	}
	p, err := store.GetPage(ctx, "people/alice")
	if err != nil || p == nil {
		t.Fatalf("GetPage returned %v, %v", p, err)
	}
	if p.Slug != "people/alice" {
		t.Fatalf("page slug: got %q", p.Slug)
	}

	// Edges — wire to graphlinks
	edges := ExtractEdges("people/alice", "Alice is the CEO of Acme. She founded Beta Co. She invested in Foobar.", "person")
	if len(edges) < 3 {
		t.Fatalf("expected >=3 edges, got %d", len(edges))
	}
	if err := store.UpsertEdges(ctx, "people/alice", edges); err != nil {
		t.Fatalf("UpsertEdges: %v", err)
	}
	out, err := store.EdgesFor(ctx, "people/alice", DirOut)
	if err != nil {
		t.Fatalf("EdgesFor: %v", err)
	}
	if len(out) != len(edges) {
		t.Fatalf("edge roundtrip: got %d / expected %d", len(out), len(edges))
	}

	// Facts
	if err := store.UpsertFact(ctx, Fact{
		Subject:   "people/alice",
		Predicate: "preference",
		Object:    "Direct replies, no fluff",
	}); err != nil {
		t.Fatalf("UpsertFact: %v", err)
	}
	facts, err := store.Recall(ctx, "people/alice", 10, "")
	if err != nil {
		t.Fatalf("Recall: %v", err)
	}
	if len(facts) != 1 || facts[0].Predicate != "preference" {
		t.Fatalf("recall: got %+v", facts)
	}

	// Hybrid search
	hits, err := store.HybridSearch(ctx, "Acme", 5)
	if err != nil {
		t.Fatalf("HybridSearch: %v", err)
	}
	if len(hits) != 1 || hits[0].Slug != "people/alice" {
		t.Fatalf("search: got %+v", hits)
	}
}

func TestRegistry_KnownBackends(t *testing.T) {
	names := ListBackends()
	found := false
	for _, n := range names {
		if n == "sqlite" {
			found = true
		}
	}
	if !found {
		t.Fatalf("sqlite missing from registry: %v", names)
	}
}
