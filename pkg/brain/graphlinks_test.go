package brain

import (
	"strings"
	"testing"
)

func has(t *testing.T, edges []Edge, target string, typ EdgeType) bool {
	t.Helper()
	for _, e := range edges {
		if e.Target == target && e.Type == typ {
			return true
		}
	}
	return false
}

// ── slugify ─────────────────────────────────────────────────────────

func TestSlugify(t *testing.T) {
	cases := map[string]string{
		"Acme AI Inc.":     "acme-ai-inc",
		"José's Pizza":     "jose-s-pizza",
		"Slack & Discord":  "slack-and-discord",
	}
	for in, want := range cases {
		got := Slugify(in)
		if got != want {
			t.Errorf("Slugify(%q) = %q; want %q", in, got, want)
		}
	}
}

// ── ExtractEdges ────────────────────────────────────────────────────

func TestExtractEdges_MentionsFromMdLink(t *testing.T) {
	edges := ExtractEdges("originals/idea-1", "Inspired by [Alice](people/alice) at Acme.", "")
	if !has(t, edges, "people/alice", EdgeMentions) {
		t.Fatalf("expected mentions edge to people/alice, got %+v", edges)
	}
}

func TestExtractEdges_MeetingEmitsAttended(t *testing.T) {
	edges := ExtractEdges(
		"meetings/2026-05-10",
		"Met with [Bob](people/bob) and [Carol](people/carol).",
		"meeting",
	)
	types := map[EdgeType]bool{}
	for _, e := range edges {
		types[e.Type] = true
	}
	if !types[EdgeAttended] {
		t.Fatal("expected an attended edge")
	}
	if types[EdgeMentions] {
		t.Fatal("meeting page should not emit mentions when attended fires")
	}
}

func TestExtractEdges_FoundedInference(t *testing.T) {
	edges := ExtractEdges("people/alice", "Alice co-founded Acme AI. She also runs Beta Co.", "")
	if !has(t, edges, "companies/acme-ai", EdgeFounded) {
		t.Fatalf("expected founded edge to companies/acme-ai, got %+v", edges)
	}
}

func TestExtractEdges_InvestedIn(t *testing.T) {
	for _, c := range []struct{ in, want string }{
		{"Dan invested in Foobar.", "companies/foobar"},
		{"Erin led Quux's seed round.", "companies/quux"},
	} {
		edges := ExtractEdges("people/x", c.in, "")
		if !has(t, edges, c.want, EdgeInvestedIn) {
			t.Errorf("input %q: missing invested_in → %s; got %+v", c.in, c.want, edges)
		}
	}
}

func TestExtractEdges_Advises(t *testing.T) {
	edges := ExtractEdges("people/frank", "Frank is an advisor to Globex.", "")
	if !has(t, edges, "people/globex", EdgeAdvises) {
		t.Fatalf("expected advises edge to people/globex, got %+v", edges)
	}
}

func TestExtractEdges_WorksAt(t *testing.T) {
	for _, c := range []struct{ in, want string }{
		{"Grace is the CEO of Acme.", "companies/acme"},
		{"Henry joined Initech in 2024.", "companies/initech"},
	} {
		edges := ExtractEdges("people/x", c.in, "")
		if !has(t, edges, c.want, EdgeWorksAt) {
			t.Errorf("input %q: missing works_at → %s; got %+v", c.in, c.want, edges)
		}
	}
}

func TestExtractEdges_StripsCodeFences(t *testing.T) {
	edges := ExtractEdges(
		"concepts/snippet",
		"Normal: [link](people/real). Code:\n```\nfake = [x](people/fake)\n```\n",
		"",
	)
	targets := map[string]bool{}
	for _, e := range edges {
		targets[e.Target] = true
	}
	if !targets["people/real"] {
		t.Fatal("expected people/real to be extracted")
	}
	if targets["people/fake"] {
		t.Fatal("people/fake came from a code fence and must be stripped")
	}
}

func TestExtractEdges_DedupSameTargetSameType(t *testing.T) {
	edges := ExtractEdges("originals/x", "[Alice](people/alice) and again [Alice](people/alice).", "")
	count := 0
	for _, e := range edges {
		if e.Target == "people/alice" && e.Type == EdgeMentions {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("expected dedup to 1; got %d", count)
	}
}

func TestExtractEdges_BareSlugRefs(t *testing.T) {
	edges := ExtractEdges("concepts/note", "See people/alice and companies/acme-ai.", "")
	targets := map[string]bool{}
	for _, e := range edges {
		targets[e.Target] = true
	}
	if !targets["people/alice"] || !targets["companies/acme-ai"] {
		t.Fatalf("bare slug refs missing; got targets %v", targets)
	}
}

// ── Reconcile ───────────────────────────────────────────────────────

func TestReconcile_AddRemove(t *testing.T) {
	prior := []Edge{
		{Source: "a", Target: "x", Type: EdgeMentions},
		{Source: "a", Target: "y", Type: EdgeMentions},
	}
	next := []Edge{
		{Source: "a", Target: "y", Type: EdgeMentions},
		{Source: "a", Target: "z", Type: EdgeMentions},
	}
	add, remove := Reconcile(prior, next)
	if len(add) != 1 || add[0].Target != "z" {
		t.Fatalf("expected add=[z], got %+v", add)
	}
	if len(remove) != 1 || remove[0].Target != "x" {
		t.Fatalf("expected remove=[x], got %+v", remove)
	}
}

func TestReconcile_NoChange(t *testing.T) {
	same := []Edge{{Source: "a", Target: "x", Type: EdgeMentions}}
	add, remove := Reconcile(same, same)
	if len(add) != 0 || len(remove) != 0 {
		t.Fatalf("identical sets should produce empty deltas; got add=%v remove=%v", add, remove)
	}
}

// Sanity: target slugs never contain whitespace or uppercase after slugify.
func TestSlugifyInvariants(t *testing.T) {
	got := Slugify("Some Name With Many   Spaces & Symbols!!")
	if strings.ContainsAny(got, " ABCDEFGHIJKLMNOPQRSTUVWXYZ") {
		t.Fatalf("slug contains whitespace or uppercase: %q", got)
	}
}
