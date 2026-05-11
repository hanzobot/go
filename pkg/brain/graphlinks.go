// Package brain implements the Hanzo Brain in Go.
//
// graphlinks.go is the zero-LLM typed-link extractor — Go port of
// @hanzo/bot-graph-links (TS), hanzo_memory.graph_links (Python),
// and hanzo_mcp::brain::graph_links (Rust).
//
// Same six edge types (mentions / attended / works_at / invested_in /
// founded / advises), same regex + role inference, same code-fence
// stripping. Pure — no I/O, no LLM.
package brain

import (
	"regexp"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// EdgeType is the typed-edge category for a relationship inferred
// between two brain entities.
type EdgeType string

const (
	EdgeMentions    EdgeType = "mentions"
	EdgeAttended    EdgeType = "attended"
	EdgeWorksAt     EdgeType = "works_at"
	EdgeInvestedIn  EdgeType = "invested_in"
	EdgeFounded     EdgeType = "founded"
	EdgeAdvises     EdgeType = "advises"
)

// Edge is a typed relationship from one slug to another, with the
// markdown phrase that triggered the inference recorded as evidence.
type Edge struct {
	Source   string   `json:"source"`
	Target   string   `json:"target"`
	Type     EdgeType `json:"type"`
	Evidence string   `json:"evidence,omitempty"`
}

// ── Patterns ────────────────────────────────────────────────────────

var (
	mdLink     = regexp.MustCompile(`\[([^\]]+)\]\(([^)#\s]+)\)`)
	bareSlug   = regexp.MustCompile(`(?i)(?:^|[^/\w])@?((?:people|companies|deals|projects|investors|firms)/[a-z0-9][a-z0-9-]*)`)
	codeFence  = regexp.MustCompile("(?s)```.*?```")
	inlineCode = regexp.MustCompile("`[^`\n]+`")
)

type rolePattern struct {
	re *regexp.Regexp
	t  EdgeType
}

// Order matters — first match wins per (source, target). Same set as TS/Py/Rust.
var rolePatterns = []rolePattern{
	{regexp.MustCompile(`(?i)\b(?:co-?)?founded\s+([^.\n]+?)(?:[.\n]|$)`), EdgeFounded},
	{regexp.MustCompile(`(?i)\bfounder\s+(?:and\s+\w+\s+)?of\s+([^.\n]+?)(?:[.\n]|$)`), EdgeFounded},

	{regexp.MustCompile(`(?i)\binvested\s+in\s+([^.\n]+?)(?:[.\n]|$)`), EdgeInvestedIn},
	{regexp.MustCompile(`(?i)\bled\s+([^.\n]+?)['` + "’" + `]s\s+(?:seed|series|round)`), EdgeInvestedIn},
	{regexp.MustCompile(`(?i)\bwrote\s+(?:a\s+)?check\s+into\s+([^.\n]+?)(?:[.\n]|$)`), EdgeInvestedIn},

	{regexp.MustCompile(`(?i)\badvises\s+([^.\n]+?)(?:[.\n]|$)`), EdgeAdvises},
	{regexp.MustCompile(`(?i)\badvisor\s+(?:to|at|for)\s+([^.\n]+?)(?:[.\n]|$)`), EdgeAdvises},

	{regexp.MustCompile(`(?i)\b(?:CEO|CTO|COO|CFO|VP|head\s+of\s+\w+|director)\s+of\s+([^.\n]+?)(?:[.\n]|$)`), EdgeWorksAt},
	{regexp.MustCompile(`\bjoined\s+([A-Z][^\s.]*(?:\s+[A-Z][^\s.]*)*)(?:\s+(?:as|in)\b|[\s.,;:!?\n]|$)`), EdgeWorksAt},
	{regexp.MustCompile(`(?i)\bworks\s+at\s+([^.\n]+?)(?:[.\n]|$)`), EdgeWorksAt},
}

// Slugify matches the cross-runtime slug convention:
// NFKD-normalize → drop combining marks → lowercase →
// `&` → " and " → non-alphanumeric → `-` → trim → cap at 80.
func Slugify(s string) string {
	s = norm.NFKD.String(s)
	var b strings.Builder
	for _, r := range s {
		if unicode.Is(unicode.Mn, r) {
			continue
		}
		b.WriteRune(r)
	}
	s = strings.ToLower(b.String())
	s = strings.ReplaceAll(s, "&", " and ")
	var out strings.Builder
	lastDash := true
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			out.WriteRune(r)
			lastDash = false
		} else if !lastDash {
			out.WriteByte('-')
			lastDash = true
		}
	}
	res := strings.Trim(out.String(), "-")
	if len(res) > 80 {
		res = res[:80]
	}
	return res
}

func stripCode(md string) string {
	return inlineCode.ReplaceAllString(codeFence.ReplaceAllString(md, ""), "")
}

func inferCategory(t EdgeType) string {
	switch t {
	case EdgeFounded, EdgeInvestedIn, EdgeWorksAt:
		return "companies"
	case EdgeAdvises:
		return "people"
	default:
		return "entities"
	}
}

// ExtractEdges returns the typed edges inferred from a single page.
// Pure — no I/O, no LLM. Safe to call on every page write.
//
//   pageType == "meeting"  →  markdown links emit `attended` not `mentions`.
func ExtractEdges(slug, content, pageType string) []Edge {
	cleaned := stripCode(content)
	seen := map[string]Edge{}
	add := func(e Edge) {
		k := e.Target + "::" + string(e.Type)
		if _, ok := seen[k]; !ok {
			seen[k] = e
		}
	}

	meeting := pageType == "meeting"

	// 1. Markdown links.
	for _, m := range mdLink.FindAllStringSubmatchIndex(cleaned, -1) {
		target := strings.TrimSpace(cleaned[m[4]:m[5]])
		if strings.HasPrefix(target, "http") || strings.HasPrefix(target, "/") || !strings.Contains(target, "/") {
			continue
		}
		t := EdgeMentions
		if meeting {
			t = EdgeAttended
		}
		add(Edge{Source: slug, Target: target, Type: t, Evidence: cleaned[m[0]:m[1]]})
	}

	// 2. Bare slug refs.
	for _, m := range bareSlug.FindAllStringSubmatchIndex(cleaned, -1) {
		target := strings.ToLower(cleaned[m[2]:m[3]])
		t := EdgeMentions
		if meeting {
			t = EdgeAttended
		}
		add(Edge{Source: slug, Target: target, Type: t, Evidence: cleaned[m[0]:m[1]]})
	}

	// 3. Role inference.
	for _, rp := range rolePatterns {
		m := rp.re.FindStringSubmatchIndex(cleaned)
		if m == nil {
			continue
		}
		raw := strings.TrimSpace(cleaned[m[2]:m[3]])
		raw = strings.TrimRight(raw, ".,;:!?")
		targetSlug := inferCategory(rp.t) + "/" + Slugify(raw)
		if strings.HasSuffix(targetSlug, "/") {
			continue
		}
		add(Edge{Source: slug, Target: targetSlug, Type: rp.t, Evidence: cleaned[m[0]:m[1]]})
	}

	out := make([]Edge, 0, len(seen))
	for _, e := range seen {
		out = append(out, e)
	}
	return out
}

// Reconcile returns the (add, remove) deltas between two edge sets,
// matching the contract used by all sibling runtimes.
func Reconcile(prior, next []Edge) (add, remove []Edge) {
	key := func(e Edge) string { return e.Source + "::" + e.Target + "::" + string(e.Type) }
	priorSet := map[string]struct{}{}
	for _, e := range prior {
		priorSet[key(e)] = struct{}{}
	}
	nextSet := map[string]struct{}{}
	for _, e := range next {
		nextSet[key(e)] = struct{}{}
	}
	for _, e := range next {
		if _, ok := priorSet[key(e)]; !ok {
			add = append(add, e)
		}
	}
	for _, e := range prior {
		if _, ok := nextSet[key(e)]; !ok {
			remove = append(remove, e)
		}
	}
	return
}
