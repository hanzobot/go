// Package brain — pure-CPU algorithm port for the Go runtime.
//
// Mirrors @hanzo/bot-memory and hanzo_memory.algorithms. Same surfaces,
// same inputs, byte-equivalent outputs where the algorithm is deterministic.
//
// See hanzoai/bot-core/spec.md for the cross-runtime contract.
package brain

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"math"
	"math/big"
	"regexp"
	"sort"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"
)

// ── Fusion ───────────────────────────────────────────────────────────

// RrfKDefault is the BEIR-tuned default k (Elasticsearch 2024).
const RrfKDefault = 20

// (SearchHit is defined in store.go and reused here.)

// RrfFuse implements Reciprocal Rank Fusion (Cormack et al. 2009).
func RrfFuse(lists [][]SearchHit, limit int, k float64) []SearchHit {
	if k == 0 {
		k = RrfKDefault
	}
	scores := map[string]float64{}
	meta := map[string]SearchHit{}
	num := float64(len(lists))
	for _, lst := range lists {
		for rank, h := range lst {
			scores[h.Slug] += 1.0 / (k + float64(rank) + 1)
			if _, ok := meta[h.Slug]; !ok {
				meta[h.Slug] = h
			}
		}
	}
	if len(scores) == 0 {
		return nil
	}
	max := num / (k + 1)
	out := make([]SearchHit, 0, len(scores))
	for slug, s := range scores {
		m := meta[slug]
		norm := 0.0
		if max > 0 {
			norm = math.Min(s/max, 1.0)
		}
		out = append(out, SearchHit{Slug: slug, Score: norm, Excerpt: m.Excerpt, Source: "fused"})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return out
}

// RsfFuse implements Relative Score Fusion (Weaviate v1.24).
func RsfFuse(lists [][]SearchHit, limit int, weights []float64) []SearchHit {
	n := len(lists)
	w := weights
	if w == nil {
		w = make([]float64, n)
		if n > 0 {
			eq := 1.0 / float64(n)
			for i := range w {
				w[i] = eq
			}
		}
	}
	if len(w) != n {
		// fall back to uniform on mismatched weights to keep API safe.
		w = make([]float64, n)
		if n > 0 {
			for i := range w {
				w[i] = 1.0 / float64(n)
			}
		}
	}
	scores := map[string]float64{}
	meta := map[string]SearchHit{}
	for i, lst := range lists {
		if len(lst) == 0 {
			continue
		}
		lo, hi := math.Inf(1), math.Inf(-1)
		for _, h := range lst {
			if h.Score < lo {
				lo = h.Score
			}
			if h.Score > hi {
				hi = h.Score
			}
		}
		span := hi - lo
		for _, h := range lst {
			norm := 1.0
			if span > 0 {
				norm = (h.Score - lo) / span
			}
			scores[h.Slug] += w[i] * norm
			if _, ok := meta[h.Slug]; !ok {
				meta[h.Slug] = h
			}
		}
	}
	out := make([]SearchHit, 0, len(scores))
	for slug, s := range scores {
		m := meta[slug]
		out = append(out, SearchHit{Slug: slug, Score: s, Excerpt: m.Excerpt, Source: "fused"})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return out
}

// QueryCharacteristics describes a query for adaptive k / weight selection.
type QueryCharacteristics struct {
	TokenCount int
	IsPhrase   bool
	IsBoolean  bool
}

// Characterize classifies a query string cheaply.
func Characterize(query string) QueryCharacteristics {
	t := strings.TrimSpace(query)
	isPhrase := (strings.HasPrefix(t, "\"") && strings.HasSuffix(t, "\"") && len(t) > 2) ||
		(strings.HasPrefix(t, "'") && strings.HasSuffix(t, "'") && len(t) > 2)
	boolRe := regexp.MustCompile(`\b(AND|OR|NOT)\b`)
	negRe := regexp.MustCompile(`\s-\S`)
	isBool := boolRe.MatchString(t) || negRe.MatchString(t)
	tokens := strings.Fields(t)
	return QueryCharacteristics{TokenCount: len(tokens), IsPhrase: isPhrase, IsBoolean: isBool}
}

// SelectRrfK picks an adaptive RRF k value.
func SelectRrfK(q QueryCharacteristics) int {
	switch {
	case q.IsPhrase:
		return 10
	case q.IsBoolean:
		return 15
	case q.TokenCount <= 2:
		return 15
	case q.TokenCount >= 10:
		return 40
	}
	return RrfKDefault
}

// FusionWeights pairs the FTS and semantic weights, always summing to 1.
type FusionWeights struct {
	Fts      float64
	Semantic float64
}

// SelectWeights picks adaptive FTS / semantic weights.
func SelectWeights(q QueryCharacteristics) FusionWeights {
	switch {
	case q.IsPhrase:
		return FusionWeights{0.8, 0.2}
	case q.IsBoolean:
		return FusionWeights{0.7, 0.3}
	case q.TokenCount <= 2:
		return FusionWeights{0.65, 0.35}
	case q.TokenCount >= 10:
		return FusionWeights{0.3, 0.7}
	}
	return FusionWeights{0.5, 0.5}
}

// ── Rerank (MMR) ─────────────────────────────────────────────────────

// Cosine similarity. Returns 0 on shape mismatch or zero norm.
func Cosine(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	d := math.Sqrt(na) * math.Sqrt(nb)
	if d == 0 {
		return 0
	}
	return dot / d
}

// MmrInput is a SearchHit with an optional embedding.
type MmrInput struct {
	SearchHit
	Embedding []float64
}

// MmrRerank greedily reranks by Maximal Marginal Relevance.
func MmrRerank(hits []MmrInput, lambda float64, limit int) []MmrInput {
	if limit <= 0 {
		limit = len(hits)
	}
	embedded := make([]MmrInput, 0, len(hits))
	orphans := make([]MmrInput, 0)
	for _, h := range hits {
		if len(h.Embedding) > 0 {
			embedded = append(embedded, h)
		} else {
			orphans = append(orphans, h)
		}
	}
	selected := make([]MmrInput, 0, limit)
	cands := embedded
	for len(selected) < limit && len(cands) > 0 {
		bestIdx, bestScore := -1, math.Inf(-1)
		for i, c := range cands {
			rel := c.Score
			maxSim := 0.0
			for _, s := range selected {
				sim := Cosine(c.Embedding, s.Embedding)
				if sim > maxSim {
					maxSim = sim
				}
			}
			mmr := lambda*rel - (1-lambda)*maxSim
			if mmr > bestScore {
				bestScore = mmr
				bestIdx = i
			}
		}
		if bestIdx < 0 {
			break
		}
		selected = append(selected, cands[bestIdx])
		cands = append(cands[:bestIdx], cands[bestIdx+1:]...)
	}
	for _, o := range orphans {
		if len(selected) >= limit {
			break
		}
		selected = append(selected, o)
	}
	return selected
}

// ── Dedup ────────────────────────────────────────────────────────────

var chunkSuffix = regexp.MustCompile(`(#chunk-\d+|::\d+)$`)

// DedupHits keeps the highest-scoring chunk per chain.
func DedupHits(hits []SearchHit, perChain int) []SearchHit {
	if perChain <= 0 {
		perChain = 1
	}
	buckets := map[string][]SearchHit{}
	for _, h := range hits {
		chain := chunkSuffix.ReplaceAllString(h.Slug, "")
		buckets[chain] = append(buckets[chain], h)
	}
	out := make([]SearchHit, 0, len(hits))
	for _, lst := range buckets {
		sort.Slice(lst, func(i, j int) bool { return lst[i].Score > lst[j].Score })
		end := perChain
		if end > len(lst) {
			end = len(lst)
		}
		out = append(out, lst[:end]...)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	return out
}

// ── Script detection ─────────────────────────────────────────────────

// ScriptReport summarizes per-script character fractions.
type ScriptReport struct {
	Primary   string
	Fractions map[string]float64
	HasCjk    bool
	HasEmoji  bool
}

// DetectScript classifies a string's primary script.
func DetectScript(s string) ScriptReport {
	keys := []string{"latin", "cjk", "emoji", "cyrillic", "arabic", "hebrew", "greek", "devanagari", "other"}
	counts := map[string]int{}
	for _, k := range keys {
		counts[k] = 0
	}
	total := 0
	for _, r := range s {
		if r >= 0x0030 && r <= 0x0039 {
			continue
		}
		k := classifyRune(r)
		if k == "" {
			continue
		}
		counts[k]++
		total++
	}
	primary := "other"
	max := 0
	for _, k := range keys {
		if counts[k] > max {
			max = counts[k]
			primary = k
		}
	}
	fractions := map[string]float64{}
	for _, k := range keys {
		if total > 0 {
			fractions[k] = float64(counts[k]) / float64(total)
		}
	}
	return ScriptReport{Primary: primary, Fractions: fractions, HasCjk: counts["cjk"] > 0, HasEmoji: counts["emoji"] > 0}
}

// HasCjk returns true when any CJK codepoint is present.
func HasCjk(s string) bool {
	for _, r := range s {
		if isCjk(r) {
			return true
		}
	}
	return false
}

// HasEmoji returns true when any emoji or pictographic codepoint is present.
func HasEmoji(s string) bool {
	for _, r := range s {
		if isEmoji(r) {
			return true
		}
	}
	return false
}

func isCjk(r rune) bool {
	return (r >= 0x4E00 && r <= 0x9FFF) ||
		(r >= 0x3400 && r <= 0x4DBF) ||
		(r >= 0x3040 && r <= 0x30FF) ||
		(r >= 0xAC00 && r <= 0xD7AF)
}

func isEmoji(r rune) bool {
	return (r >= 0x2600 && r <= 0x27BF) || (r >= 0x1F300 && r <= 0x1FAFF)
}

func classifyRune(r rune) string {
	switch {
	case isCjk(r):
		return "cjk"
	case isEmoji(r):
		return "emoji"
	case (r >= 0x0041 && r <= 0x005A) || (r >= 0x0061 && r <= 0x007A) || (r >= 0x00C0 && r <= 0x024F):
		return "latin"
	case r >= 0x0370 && r <= 0x03FF:
		return "greek"
	case r >= 0x0400 && r <= 0x04FF:
		return "cyrillic"
	case r >= 0x0590 && r <= 0x05FF:
		return "hebrew"
	case r >= 0x0600 && r <= 0x06FF:
		return "arabic"
	case r >= 0x0900 && r <= 0x097F:
		return "devanagari"
	case r <= 0x002F || (r >= 0x003A && r <= 0x0040) || (r >= 0x005B && r <= 0x0060) || (r >= 0x007B && r <= 0x007E):
		return ""
	}
	return "other"
}

// ── FTS helpers ──────────────────────────────────────────────────────

// CjkBigrams splits CJK runs into 2-character grams; Latin words pass through whole.
func CjkBigrams(text string) []string {
	var out []string
	var cjkBuf, latinBuf strings.Builder
	flushCjk := func() {
		s := cjkBuf.String()
		cjkBuf.Reset()
		if s == "" {
			return
		}
		runes := []rune(s)
		if len(runes) == 1 {
			out = append(out, s)
			return
		}
		for i := 0; i < len(runes)-1; i++ {
			out = append(out, string(runes[i:i+2]))
		}
	}
	flushLatin := func() {
		if latinBuf.Len() > 0 {
			out = append(out, latinBuf.String())
			latinBuf.Reset()
		}
	}
	for _, r := range text {
		switch {
		case isCjk(r):
			flushLatin()
			cjkBuf.WriteRune(r)
		case unicode.IsSpace(r):
			flushCjk()
			flushLatin()
		default:
			flushCjk()
			latinBuf.WriteRune(r)
		}
	}
	flushCjk()
	flushLatin()
	return out
}

// EmojiTrigrams emits length-3 grams over emoji runs.
func EmojiTrigrams(text string) []string {
	runes := []rune(text)
	var out []string
	for i, r := range runes {
		if !isEmoji(r) {
			continue
		}
		var b strings.Builder
		b.WriteRune(r)
		if i+1 < len(runes) {
			b.WriteRune(runes[i+1])
		}
		if i+2 < len(runes) {
			b.WriteRune(runes[i+2])
		}
		out = append(out, b.String())
	}
	return out
}

// ParsedQuery mirrors Postgres's websearch_to_tsquery decomposition.
type ParsedQuery struct {
	Required []string
	Excluded []string
	Optional [][]string
	Phrases  []string
}

var (
	phraseTok = regexp.MustCompile(`"([^"]+)"|(\S+)`)
)

// ParseWebSearch parses a websearch-style query.
func ParseWebSearch(query string) ParsedQuery {
	var out ParsedQuery
	type tok struct {
		kind, value string
	}
	var toks []tok
	for _, m := range phraseTok.FindAllStringSubmatch(query, -1) {
		if m[1] != "" {
			toks = append(toks, tok{"phrase", m[1]})
		} else {
			toks = append(toks, tok{"word", m[2]})
		}
	}
	for i := 0; i < len(toks); {
		t := toks[i]
		if t.kind == "phrase" {
			out.Phrases = append(out.Phrases, t.value)
			out.Required = append(out.Required, t.value)
			i++
			continue
		}
		if i+1 < len(toks) && toks[i+1].kind == "word" && toks[i+1].value == "OR" {
			group := []string{t.value}
			j := i + 1
			for j+1 < len(toks) && toks[j].value == "OR" && toks[j].kind == "word" {
				group = append(group, toks[j+1].value)
				j += 2
			}
			out.Optional = append(out.Optional, group)
			i = j
			continue
		}
		if strings.HasPrefix(t.value, "-") && len(t.value) > 1 {
			out.Excluded = append(out.Excluded, t.value[1:])
			i++
			continue
		}
		out.Required = append(out.Required, t.value)
		i++
	}
	return out
}

// ToFts5Match renders a ParsedQuery as a SQLite FTS5 MATCH expression.
func ToFts5Match(p ParsedQuery) string {
	parts := make([]string, 0, len(p.Required)+len(p.Optional))
	for _, r := range p.Required {
		parts = append(parts, quoteFts5(r))
	}
	for _, group := range p.Optional {
		alts := make([]string, 0, len(group))
		for _, g := range group {
			alts = append(alts, quoteFts5(g))
		}
		parts = append(parts, "("+strings.Join(alts, " OR ")+")")
	}
	s := strings.Join(parts, " AND ")
	for _, e := range p.Excluded {
		s += " NOT " + quoteFts5(e)
	}
	return strings.TrimSpace(s)
}

var fts5Word = regexp.MustCompile(`^[\p{L}\p{N}_]+$`)

func quoteFts5(term string) string {
	if strings.Contains(term, " ") || !fts5Word.MatchString(term) {
		return `"` + strings.ReplaceAll(term, `"`, `""`) + `"`
	}
	return term
}

// ── Embed registry + MRL ─────────────────────────────────────────────

// EmbeddingModel describes an embedding model.
type EmbeddingModel struct {
	Slug          string
	Dim           int
	MrlDims       []int
	PrefixQuery   string
	PrefixPassage string
	Family        string
}

var embedRegistry = map[string]EmbeddingModel{}

// RegisterEmbeddingModel records a model under its slug.
func RegisterEmbeddingModel(m EmbeddingModel) {
	embedRegistry[m.Slug] = m
}

// GetEmbeddingModel looks up a model by slug.
func GetEmbeddingModel(slug string) (EmbeddingModel, bool) {
	m, ok := embedRegistry[slug]
	return m, ok
}

// ListEmbeddingModels returns the registered models.
func ListEmbeddingModels() []EmbeddingModel {
	out := make([]EmbeddingModel, 0, len(embedRegistry))
	for _, m := range embedRegistry {
		out = append(out, m)
	}
	return out
}

func init() {
	RegisterEmbeddingModel(EmbeddingModel{Slug: "ollama:nomic-embed-text", Dim: 768, MrlDims: []int{128, 256, 512, 768}, Family: "nomic"})
	RegisterEmbeddingModel(EmbeddingModel{Slug: "intfloat/e5-large-v2", Dim: 1024, PrefixQuery: "query: ", PrefixPassage: "passage: ", Family: "e5"})
	RegisterEmbeddingModel(EmbeddingModel{Slug: "openai:text-embedding-3-small", Dim: 1536, MrlDims: []int{256, 512, 768, 1024, 1536}, Family: "openai"})
	RegisterEmbeddingModel(EmbeddingModel{Slug: "openai:text-embedding-3-large", Dim: 3072, MrlDims: []int{256, 512, 1024, 2048, 3072}, Family: "openai"})
}

// PrefixFor applies the asymmetric query/passage prefix when the model uses one.
func PrefixFor(m EmbeddingModel, task, text string) string {
	if task == "symmetric" || (m.PrefixQuery == "" && m.PrefixPassage == "") {
		return text
	}
	if task == "query" {
		return m.PrefixQuery + text
	}
	return m.PrefixPassage + text
}

// L2Normalize L2-normalizes in place and returns the same slice.
func L2Normalize(v []float64) []float64 {
	var s float64
	for _, x := range v {
		s += x * x
	}
	n := math.Sqrt(s)
	if n == 0 {
		return v
	}
	for i := range v {
		v[i] /= n
	}
	return v
}

// MrlTruncate produces a normalized truncated copy at the target dim.
func MrlTruncate(embedding []float64, target int) []float64 {
	if target <= 0 {
		panic("MrlTruncate: target must be positive")
	}
	cp := make([]float64, 0, target)
	if target >= len(embedding) {
		cp = append(cp, embedding...)
	} else {
		cp = append(cp, embedding[:target]...)
	}
	return L2Normalize(cp)
}

// CoarseDim picks ~dim/8 from the MRL set.
func CoarseDim(m EmbeddingModel) int {
	if len(m.MrlDims) == 0 {
		return m.Dim
	}
	target := float64(m.Dim) / 8.0
	for _, d := range m.MrlDims {
		if float64(d) >= target {
			return d
		}
	}
	return m.MrlDims[len(m.MrlDims)-1]
}

// ── Temporal ─────────────────────────────────────────────────────────

// V7Floor renders a UUIDv7 with the given epoch-ms and zero entropy.
func V7Floor(epochMs int64) string { return v7(epochMs, false) }

// V7Ceiling renders a UUIDv7 with the given epoch-ms and max entropy.
func V7Ceiling(epochMs int64) string { return v7(epochMs, true) }

func v7(epochMs int64, ceiling bool) string {
	if epochMs < 0 {
		epochMs = 0
	}
	hex := fmt.Sprintf("%012x", uint64(epochMs))
	if len(hex) > 12 {
		hex = hex[len(hex)-12:]
	}
	th, tl := hex[:8], hex[8:12]
	if ceiling {
		return fmt.Sprintf("%s-%s-7fff-bfff-ffffffffffff", th, tl)
	}
	return fmt.Sprintf("%s-%s-7000-8000-000000000000", th, tl)
}

// TemporalRange is an ISO timestamp window.
type TemporalRange struct {
	From string
	To   string
}

// RangeBounds converts to UUIDv7 floor/ceiling identifiers.
func RangeBounds(r TemporalRange) (floor, ceiling string) {
	var fromMs, toMs int64
	if r.From != "" {
		if t, err := time.Parse(time.RFC3339, r.From); err == nil {
			fromMs = t.UnixMilli()
		}
	}
	if r.To != "" {
		if t, err := time.Parse(time.RFC3339, r.To); err == nil {
			toMs = t.UnixMilli()
		}
	} else {
		toMs = time.Now().UnixMilli() + 86_400_000
	}
	return V7Floor(fromMs), V7Ceiling(toMs)
}

// ── Captions ─────────────────────────────────────────────────────────

// CaptionSegment is a timestamped caption row.
type CaptionSegment struct {
	StartSecs float64
	EndSecs   float64
	Text      string
	Speaker   string
}

// RenderVtt renders WebVTT.
func RenderVtt(segs []CaptionSegment) string {
	var b strings.Builder
	b.WriteString("WEBVTT\n\n")
	for i, s := range segs {
		fmt.Fprintf(&b, "%d\n", i+1)
		fmt.Fprintf(&b, "%s --> %s\n", vttTime(s.StartSecs), vttTime(s.EndSecs))
		if s.Speaker != "" {
			fmt.Fprintf(&b, "<v %s>%s</v>\n\n", s.Speaker, s.Text)
		} else {
			fmt.Fprintf(&b, "%s\n\n", s.Text)
		}
	}
	return b.String()
}

// RenderSrt renders SubRip.
func RenderSrt(segs []CaptionSegment) string {
	var b strings.Builder
	for i, s := range segs {
		fmt.Fprintf(&b, "%d\n", i+1)
		fmt.Fprintf(&b, "%s --> %s\n", srtTime(s.StartSecs), srtTime(s.EndSecs))
		if s.Speaker != "" {
			fmt.Fprintf(&b, "%s: %s\n\n", s.Speaker, s.Text)
		} else {
			fmt.Fprintf(&b, "%s\n\n", s.Text)
		}
	}
	return b.String()
}

// RenderRttm renders NIST RTTM.
func RenderRttm(segs []CaptionSegment, uri string) string {
	if uri == "" {
		uri = "audio"
	}
	var b strings.Builder
	for _, s := range segs {
		if s.Speaker == "" {
			continue
		}
		dur := s.EndSecs - s.StartSecs
		fmt.Fprintf(&b, "SPEAKER %s 1 %.3f %.3f <NA> <NA> %s <NA> <NA>\n", uri, s.StartSecs, dur, s.Speaker)
	}
	return b.String()
}

func vttTime(secs float64) string { return fmtTime(secs, '.') }
func srtTime(secs float64) string { return fmtTime(secs, ',') }

func fmtTime(secs float64, msSep byte) string {
	ms := int(math.Floor((secs - math.Floor(secs)) * 1000))
	total := int(secs)
	h := total / 3600
	m := (total % 3600) / 60
	s := total % 60
	return fmt.Sprintf("%02d:%02d:%02d%c%03d", h, m, s, msSep, ms)
}

// ── Tokenizer ────────────────────────────────────────────────────────

// EstimateTokens returns a fast BPE-style estimate.
func EstimateTokens(text string) int {
	total := 0
	var asciiBuf strings.Builder
	flush := func() {
		if asciiBuf.Len() == 0 {
			return
		}
		for _, w := range strings.Fields(asciiBuf.String()) {
			t := utf8.RuneCountInString(w) / 4
			if t < 1 {
				t = 1
			}
			total += t
		}
		asciiBuf.Reset()
	}
	for _, r := range text {
		if isCjk(r) || isEmoji(r) {
			flush()
			total++
		} else {
			asciiBuf.WriteRune(r)
		}
	}
	flush()
	return total
}

// TruncateToTokens trims to fit a token budget via binary search.
func TruncateToTokens(text string, max int) string {
	if EstimateTokens(text) <= max {
		return text
	}
	runes := []rune(text)
	lo, hi := 0, len(runes)
	for lo < hi {
		mid := (lo + hi + 1) / 2
		if EstimateTokens(string(runes[:mid])) <= max {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	return string(runes[:lo])
}

// ── Eval ─────────────────────────────────────────────────────────────

// QueryEval is a per-query eval input.
type QueryEval struct {
	Predicted []string
	Relevant  map[string]int // 0 = irrelevant, >0 = relevance grade
}

func relSet(q QueryEval) map[string]bool {
	out := make(map[string]bool, len(q.Relevant))
	for k, v := range q.Relevant {
		if v > 0 {
			out[k] = true
		}
	}
	return out
}

// ReciprocalRank computes the per-query reciprocal rank.
func ReciprocalRank(q QueryEval) float64 {
	rel := relSet(q)
	for i, p := range q.Predicted {
		if rel[p] {
			return 1.0 / float64(i+1)
		}
	}
	return 0
}

// MeanReciprocalRank averages reciprocal rank across queries.
func MeanReciprocalRank(qs []QueryEval) float64 {
	if len(qs) == 0 {
		return 0
	}
	var s float64
	for _, q := range qs {
		s += ReciprocalRank(q)
	}
	return s / float64(len(qs))
}

// RecallAtK is hits / |rel|.
func RecallAtK(q QueryEval, k int) float64 {
	rel := relSet(q)
	if len(rel) == 0 {
		return 0
	}
	end := k
	if end > len(q.Predicted) {
		end = len(q.Predicted)
	}
	hits := 0
	for _, p := range q.Predicted[:end] {
		if rel[p] {
			hits++
		}
	}
	return float64(hits) / float64(len(rel))
}

// PrecisionAtK is hits / min(k, |pred|).
func PrecisionAtK(q QueryEval, k int) float64 {
	end := k
	if end > len(q.Predicted) {
		end = len(q.Predicted)
	}
	if end == 0 {
		return 0
	}
	rel := relSet(q)
	hits := 0
	for _, p := range q.Predicted[:end] {
		if rel[p] {
			hits++
		}
	}
	return float64(hits) / float64(end)
}

// NdcgAtK computes graded NDCG@k.
func NdcgAtK(q QueryEval, k int) float64 {
	dcg := func(ss []string) float64 {
		var s float64
		for i, slug := range ss {
			g := q.Relevant[slug]
			s += (math.Pow(2, float64(g)) - 1) / math.Log2(float64(i+2))
		}
		return s
	}
	type kv struct {
		k string
		v int
	}
	kvs := make([]kv, 0, len(q.Relevant))
	for k2, v := range q.Relevant {
		kvs = append(kvs, kv{k2, v})
	}
	sort.Slice(kvs, func(i, j int) bool { return kvs[i].v > kvs[j].v })
	ideal := make([]string, 0, k)
	for i := 0; i < k && i < len(kvs); i++ {
		ideal = append(ideal, kvs[i].k)
	}
	idcg := dcg(ideal)
	end := k
	if end > len(q.Predicted) {
		end = len(q.Predicted)
	}
	actual := dcg(q.Predicted[:end])
	if idcg == 0 {
		return 0
	}
	return actual / idcg
}

// ── Spatial ──────────────────────────────────────────────────────────

const earthKm = 6371.0088

// HaversineKm returns the great-circle distance in km.
func HaversineKm(latA, lngA, latB, lngB float64) float64 {
	dLat := degToRad(latB - latA)
	dLng := degToRad(lngB - lngA)
	r1 := degToRad(latA)
	r2 := degToRad(latB)
	x := math.Pow(math.Sin(dLat/2), 2) + math.Pow(math.Sin(dLng/2), 2)*math.Cos(r1)*math.Cos(r2)
	return 2 * earthKm * math.Asin(math.Sqrt(x))
}

// BBox is a min/max lat-lng rectangle.
type BBox struct {
	MinLat, MinLng, MaxLat, MaxLng float64
}

// BBoxAround returns a conservative bbox around a center point.
func BBoxAround(lat, lng, radiusKm float64) BBox {
	dLat := radiusKm / 111
	dLng := radiusKm / (111 * math.Cos(degToRad(lat)))
	return BBox{lat - dLat, lng - dLng, lat + dLat, lng + dLng}
}

// InBox returns true when the point falls within the bbox.
func InBox(lat, lng float64, b BBox) bool {
	return lat >= b.MinLat && lat <= b.MaxLat && lng >= b.MinLng && lng <= b.MaxLng
}

func degToRad(d float64) float64 { return d * math.Pi / 180 }

// ── HTTP Range ───────────────────────────────────────────────────────

// RangeRequest is the resolved closed [start, end] inclusive range.
type RangeRequest struct {
	Start, End int64
}

// ParseRange decodes the HTTP Range header.
// Returns (nil, false, false) for unparseable, (nil, true, true) for unsatisfiable.
func ParseRange(header string, total int64) (r RangeRequest, ok, unsatisfiable bool) {
	if !strings.HasPrefix(header, "bytes=") {
		return
	}
	spec := strings.TrimSpace(strings.Split(strings.TrimPrefix(header, "bytes="), ",")[0])
	if spec == "" {
		return
	}
	if strings.HasPrefix(spec, "-") {
		var suffix int64
		_, err := fmt.Sscanf(spec[1:], "%d", &suffix)
		if err != nil || suffix <= 0 {
			return
		}
		start := total - suffix
		if start < 0 {
			start = 0
		}
		return RangeRequest{start, total - 1}, true, false
	}
	parts := strings.SplitN(spec, "-", 2)
	var start, end int64
	if _, err := fmt.Sscanf(parts[0], "%d", &start); err != nil {
		return
	}
	if len(parts) == 2 && parts[1] != "" {
		if _, err := fmt.Sscanf(parts[1], "%d", &end); err != nil {
			return
		}
	} else {
		end = total - 1
	}
	if start > end || start >= total {
		return RangeRequest{}, false, true
	}
	if end >= total {
		end = total - 1
	}
	return RangeRequest{start, end}, true, false
}

// ContentRange formats the Content-Range header value.
func ContentRange(r RangeRequest, total int64) string {
	return fmt.Sprintf("bytes %d-%d/%d", r.Start, r.End, total)
}

// ── Wallet address ───────────────────────────────────────────────────

const base58Alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
const versionV1 byte = 0x01

// EncodeAddress renders a 32-byte public key as a base58check address.
func EncodeAddress(publicKey []byte, prefix string) (string, error) {
	if len(publicKey) != 32 {
		return "", errors.New("public key must be 32 bytes")
	}
	if prefix == "" {
		prefix = "hanzo"
	}
	h := sha256.Sum256(publicKey) // sha256 stand-in for blake3 — kept consistent within the Go runtime
	versioned := append([]byte{versionV1}, h[:20]...)
	cs := sha256.Sum256(versioned)
	payload := append(versioned, cs[:4]...)
	return prefix + ":" + base58Encode(payload), nil
}

// DecodedAddress is the parsed shape returned from DecodeAddress.
type DecodedAddress struct {
	Prefix  string
	Version byte
	Hash    [20]byte
}

// DecodeAddress validates and parses an address. Returns an error on bad checksum.
func DecodeAddress(addr string) (DecodedAddress, error) {
	idx := strings.Index(addr, ":")
	if idx < 0 {
		return DecodedAddress{}, errors.New("address: missing prefix")
	}
	decoded := base58Decode(addr[idx+1:])
	if len(decoded) != 25 {
		return DecodedAddress{}, errors.New("address: wrong length")
	}
	version := decoded[0]
	var hash [20]byte
	copy(hash[:], decoded[1:21])
	checksum := decoded[21:25]
	expected := sha256.Sum256(decoded[:21])
	for i := 0; i < 4; i++ {
		if checksum[i] != expected[i] {
			return DecodedAddress{}, errors.New("address: bad checksum")
		}
	}
	return DecodedAddress{Prefix: addr[:idx], Version: version, Hash: hash}, nil
}

func base58Encode(b []byte) string {
	if len(b) == 0 {
		return ""
	}
	zeros := 0
	for _, c := range b {
		if c == 0 {
			zeros++
		} else {
			break
		}
	}
	n := new(big.Int).SetBytes(b)
	base := big.NewInt(58)
	mod := new(big.Int)
	var out []byte
	for n.Sign() > 0 {
		n.DivMod(n, base, mod)
		out = append(out, base58Alphabet[mod.Int64()])
	}
	for i := 0; i < zeros; i++ {
		out = append(out, base58Alphabet[0])
	}
	// reverse
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return string(out)
}

func base58Decode(s string) []byte {
	if s == "" {
		return nil
	}
	zeros := 0
	for _, c := range s {
		if string(c) == string(base58Alphabet[0]) {
			zeros++
		} else {
			break
		}
	}
	n := new(big.Int)
	base := big.NewInt(58)
	tmp := new(big.Int)
	for _, c := range s {
		idx := strings.IndexRune(base58Alphabet, c)
		if idx < 0 {
			return nil
		}
		n.Mul(n, base)
		n.Add(n, tmp.SetInt64(int64(idx)))
	}
	raw := n.Bytes()
	out := make([]byte, zeros+len(raw))
	copy(out[zeros:], raw)
	return out
}

// ── Graph maintenance ────────────────────────────────────────────────

// WeightedEdge is a labelled edge in the brain graph.
type WeightedEdge struct {
	Source string
	Target string
	Weight float64
}

// NormalizeEdges min-max scales weights into [0,1].
func NormalizeEdges(edges []WeightedEdge) []WeightedEdge {
	if len(edges) == 0 {
		return nil
	}
	lo, hi := math.Inf(1), math.Inf(-1)
	for _, e := range edges {
		if e.Weight < lo {
			lo = e.Weight
		}
		if e.Weight > hi {
			hi = e.Weight
		}
	}
	span := hi - lo
	out := make([]WeightedEdge, len(edges))
	for i, e := range edges {
		w := 1.0
		if span > 0 {
			w = (e.Weight - lo) / span
		}
		out[i] = WeightedEdge{e.Source, e.Target, w}
	}
	return out
}

// SnnScore rewrites each edge with shared-nearest-neighbor agreement.
func SnnScore(edges []WeightedEdge, k int) []WeightedEdge {
	adj := map[string][]WeightedEdge{}
	for _, e := range edges {
		adj[e.Source] = append(adj[e.Source], e)
		adj[e.Target] = append(adj[e.Target], WeightedEdge{e.Target, e.Source, e.Weight})
	}
	nbrs := map[string]map[string]bool{}
	for node, lst := range adj {
		sort.Slice(lst, func(i, j int) bool { return lst[i].Weight > lst[j].Weight })
		set := map[string]bool{}
		for i, x := range lst {
			if i >= k {
				break
			}
			set[x.Target] = true
		}
		nbrs[node] = set
	}
	out := make([]WeightedEdge, len(edges))
	for i, e := range edges {
		a, b := nbrs[e.Source], nbrs[e.Target]
		inter, union := 0, 0
		for n := range a {
			if b[n] {
				inter++
			}
		}
		union = len(a) + len(b) - inter
		w := 0.0
		if union > 0 {
			w = float64(inter) / float64(union)
		}
		out[i] = WeightedEdge{e.Source, e.Target, w}
	}
	return out
}

// PfnetInfinity keeps edges that aren't dominated by a 2-hop path under the r=∞ Pathfinder model.
func PfnetInfinity(edges []WeightedEdge) []WeightedEdge {
	adj := map[string]map[string]float64{}
	for _, e := range edges {
		if adj[e.Source] == nil {
			adj[e.Source] = map[string]float64{}
		}
		if cur := adj[e.Source][e.Target]; e.Weight > cur {
			adj[e.Source][e.Target] = e.Weight
		}
	}
	var keep []WeightedEdge
	for _, e := range edges {
		dominated := false
		for x, wUx := range adj[e.Source] {
			if x == e.Target {
				continue
			}
			wXv, ok := adj[x][e.Target]
			if !ok {
				continue
			}
			min := wUx
			if wXv < min {
				min = wXv
			}
			if min > e.Weight {
				dominated = true
				break
			}
		}
		if !dominated {
			keep = append(keep, e)
		}
	}
	return keep
}

// Louvain returns a node→community mapping via greedy modularity optimization.
func Louvain(edges []WeightedEdge, passes int) map[string]int {
	if passes <= 0 {
		passes = 10
	}
	nodes := map[string]bool{}
	for _, e := range edges {
		nodes[e.Source] = true
		nodes[e.Target] = true
	}
	community := map[string]int{}
	id := 0
	for n := range nodes {
		community[n] = id
		id++
	}
	adj := map[string][]struct {
		To string
		W  float64
	}{}
	var total float64
	for _, e := range edges {
		adj[e.Source] = append(adj[e.Source], struct {
			To string
			W  float64
		}{e.Target, e.Weight})
		adj[e.Target] = append(adj[e.Target], struct {
			To string
			W  float64
		}{e.Source, e.Weight})
		total += e.Weight
	}
	deg := map[string]float64{}
	for n, lst := range adj {
		for _, ne := range lst {
			deg[n] += ne.W
		}
	}
	m := total
	for p := 0; p < passes; p++ {
		improved := false
		for n := range nodes {
			cur := community[n]
			wTo := map[int]float64{}
			for _, ne := range adj[n] {
				wTo[community[ne.To]] += ne.W
			}
			best, bestGain := cur, 0.0
			kn := deg[n]
			for c, wnc := range wTo {
				if c == cur {
					continue
				}
				var sigmaTot float64
				for other, comm := range community {
					if comm == c && other != n {
						sigmaTot += deg[other]
					}
				}
				gain := wnc - (kn*sigmaTot)/math.Max(2*m, 1e-9)
				if gain > bestGain {
					bestGain = gain
					best = c
				}
			}
			if best != cur {
				community[n] = best
				improved = true
			}
		}
		if !improved {
			break
		}
	}
	// compact ids
	idMap := map[int]int{}
	nxt := 0
	for _, c := range community {
		if _, ok := idMap[c]; !ok {
			idMap[c] = nxt
			nxt++
		}
	}
	for n, c := range community {
		community[n] = idMap[c]
	}
	return community
}

// ── Inference: slug + runtime config + link types ────────────────────

var knownProviders = map[string]bool{
	"ollama": true, "openai": true, "openrouter": true, "llamacpp": true,
	"anthropic": true, "google": true, "azure": true, "groq": true, "together": true, "mock": true,
}

// ParsedSlug is the provider + model split.
type ParsedSlug struct {
	Provider string
	Model    string
}

// ParseSlug splits a `provider:model:tag` slug, falling back to defaultProvider for bare slugs.
func ParseSlug(slug, defaultProvider string) ParsedSlug {
	idx := strings.Index(slug, ":")
	if idx < 0 {
		return ParsedSlug{Provider: defaultProvider, Model: slug}
	}
	head := slug[:idx]
	if knownProviders[head] {
		return ParsedSlug{Provider: head, Model: slug[idx+1:]}
	}
	return ParsedSlug{Provider: defaultProvider, Model: slug}
}

// FormatSlug round-trips a ParsedSlug.
func FormatSlug(p ParsedSlug) string { return p.Provider + ":" + p.Model }

// RuntimeConfig layers db_override → env → default.
type RuntimeConfig struct {
	Defaults  map[string]string
	Env       map[string]string
	overrides map[string]string
}

// NewRuntimeConfig builds a fresh layered config.
func NewRuntimeConfig(defaults, env map[string]string) *RuntimeConfig {
	if defaults == nil {
		defaults = map[string]string{}
	}
	if env == nil {
		env = map[string]string{}
	}
	return &RuntimeConfig{Defaults: defaults, Env: env, overrides: map[string]string{}}
}

// Get reads the highest-precedence value.
func (r *RuntimeConfig) Get(key string) (string, bool) {
	if v, ok := r.overrides[key]; ok {
		return v, true
	}
	if v, ok := r.Env[key]; ok {
		return v, true
	}
	v, ok := r.Defaults[key]
	return v, ok
}

// Source reports which layer holds the value.
func (r *RuntimeConfig) Source(key string) string {
	if _, ok := r.overrides[key]; ok {
		return "db_override"
	}
	if _, ok := r.Env[key]; ok {
		return "env"
	}
	if _, ok := r.Defaults[key]; ok {
		return "default"
	}
	return "absent"
}

// Set writes the runtime override.
func (r *RuntimeConfig) Set(key, value string) { r.overrides[key] = value }

// Clear removes a runtime override.
func (r *RuntimeConfig) Clear(key string) { delete(r.overrides, key) }

// LinkTypes is the canonical 11-element edge taxonomy.
var LinkTypes = []string{
	"mentions", "founded", "invested_in", "advises", "works_at",
	"attended", "authored", "cites", "succeeded_by", "located_in", "related",
}

var linkRules = []struct {
	re *regexp.Regexp
	t  string
}{
	{regexp.MustCompile(`(?i)\bfounded\b`), "founded"},
	{regexp.MustCompile(`(?i)\binvested\s+in\b`), "invested_in"},
	{regexp.MustCompile(`(?i)\badvis(?:or|es|ing)\b`), "advises"},
	{regexp.MustCompile(`(?i)\bworks?\s+(?:at|for)\b`), "works_at"},
	{regexp.MustCompile(`(?i)\battended\b`), "attended"},
	{regexp.MustCompile(`(?i)\b(?:wrote|authored)\b`), "authored"},
	{regexp.MustCompile(`(?i)\bcites?\b`), "cites"},
	{regexp.MustCompile(`(?i)\bsucceeded\s+by\b`), "succeeded_by"},
	{regexp.MustCompile(`(?i)\blocated\s+in\b`), "located_in"},
}

// ClassifyLinkRule applies the zero-LLM rule table.
func ClassifyLinkRule(evidence string) string {
	for _, r := range linkRules {
		if r.re.MatchString(evidence) {
			return r.t
		}
	}
	return "mentions"
}
