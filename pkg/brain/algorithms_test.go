package brain

import (
	"math"
	"strings"
	"testing"
	"time"
)

func hit(slug string, score float64) SearchHit {
	return SearchHit{Slug: slug, Score: score, Excerpt: slug, Source: "keyword"}
}

func nearly(a, b, eps float64) bool { return math.Abs(a-b) <= eps }

// ── Fusion ────────────────────────────────────────────────────────────

func TestRrfNormalizesTop(t *testing.T) {
	r := RrfFuse([][]SearchHit{{hit("a", 1), hit("b", 0.5)}}, 10, 0)
	if r[0].Slug != "a" {
		t.Fatalf("want a got %v", r[0].Slug)
	}
	if !nearly(r[0].Score, 1.0, 0.01) {
		t.Fatalf("want ~1 got %v", r[0].Score)
	}
}

func TestRsfPreservesMagnitude(t *testing.T) {
	r := RsfFuse([][]SearchHit{{hit("a", 100), hit("b", 50)}, {hit("a", 1), hit("c", 0.5)}}, 10, nil)
	if r[0].Slug != "a" {
		t.Fatalf("want a got %v", r[0].Slug)
	}
	if len(r) != 3 {
		t.Fatalf("want 3 got %d", len(r))
	}
}

func TestSelectRrfK(t *testing.T) {
	if got := SelectRrfK(Characterize(`"hello world"`)); got != 10 {
		t.Fatalf("phrase: %d", got)
	}
	if got := SelectRrfK(Characterize("foo AND bar")); got != 15 {
		t.Fatalf("bool: %d", got)
	}
	if got := SelectRrfK(Characterize("rust")); got != 15 {
		t.Fatalf("short: %d", got)
	}
	if got := SelectRrfK(Characterize("a b c d e f g h i j")); got != 40 {
		t.Fatalf("long: %d", got)
	}
}

func TestSelectWeights(t *testing.T) {
	short := SelectWeights(Characterize("rust"))
	if short.Fts <= short.Semantic {
		t.Fatalf("short should lean FTS")
	}
	long := SelectWeights(Characterize("how do retrieval augmented generation systems typically work in production scale"))
	if long.Semantic <= long.Fts {
		t.Fatalf("long should lean semantic")
	}
}

// ── Rerank ────────────────────────────────────────────────────────────

func TestCosine(t *testing.T) {
	if !nearly(Cosine([]float64{1, 0}, []float64{1, 0}), 1, 1e-6) {
		t.Fatal("identical")
	}
	if !nearly(Cosine([]float64{1, 0}, []float64{0, 1}), 0, 1e-6) {
		t.Fatal("orthogonal")
	}
}

func TestMmrDiverseSecond(t *testing.T) {
	hits := []MmrInput{
		{SearchHit: hit("a", 0.9), Embedding: []float64{1, 0}},
		{SearchHit: hit("b", 0.85), Embedding: []float64{1, 0.01}},
		{SearchHit: hit("c", 0.6), Embedding: []float64{0, 1}},
	}
	out := MmrRerank(hits, 0.2, 2)
	if out[0].Slug != "a" || out[1].Slug != "c" {
		t.Fatalf("got %v %v", out[0].Slug, out[1].Slug)
	}
}

// ── Dedup ─────────────────────────────────────────────────────────────

func TestDedupHits(t *testing.T) {
	out := DedupHits([]SearchHit{
		hit("page/foo#chunk-0", 0.5),
		hit("page/foo#chunk-1", 0.8),
		hit("page/bar", 0.6),
	}, 1)
	if len(out) != 2 {
		t.Fatalf("want 2 got %d", len(out))
	}
}

// ── Script + FTS ──────────────────────────────────────────────────────

func TestDetectScript(t *testing.T) {
	if DetectScript("こんにちは世界").Primary != "cjk" {
		t.Fatal("cjk")
	}
	if DetectScript("Hello world").Primary != "latin" {
		t.Fatal("latin")
	}
	if DetectScript("Привет").Primary != "cyrillic" {
		t.Fatal("cyrillic")
	}
}

func TestCjkBigrams(t *testing.T) {
	out := CjkBigrams("hello 世界 こんにちは")
	if !contains(out, "hello") || !contains(out, "世界") || !contains(out, "こん") {
		t.Fatalf("got %v", out)
	}
}

func TestEmojiTrigrams(t *testing.T) {
	out := EmojiTrigrams("hi 🚀🌌🌟")
	if len(out) == 0 {
		t.Fatal("expected trigrams")
	}
}

func TestParseWebSearch(t *testing.T) {
	p := ParseWebSearch(`"hello world" foo OR bar -baz qux`)
	if len(p.Phrases) != 1 || p.Phrases[0] != "hello world" {
		t.Fatalf("phrases %v", p.Phrases)
	}
	if len(p.Optional) != 1 || p.Optional[0][0] != "foo" || p.Optional[0][1] != "bar" {
		t.Fatalf("optional %v", p.Optional)
	}
	if len(p.Excluded) != 1 || p.Excluded[0] != "baz" {
		t.Fatalf("excluded %v", p.Excluded)
	}
}

func TestToFts5Match(t *testing.T) {
	sql := ToFts5Match(ParseWebSearch("apple OR orange -spoil"))
	if !strings.Contains(sql, "apple OR orange") || !strings.Contains(sql, "NOT spoil") {
		t.Fatalf("sql=%q", sql)
	}
}

// ── Embed registry + MRL ──────────────────────────────────────────────

func TestEmbedRegistry(t *testing.T) {
	if m, ok := GetEmbeddingModel("ollama:nomic-embed-text"); !ok || m.Dim != 768 {
		t.Fatal("nomic")
	}
	if m, ok := GetEmbeddingModel("openai:text-embedding-3-small"); !ok || m.Dim != 1536 {
		t.Fatal("openai small")
	}
}

func TestPrefixFor(t *testing.T) {
	e5, _ := GetEmbeddingModel("intfloat/e5-large-v2")
	if PrefixFor(e5, "query", "x") != "query: x" {
		t.Fatal("e5 query")
	}
	if PrefixFor(e5, "passage", "x") != "passage: x" {
		t.Fatal("e5 passage")
	}
	nomic, _ := GetEmbeddingModel("ollama:nomic-embed-text")
	if PrefixFor(nomic, "query", "x") != "x" {
		t.Fatal("nomic symmetric")
	}
}

func TestMrlTruncate(t *testing.T) {
	v := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	tnc := MrlTruncate(v, 4)
	if len(tnc) != 4 {
		t.Fatalf("len %d", len(tnc))
	}
	var s float64
	for _, x := range tnc {
		s += x * x
	}
	if !nearly(math.Sqrt(s), 1, 1e-6) {
		t.Fatalf("not unit %v", math.Sqrt(s))
	}
}

func TestCoarseDim(t *testing.T) {
	m, _ := GetEmbeddingModel("openai:text-embedding-3-large")
	cd := CoarseDim(m)
	if cd < 256 || cd > 512 {
		t.Fatalf("coarse=%d", cd)
	}
}

func TestL2NormalizeZero(t *testing.T) {
	v := []float64{0, 0, 0}
	if L2Normalize(v)[0] != 0 {
		t.Fatal("zero stays zero")
	}
}

// ── Temporal ──────────────────────────────────────────────────────────

func TestV7Bounds(t *testing.T) {
	now := time.Now().UnixMilli()
	if V7Floor(now) >= V7Ceiling(now) {
		t.Fatal("floor < ceiling expected")
	}
}

func TestRangeBounds(t *testing.T) {
	f, c := RangeBounds(TemporalRange{From: "2026-01-01T00:00:00Z", To: "2026-12-31T23:59:59Z"})
	if f >= c {
		t.Fatal("floor < ceiling expected")
	}
}

// ── Captions ──────────────────────────────────────────────────────────

func TestCaptionRenderers(t *testing.T) {
	segs := []CaptionSegment{
		{StartSecs: 0, EndSecs: 1.5, Text: "hi", Speaker: "S0"},
		{StartSecs: 1.5, EndSecs: 3, Text: "world", Speaker: "S1"},
	}
	if !strings.HasPrefix(RenderVtt(segs), "WEBVTT") {
		t.Fatal("vtt header")
	}
	if !strings.Contains(RenderSrt(segs), "00:00:00,000 --> 00:00:01,500") {
		t.Fatal("srt arrow")
	}
	if !strings.HasPrefix(RenderRttm(segs, ""), "SPEAKER") {
		t.Fatal("rttm header")
	}
}

// ── Tokenizer ─────────────────────────────────────────────────────────

func TestEstimateTokens(t *testing.T) {
	if EstimateTokens("hi there friend") <= EstimateTokens("hi") {
		t.Fatal("longer => more tokens")
	}
	if EstimateTokens("こんにちは") != 5 {
		t.Fatalf("cjk per char")
	}
}

func TestTruncateToTokens(t *testing.T) {
	long := strings.Repeat("alpha ", 100)
	tr := TruncateToTokens(long, 20)
	if EstimateTokens(tr) > 20 {
		t.Fatalf("over budget")
	}
}

// ── Eval ──────────────────────────────────────────────────────────────

func qe() QueryEval {
	return QueryEval{
		Predicted: []string{"a", "b", "c", "d"},
		Relevant:  map[string]int{"c": 1, "d": 1},
	}
}

func TestReciprocalRank(t *testing.T) {
	if !nearly(ReciprocalRank(qe()), 1.0/3.0, 1e-6) {
		t.Fatal("rr 1/3")
	}
}

func TestRecallAtK(t *testing.T) {
	q := qe()
	if RecallAtK(q, 2) != 0 {
		t.Fatal("recall@2 = 0")
	}
	if RecallAtK(q, 4) != 1 {
		t.Fatal("recall@4 = 1")
	}
}

func TestPrecisionAtK(t *testing.T) {
	if !nearly(PrecisionAtK(qe(), 4), 0.5, 1e-6) {
		t.Fatal("precision@4 = 0.5")
	}
}

func TestNdcg(t *testing.T) {
	graded := QueryEval{Predicted: []string{"a", "b"}, Relevant: map[string]int{"a": 3, "b": 1}}
	if NdcgAtK(graded, 2) <= 0.9 {
		t.Fatalf("ndcg %v", NdcgAtK(graded, 2))
	}
}

func TestMrr(t *testing.T) {
	if MeanReciprocalRank([]QueryEval{qe()}) <= 0 {
		t.Fatal("mrr > 0")
	}
}

// ── Spatial ───────────────────────────────────────────────────────────

func TestHaversineZero(t *testing.T) {
	if !nearly(HaversineKm(0, 0, 0, 0), 0, 1e-6) {
		t.Fatal("zero")
	}
}

func TestHaversineNycLa(t *testing.T) {
	d := HaversineKm(40.7128, -74.006, 34.0522, -118.2437)
	if math.Abs(d-3935) > 50 {
		t.Fatalf("nyc-la %v", d)
	}
}

func TestBBoxAround(t *testing.T) {
	box := BBoxAround(37.77, -122.42, 10)
	if !InBox(37.77, -122.42, box) {
		t.Fatal("center inside")
	}
	if InBox(0, 0, box) {
		t.Fatal("origin outside")
	}
}

// ── HTTP Range ────────────────────────────────────────────────────────

func TestRangeClosed(t *testing.T) {
	r, ok, _ := ParseRange("bytes=0-99", 1000)
	if !ok || r.Start != 0 || r.End != 99 {
		t.Fatalf("got %+v", r)
	}
}

func TestRangeSuffix(t *testing.T) {
	r, ok, _ := ParseRange("bytes=-100", 1000)
	if !ok || r.Start != 900 || r.End != 999 {
		t.Fatalf("got %+v", r)
	}
}

func TestRangeUnsatisfiable(t *testing.T) {
	_, _, u := ParseRange("bytes=2000-3000", 1000)
	if !u {
		t.Fatal("expected unsatisfiable")
	}
}

func TestContentRange(t *testing.T) {
	if ContentRange(RangeRequest{0, 99}, 1000) != "bytes 0-99/1000" {
		t.Fatal("fmt")
	}
}

// ── Wallet address ────────────────────────────────────────────────────

func TestAddressRoundTrip(t *testing.T) {
	pk := make([]byte, 32)
	for i := range pk {
		pk[i] = byte(i)
	}
	addr, err := EncodeAddress(pk, "hanzo")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(addr, "hanzo:") {
		t.Fatal("prefix")
	}
	dec, err := DecodeAddress(addr)
	if err != nil {
		t.Fatal(err)
	}
	if dec.Prefix != "hanzo" || dec.Version != 1 {
		t.Fatalf("decoded %+v", dec)
	}
}

func TestAddressBadChecksum(t *testing.T) {
	if _, err := DecodeAddress("hanzo:11111111111111111111111111"); err == nil {
		t.Fatal("expected error")
	}
}

// ── Graph maintenance ─────────────────────────────────────────────────

func TestNormalizeEdges(t *testing.T) {
	out := NormalizeEdges([]WeightedEdge{{"a", "b", 10}, {"b", "c", 5}})
	if !nearly(out[0].Weight, 1, 1e-6) || !nearly(out[1].Weight, 0, 1e-6) {
		t.Fatalf("got %+v", out)
	}
}

func TestSnnBounds(t *testing.T) {
	edges := []WeightedEdge{{"a", "b", 0.9}, {"a", "c", 0.8}, {"b", "c", 0.7}}
	for _, e := range SnnScore(edges, 2) {
		if e.Weight < 0 || e.Weight > 1 {
			t.Fatalf("out of bounds %v", e.Weight)
		}
	}
}

func TestPfnetInfinity(t *testing.T) {
	out := PfnetInfinity([]WeightedEdge{{"a", "b", 0.9}, {"b", "c", 0.9}, {"a", "c", 0.5}})
	for _, e := range out {
		if e.Source == "a" && e.Target == "c" {
			t.Fatal("should be dominated")
		}
	}
}

func TestLouvain(t *testing.T) {
	edges := []WeightedEdge{{"a", "b", 1}, {"b", "c", 1}, {"a", "c", 1}}
	c := Louvain(edges, 0)
	if len(c) != 3 {
		t.Fatalf("got %d nodes", len(c))
	}
}

// ── Slug + runtime config + link types ────────────────────────────────

func TestParseSlug(t *testing.T) {
	if p := ParseSlug("openai:gpt-4o", "ollama"); p.Provider != "openai" || p.Model != "gpt-4o" {
		t.Fatalf("explicit %+v", p)
	}
	if p := ParseSlug("qwen3:8b", "ollama"); p.Provider != "ollama" || p.Model != "qwen3:8b" {
		t.Fatalf("implicit %+v", p)
	}
	if FormatSlug(ParsedSlug{Provider: "openai", Model: "gpt-4o"}) != "openai:gpt-4o" {
		t.Fatal("round trip")
	}
}

func TestRuntimeConfigPrecedence(t *testing.T) {
	rc := NewRuntimeConfig(map[string]string{"K": "default"}, map[string]string{"K": "env"})
	if v, _ := rc.Get("K"); v != "env" {
		t.Fatal("env wins over default")
	}
	rc.Set("K", "override")
	if v, _ := rc.Get("K"); v != "override" {
		t.Fatal("override wins")
	}
	if rc.Source("K") != "db_override" {
		t.Fatal("source label")
	}
	rc.Clear("K")
	if v, _ := rc.Get("K"); v != "env" {
		t.Fatal("revert to env after clear")
	}
}

func TestClassifyLinkRule(t *testing.T) {
	if ClassifyLinkRule("Alice founded Acme") != "founded" {
		t.Fatal("founded")
	}
	if ClassifyLinkRule("Alice invested in Acme") != "invested_in" {
		t.Fatal("invested_in")
	}
	if ClassifyLinkRule("worked together") != "mentions" {
		t.Fatal("default mentions")
	}
}

// helpers

func contains(xs []string, x string) bool {
	for _, v := range xs {
		if v == x {
			return true
		}
	}
	return false
}
