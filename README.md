# bot-go

> Go runtime of the Hanzo Bot. Single static binary. Embeds in `hanzoai/node` or runs standalone.

Same contract as the TS [`hanzoai/bot`](https://github.com/hanzoai/bot)
(OpenClaw fork), implementing the spec at
[`hanzobot/core`](https://github.com/hanzobot/core). A brain.db
written by the TS bot is read by this binary and vice versa.

Pure-CPU algorithm port (fusion, rerank, FTS, MRL, eval, spatial,
captions, graph maintenance, slug + runtime config) lives at
`pkg/brain/algorithms.go` and mirrors the TS / Python / Rust runtimes
hosted at [`hanzoai/brain`](https://github.com/hanzoai/brain). Byte-
equivalent outputs where the algorithm is deterministic; same test
corpus passes in all four ports.

## Install

```bash
go install github.com/hanzobot/go/cmd/hanzo-bot@latest
```

Pure Go, no cgo (modernc.org/sqlite). Builds anywhere — `GOOS=linux`,
`GOOS=darwin`, `GOOS=windows`, `GOOS=js`/wasm, all from one tree.

## Try it

```bash
hanzo-bot brain init                           # opens ~/.hanzo/brain/brain.db
hanzo-bot brain ingest README.md               # ingest + auto-extract edges
hanzo-bot brain search "founded"               # hybrid FTS search
hanzo-bot brain recall people/alice            # facts for an entity
hanzo-bot recipes list                         # installed recipes
hanzo-bot recipes show email                   # print one as JSON
```

## What ships

```
pkg/
  brain/      pluggable BrainStore — SQLite default, registry for more
              graph_links — zero-LLM typed-link extractor (6 edge types)
              store       — pages / edges / facts / hybrid search
  recipe/     YAML ingest recipes (email embedded; user dirs picked up
              from $HANZO_BRAIN_RECIPES and ~/.hanzo/recipes)
cmd/
  hanzo-bot/  CLI entry point
```

## Embedding

```go
import (
    "context"
    "github.com/hanzobot/go/pkg/brain"
)

func newBrain(ctx context.Context) (brain.BrainStore, error) {
    return brain.Open(ctx, brain.Config{})  // SQLite at ~/.hanzo/brain/brain.db
}
```

Plug a different backend by calling `brain.RegisterBackend("name", factory)`
from your own package before `brain.Open`. Same plug-shape as the TS port.

## Cross-runtime guarantees

- `~/.hanzo/brain/brain.db` schema is identical to the TS, Python, and Rust ports.
- Graph-links regex set + slugify algorithm are byte-equivalent (same test corpus passes in all four).
- Recipe YAML is byte-identical to `hanzoai/bot/extensions/recipes-brain/recipes/`.

## Status

Brain, graph-links, recipes — done. Channel adapters (Slack, Telegram,
Discord, WhatsApp, iMessage, …) and the gateway HTTP/WS server land in
follow-up commits. Wire format will be ZAP.

## Tests

```bash
go test ./...
```

58 tests in `pkg/brain` covering the full algorithm port (fusion, rerank,
FTS, embed registry, MRL, temporal, captions, tokenizer, eval, spatial,
HTTP range, wallet address, graph maintenance, slug parsing, runtime
config, link classification) plus graph-links (slugify, all six edge
types, code-fence stripping, dedup, bare slug refs, reconcile
add/remove) and the end-to-end SQLite store (upsert/get pages, edges
roundtrip, fact recall, hybrid search). Plus recipe load in
`pkg/recipe`.

## License

MIT.
