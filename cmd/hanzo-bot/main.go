// hanzo-bot is the Go runtime of the Hanzo Bot.
//
// Minimal CLI surface — the full bot/router/channel adapters land in
// follow-up commits. This entry point ships:
//
//   hanzo-bot brain init           — open the brain at ~/.hanzo/brain/brain.db
//   hanzo-bot brain ingest <file>  — add a markdown page + extract typed edges
//   hanzo-bot brain recall <slug>  — list facts for an entity
//   hanzo-bot brain search <query> — hybrid FTS search across pages
//   hanzo-bot recipes list         — list installed recipes
//   hanzo-bot recipes show <name>  — print one recipe as JSON
//
// Standalone OR embeddable. The exported pkg/brain + pkg/recipe + pkg/channel
// types are importable from hanzoai/node, hanzoai/cloud, etc.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/hanzobot/go/pkg/brain"
	"github.com/hanzobot/go/pkg/recipe"
)

func main() {
	if len(os.Args) < 2 {
		usage(0)
	}
	switch os.Args[1] {
	case "brain":
		brainCmd(os.Args[2:])
	case "recipes":
		recipesCmd(os.Args[2:])
	case "-h", "--help", "help":
		usage(0)
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n\n", os.Args[1])
		usage(2)
	}
}

func usage(code int) {
	fmt.Println(`hanzo-bot — Hanzo Bot, Go runtime

Usage:
  hanzo-bot brain init
  hanzo-bot brain ingest <file.md>
  hanzo-bot brain recall <slug>
  hanzo-bot brain search <query>
  hanzo-bot recipes list
  hanzo-bot recipes show <name>

Canonical brain: ~/.hanzo/brain/brain.db (shared with hanzo-mcp, hanzo-dev,
the python SDK, and the TS bot — same schema, same paths).`)
	os.Exit(code)
}

func brainCmd(args []string) {
	ctx := context.Background()
	store, err := brain.Open(ctx, brain.Config{})
	must(err)
	defer store.Close()

	if len(args) == 0 {
		usage(2)
	}
	switch args[0] {
	case "init":
		fmt.Printf("brain ready at %s\n", filepath.Join(brain.DefaultDataDir(), "brain.db"))
	case "ingest":
		if len(args) != 2 {
			usage(2)
		}
		raw, err := os.ReadFile(args[1])
		must(err)
		slug := strings.TrimSuffix(filepath.Base(args[1]), ".md")
		must(store.UpsertPage(ctx, slug, string(raw), nil))
		edges := brain.ExtractEdges(slug, string(raw), "")
		must(store.UpsertEdges(ctx, slug, edges))
		fmt.Printf("ingested %s → %d edges\n", slug, len(edges))
	case "recall":
		if len(args) != 2 {
			usage(2)
		}
		facts, err := store.Recall(ctx, args[1], 50, "")
		must(err)
		_ = json.NewEncoder(os.Stdout).Encode(facts)
	case "search":
		if len(args) != 2 {
			usage(2)
		}
		hits, err := store.HybridSearch(ctx, args[1], 5)
		must(err)
		_ = json.NewEncoder(os.Stdout).Encode(hits)
	default:
		usage(2)
	}
}

func recipesCmd(args []string) {
	if len(args) == 0 {
		usage(2)
	}
	switch args[0] {
	case "list":
		for _, n := range recipe.List() {
			fmt.Println(n)
		}
	case "show":
		if len(args) != 2 {
			usage(2)
		}
		r, err := recipe.Load(args[1])
		must(err)
		_ = json.NewEncoder(os.Stdout).Encode(r)
	default:
		usage(2)
	}
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
