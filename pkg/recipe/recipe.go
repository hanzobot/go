// Package recipe loads YAML ingest recipes for the Hanzo Brain.
// Same shape as @hanzo/bot-recipes-brain (TS), hanzo_memory.recipes
// (Python), and hanzo_mcp::brain::recipes (Rust). A recipe written by
// one runtime works in any other runtime.
package recipe

import (
	"embed"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

//go:embed recipes/*.yaml
var builtin embed.FS

// Recipe is the parsed YAML document. Open shape — runtimes accept any
// keys; the canonical ones are recipe, version, backend, auth, cron,
// ingest, classify, draft, enqueue, notify, on_swipe_*.
type Recipe map[string]any

// Name returns the recipe's "recipe" field (its slug).
func (r Recipe) Name() string {
	if v, ok := r["recipe"].(string); ok {
		return v
	}
	return ""
}

// Version returns the recipe's declared version (defaults to 1).
func (r Recipe) Version() int {
	switch v := r["version"].(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	}
	return 1
}

// userDirs returns the user-configurable recipe roots, in priority
// order. The first match for a name wins.
func userDirs() []string {
	dirs := []string{}
	if env := os.Getenv("HANZO_BRAIN_RECIPES"); env != "" {
		dirs = append(dirs, env)
	}
	if home, err := os.UserHomeDir(); err == nil {
		dirs = append(dirs, filepath.Join(home, ".hanzo", "recipes"))
	}
	return dirs
}

// List returns the names (without `.yaml`) of every recipe available
// in the user dirs and the embedded built-in set, deduplicated.
func List() []string {
	seen := map[string]bool{}
	for _, d := range userDirs() {
		entries, err := os.ReadDir(d)
		if err != nil {
			continue
		}
		for _, e := range entries {
			if !e.IsDir() && strings.HasSuffix(e.Name(), ".yaml") {
				seen[strings.TrimSuffix(e.Name(), ".yaml")] = true
			}
		}
	}
	// Also list embedded built-ins.
	if entries, err := builtin.ReadDir("recipes"); err == nil {
		for _, e := range entries {
			if strings.HasSuffix(e.Name(), ".yaml") {
				seen[strings.TrimSuffix(e.Name(), ".yaml")] = true
			}
		}
	}
	names := make([]string, 0, len(seen))
	for n := range seen {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// Load reads, parses, and returns the named recipe. User dirs win
// over the embedded built-in copy.
func Load(name string) (Recipe, error) {
	// User dirs first.
	for _, d := range userDirs() {
		path := filepath.Join(d, name+".yaml")
		if raw, err := os.ReadFile(path); err == nil {
			return parse(raw, path)
		}
	}
	// Embedded built-in fallback.
	if raw, err := builtin.ReadFile("recipes/" + name + ".yaml"); err == nil {
		return parse(raw, "builtin/"+name)
	}
	return nil, fmt.Errorf("recipe %q not found. set HANZO_BRAIN_RECIPES or drop into ~/.hanzo/recipes. available: %v", name, List())
}

func parse(raw []byte, where string) (Recipe, error) {
	var r Recipe
	if err := yaml.Unmarshal(raw, &r); err != nil {
		return nil, fmt.Errorf("parse %s: %w", where, err)
	}
	if r.Name() == "" {
		return nil, errors.New("recipe missing required `recipe:` field")
	}
	return r, nil
}
