package recipe

import (
	"testing"
)

func TestList_IncludesEmbeddedEmail(t *testing.T) {
	names := List()
	got := false
	for _, n := range names {
		if n == "email" {
			got = true
		}
	}
	if !got {
		t.Fatalf("expected email recipe in List(); got %v", names)
	}
}

func TestLoad_Email(t *testing.T) {
	r, err := Load("email")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if r.Name() != "email" {
		t.Fatalf("recipe name: got %q", r.Name())
	}
	if r.Version() != 1 {
		t.Fatalf("recipe version: got %d", r.Version())
	}
	if r["backend"] != "gmail" {
		t.Fatalf("backend: got %v", r["backend"])
	}
	if cron, _ := r["cron"].(string); cron == "" {
		t.Fatalf("cron missing or non-string: %v", r["cron"])
	}
}

func TestLoad_Unknown(t *testing.T) {
	_, err := Load("does-not-exist")
	if err == nil {
		t.Fatal("expected error for unknown recipe")
	}
}
