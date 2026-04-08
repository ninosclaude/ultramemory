package graph

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"github.com/sharpner/ultramemory/secureindex"
	"github.com/sharpner/ultramemory/store"
)

// GenerateCommunityReports stores fact-based community summaries for communities
// with ≥3 Person members (Leiden §4 community context).
//
// Reports are built directly from edge facts — no LLM generation.
// LLM-generated prose summaries caused hallucinations in testing (e.g., adding
// "LGBTQ+ support group" membership not present in the actual conversation),
// which degraded open-domain retrieval by -2.8% and overall by -1.7%.
// Fact-only reports are grounded, verifiable, and prevent context pollution.
func GenerateCommunityReports(ctx context.Context, db *store.DB, groupID string) error {
	inputs, err := db.CommunityInputsForGroup(ctx, groupID, 3)
	if err != nil {
		return fmt.Errorf("load community inputs: %w", err)
	}
	if err := db.ClearCommunityReports(ctx, groupID); err != nil {
		return fmt.Errorf("clear community reports: %w", err)
	}
	if len(inputs) == 0 {
		return nil
	}

	generated := 0
	for _, inp := range inputs {
		if len(inp.KeyFacts) == 0 {
			continue
		}
		// Build a fact-only report: entity roster + key facts.
		// No LLM call — prevents hallucination of training-data knowledge.
		var sb strings.Builder
		sb.WriteString("People: ")
		sb.WriteString(strings.Join(inp.EntityNames, ", "))
		sb.WriteString(". Key facts: ")
		sb.WriteString(strings.Join(inp.KeyFacts, " "))

		report := sb.String()
		if err := db.StoreCommunityReport(ctx, groupID, inp.CommunityID, report); err != nil {
			slog.Warn("store community report failed", "community", inp.CommunityID, "err", err)
			continue
		}
		generated++
	}
	slog.Info("community reports generated (fact-only)", "group", groupID, "count", generated, "total", len(inputs))
	return nil
}

// GenerateSecureCommunityReports stores encrypted fact-only community summaries.
func GenerateSecureCommunityReports(ctx context.Context, db *store.DB, groupID, personalKey string) error {
	inputs, err := db.CommunityInputsForGroupSecure(ctx, groupID, 3, personalKey)
	if err != nil {
		return fmt.Errorf("load secure community inputs: %w", err)
	}
	if err := db.ClearCommunityReports(ctx, groupID); err != nil {
		return fmt.Errorf("clear secure community reports: %w", err)
	}
	if len(inputs) == 0 {
		return nil
	}

	generated := 0
	for _, inp := range inputs {
		if len(inp.KeyFacts) == 0 {
			continue
		}
		var sb strings.Builder
		sb.WriteString("People: ")
		sb.WriteString(strings.Join(inp.EntityNames, ", "))
		sb.WriteString(". Key facts: ")
		sb.WriteString(strings.Join(inp.KeyFacts, " "))

		reportCiphertext, err := secureindex.EncryptString(personalKey, "community-report", sb.String())
		if err != nil {
			slog.Warn("encrypt community report failed", "community", inp.CommunityID, "err", err)
			continue
		}
		if err := db.StoreCommunityReport(ctx, groupID, inp.CommunityID, reportCiphertext); err != nil {
			slog.Warn("store secure community report failed", "community", inp.CommunityID, "err", err)
			continue
		}
		generated++
	}
	slog.Info("secure community reports generated (fact-only)", "group", groupID, "count", generated, "total", len(inputs))
	return nil
}
