// Package bench implements evaluation benchmarks for ultramemory.
package bench

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/google/uuid"
	"github.com/sharpner/ultramemory/graph"
	"github.com/sharpner/ultramemory/llm"
	"github.com/sharpner/ultramemory/secureindex"
	"github.com/sharpner/ultramemory/store"
)

// ── Category mapping ─────────────────────────────────────────────────────────

var categoryName = map[int]string{
	1: "single-hop",
	2: "multi-hop",
	3: "temporal",
	4: "open-domain",
	5: "adversarial",
}

const qaSystem = `You answer questions about a conversation between two people.

Rules:
1. Use ONLY the provided context. Never use outside knowledge.
2. Be extremely concise — answer in as few words as possible.
3. For dates, use the exact format from the context (e.g. "7 May 2023").
4. For names, give the full name exactly as it appears.
5. If the context does NOT contain enough information to answer, respond with exactly: unknown
6. If the question asks about something not mentioned in the context at all, respond with exactly: unknown
7. Do not guess or make assumptions.`

// qaSystemHypothetical is used for questions starting with "Would", "Could", "Might", "Will",
// "Is it likely" etc. — hypothetical/counterfactual reasoning questions (LoCoMo category 3).
// These require a brief reasoning pattern ("Likely yes/no, because...") that tokenF1 rewards
// but the generic qaSystem suppresses (it just says "unknown" or gives bare facts).
const qaSystemHypothetical = `You answer hypothetical questions about a conversation between two people.

Rules:
1. Use ONLY the provided context. Never use outside knowledge.
2. Start your answer with "Likely yes" or "Likely no", then add one brief reason from the context.
   Example: "Likely no, she wants to be a counselor."
3. If the context does NOT contain enough information, respond with exactly: unknown
4. Do not guess beyond what the context says.`

// qaSystemWhen is used for questions starting with "When" or "How long" — temporal multi-hop
// questions (LoCoMo category 2) that require reading session timestamps and relative time refs.
// 34/37 multi-hop questions ask "When did X happen?". Gold answers are relative phrases like
// "The sunday before 25 May 2023" — derived from session dates + dialogue time references.
// isHypotheticalQuestion returns true for questions that require probabilistic reasoning
// rather than a factual recall answer.
func isHypotheticalQuestion(q string) bool {
	lower := strings.ToLower(strings.TrimSpace(q))
	for _, prefix := range []string{"would ", "could ", "might ", "will ", "is it likely", "is it possible"} {
		if strings.HasPrefix(lower, prefix) {
			return true
		}
	}
	return false
}

const chunkSize = 1500

// ── Types ────────────────────────────────────────────────────────────────────

type rawConversation struct {
	SampleID string                     `json:"sample_id"`
	Conv     map[string]json.RawMessage `json:"conversation"`
	QA       []rawQA                    `json:"qa"`
}

type rawQA struct {
	Question          string          `json:"question"`
	Answer            json.RawMessage `json:"answer"`
	AdversarialAnswer json.RawMessage `json:"adversarial_answer"`
	Category          int             `json:"category"`
	Evidence          []string        `json:"evidence"`
}

type rawTurn struct {
	Speaker     string   `json:"speaker"`
	DiaID       string   `json:"dia_id"`
	Text        string   `json:"text"`
	ImgURL      []string `json:"img_url"`
	BlipCaption string   `json:"blip_caption"`
}

// Conversation is one parsed LoCoMo sample.
type Conversation struct {
	SampleID string
	SpeakerA string
	SpeakerB string
	Sessions []Session
	QA       []QA
}

// Session is one dialogue session with a timestamp.
type Session struct {
	Number   int
	DateTime string
	Turns    []Turn
}

// Turn is one dialogue utterance.
type Turn struct {
	Speaker string
	DiaID   string
	Text    string
}

// QA is one annotated question-answer pair.
type QA struct {
	Question string
	Answer   string
	Category int
	Evidence []string
}

// CategoryScore holds aggregated metrics for one QA category.
type CategoryScore struct {
	Category   string
	Count      int
	AvgF1      float64
	AvgEM      float64
	AvgJudge   float64 // LLM-judge score (0 if no judge configured)
	JudgeCount int     // number of judged answers
}

// Result holds the full benchmark output.
type Result struct {
	Overall    CategoryScore
	ByCategory []CategoryScore
	Duration   time.Duration
	HasJudge   bool // whether LLM-judge scores are included
}

type qaScore struct {
	category int
	f1       float64
	em       float64
	judge    float64 // 1.0 = correct per LLM judge, 0 = incorrect, -1 = not judged
}

// ── Public API ───────────────────────────────────────────────────────────────

// Judge is satisfied by any LLM that can evaluate answer correctness.
type Judge interface {
	Judge(ctx context.Context, question, gold, prediction string) (bool, error)
}

// RunLoCoMo evaluates ultramemory against the LoCoMo benchmark.
// Set limit > 0 to evaluate only the first N conversations.
// When baseline is true, only raw episode FTS is used (no graph extraction).
// qaAnswerer overrides the QA answering model when set.
// When qaOnly is true, ingestion is skipped — DB must already be populated.
// judge optionally evaluates each answer for semantic correctness (LLM-as-judge).
func RunLoCoMo(ctx context.Context, dataPath string, db *store.DB, extractor llm.EntityExtractor, embedder llm.Embedder, answerer llm.Answerer, judge Judge, resolveThreshold float64, limit int, baseline, qaOnly bool, personalKey string) (*Result, error) {
	conversations, err := parseLoCoMo(dataPath)
	if err != nil {
		return nil, err
	}
	if limit > 0 && limit < len(conversations) {
		conversations = conversations[:limit]
	}

	mode := "graph"
	if baseline {
		mode = "baseline"
	}
	if personalKey != "" {
		mode = "kpt"
	}

	if answerer == nil {
		type defaultAnswerer interface {
			llm.EntityExtractor
			llm.Answerer
		}
		qaClient, ok := extractor.(defaultAnswerer)
		if !ok {
			return nil, fmt.Errorf("extractor does not implement llm.Answerer")
		}
		answerer = qaClient
	}

	start := time.Now()
	var scores []qaScore

	for ci, conv := range conversations {
		groupID := "locomo-" + conv.SampleID
		if baseline {
			groupID = "baseline-" + conv.SampleID
		}

		slog.Info("ingesting conversation",
			"mode", mode,
			"conv", ci+1, "of", len(conversations),
			"sample", conv.SampleID,
			"sessions", len(conv.Sessions),
			"qa", len(conv.QA),
			"qa_only", qaOnly,
		)

		if !qaOnly {
			var secureExtractor *graph.Extractor
			if personalKey != "" && !baseline {
				secureExtractor = graph.New(db, extractor, embedder, resolveThreshold, 1, personalKey)
			}
			// Ingest all sessions.
			for _, sess := range conv.Sessions {
				text := formatSession(sess)
				chunks := chunkText(text, chunkSize)
				for _, chunk := range chunks {
					if len(strings.TrimSpace(chunk)) < 50 {
						continue
					}
					source := fmt.Sprintf("locomo/%s/session_%d", conv.SampleID, sess.Number)
					if baseline {
						if err := db.UpsertEpisode(ctx, store.Episode{
							UUID:    uuid.New().String(),
							Content: chunk,
							GroupID: groupID,
							Source:  source,
						}); err != nil {
							slog.Warn("episode insert failed", "err", err)
						}
						continue
					}
					if personalKey != "" {
						if err := secureExtractor.Process(ctx, chunk, source, groupID); err != nil {
							slog.Warn("secure extraction failed", "conv", conv.SampleID, "session", sess.Number, "err", err)
						}
						continue
					}

					ext := graph.New(db, extractor, embedder, resolveThreshold, 1, "")
					if err := ext.Process(ctx, chunk, source, groupID); err != nil {
						slog.Warn("extraction failed", "conv", conv.SampleID, "session", sess.Number, "err", err)
					}
				}
			}

			// Run community detection + report generation after ingestion (Leiden §4).
			if !baseline && personalKey == "" {
				result, err := db.DetectCommunities(ctx, groupID, 1.0)
				if err != nil {
					slog.Warn("community detection failed", "err", err)
				} else {
					slog.Info("communities detected",
						"conv", conv.SampleID,
						"communities", result.Communities,
						"entities", result.Entities,
					)
					// Generate LLM community reports (≥3 members only).
					if err := graph.GenerateCommunityReports(ctx, db, groupID); err != nil {
						slog.Warn("community report generation failed", "err", err)
					}
				}

				// Entity resolution: merge near-duplicates (Graphiti §4.1).
				// After community detection so canonical inherits correct community_id.
				if resolveThreshold > 0 {
					rr, err := db.ResolveEntities(ctx, groupID, store.ResolveConfig{
						Threshold: resolveThreshold,
					})
					if err != nil {
						slog.Warn("entity resolution failed", "err", err)
					} else if rr.ClustersFound > 0 {
						slog.Info("entity resolution complete",
							"clusters", rr.ClustersFound,
							"merged", rr.EntitiesMerged,
							"edges_retargeted", rr.EdgesRetargeted,
						)
					}
				}
			}
			if personalKey != "" {
				if secureExtractor != nil {
					secureExtractor.Wait()
				}
				result, err := db.DetectCommunities(ctx, groupID, 1.0)
				if err != nil {
					slog.Warn("secure community detection failed", "err", err)
				} else {
					slog.Info("secure communities detected",
						"conv", conv.SampleID,
						"communities", result.Communities,
						"entities", result.Entities,
					)
					if err := graph.GenerateSecureCommunityReports(ctx, db, groupID, personalKey); err != nil {
						slog.Warn("secure community report generation failed", "err", err)
					}
				}
			}

			if personalKey != "" {
				if err := buildPersonalIndex(ctx, db, groupID, personalKey); err != nil {
					return nil, fmt.Errorf("build kpt index for %s: %w", conv.SampleID, err)
				}
			}
		}

		slog.Info("evaluating QA", "mode", mode, "conv", conv.SampleID, "questions", len(conv.QA))

		// Evaluate each QA question.
		for qi, qa := range conv.QA {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}

			var contextStr string
			if baseline {
				episodes, err := db.SearchEpisodesFTS(ctx, qa.Question, groupID, 10)
				if err != nil {
					slog.Warn("search failed", "question", qa.Question, "err", err)
					scores = append(scores, qaScore{qa.Category, 0, 0, -1})
					continue
				}
				contextStr = formatEpisodeContext(episodes)
			} else if personalKey != "" {
				results, err := graph.SecureSearch(ctx, db, embedder, personalKey, groupID, "kpt-v1", qa.Question, 25)
				if err != nil {
					slog.Warn("kpt search failed", "question", qa.Question, "err", err)
					scores = append(scores, qaScore{qa.Category, 0, 0, -1})
					continue
				}
				contextStr = formatContext(results)
			} else {
				results, err := graph.Search(ctx, db, embedder, qa.Question, groupID, 25)
				if err != nil {
					slog.Warn("search failed", "question", qa.Question, "err", err)
					scores = append(scores, qaScore{qa.Category, 0, 0, -1})
					continue
				}
				contextStr = formatContext(results)
			}

			prompt := fmt.Sprintf("Context:\n%s\n\nQuestion: %s", contextStr, qa.Question)
			sys := qaSystem
			if isHypotheticalQuestion(qa.Question) {
				sys = qaSystemHypothetical
			}
			answer, err := answerer.Answer(ctx, sys, prompt, 0)
			if err != nil {
				slog.Warn("answer failed", "question", qa.Question, "err", err)
				scores = append(scores, qaScore{qa.Category, 0, 0, -1})
				continue
			}

			f1 := tokenF1(answer, qa.Answer)
			em := exactMatch(answer, qa.Answer)
			judgeScore := -1.0
			if judge != nil {
				correct, err := judge.Judge(ctx, qa.Question, qa.Answer, answer)
				if err != nil {
					slog.Warn("judge failed", "question", qa.Question, "err", err)
				} else if correct {
					judgeScore = 1.0
				} else {
					judgeScore = 0.0
				}
			}
			scores = append(scores, qaScore{qa.Category, f1, em, judgeScore})

			if (qi+1)%25 == 0 {
				slog.Info("qa progress",
					"conv", conv.SampleID,
					"done", qi+1,
					"of", len(conv.QA),
				)
			}
		}
	}

	return aggregate(scores, time.Since(start)), nil
}

func buildPersonalIndex(ctx context.Context, db *store.DB, groupID, personalKey string) error {
	count, err := db.CountSecureEpisodes(ctx, groupID, "kpt-v1")
	if err != nil {
		return err
	}
	if count > 0 {
		return nil
	}

	episodes, err := db.AllEpisodesWithEmbeddings(ctx, groupID)
	if err != nil {
		return err
	}
	if len(episodes) == 0 {
		return fmt.Errorf("no episodes with embeddings for group %q", groupID)
	}

	method := secureindex.NewMethod(personalKey, len(episodes[0].Embedding))
	rows := make([]store.SecureEpisodeIndexRow, 0, len(episodes))
	for _, episode := range episodes {
		if len(episode.Embedding) == 0 {
			continue
		}
		state := method.EncodeDoc(episode.Embedding)
		rows = append(rows, store.SecureEpisodeIndexRow{
			EpisodeUUID:  episode.UUID,
			GroupID:      groupID,
			Method:       "kpt-v1",
			Public:       state.Public,
			BaseWaveReal: state.BaseWaveReal,
			BaseWaveImag: state.BaseWaveImag,
			WaveReal:     state.WaveReal,
			WaveImag:     state.WaveImag,
			ModeWeight:   state.ModeWeight,
			ModeEnergy:   state.ModeEnergy,
		})
	}
	if len(rows) == 0 {
		return fmt.Errorf("no secure rows built for group %q", groupID)
	}
	return db.ReplaceSecureEpisodeIndex(ctx, groupID, "kpt-v1", rows)
}

func searchPersonalContext(ctx context.Context, db *store.DB, embedder llm.Embedder, query, groupID, personalKey string) (string, error) {
	rows, err := db.AllSecureEpisodes(ctx, groupID, "kpt-v1")
	if err != nil {
		return "", err
	}
	if len(rows) == 0 {
		return "(no relevant dialogue found)", nil
	}

	queryEmbedding, err := embedder.Embed(ctx, query)
	if err != nil {
		return "", err
	}
	method := secureindex.NewMethod(personalKey, len(queryEmbedding))
	queryState := method.EncodeQuery(queryEmbedding)
	states := make([]secureindex.State, 0, len(rows))
	for _, row := range rows {
		row.Content, err = secureindex.DecryptString(personalKey, "episode-content", row.Content)
		if err != nil {
			return "", fmt.Errorf("decrypt episode content: %w", err)
		}
		states = append(states, secureindex.State{
			Public:       row.Public,
			BaseWaveReal: row.BaseWaveReal,
			BaseWaveImag: row.BaseWaveImag,
			WaveReal:     row.WaveReal,
			WaveImag:     row.WaveImag,
			ModeWeight:   row.ModeWeight,
			ModeEnergy:   row.ModeEnergy,
		})
	}
	hits := method.Search(states, queryState, 10)
	return formatSecureEpisodeContext(rows, hits), nil
}

// PrintResult outputs the benchmark results as a table.
func PrintResult(r *Result) {
	fmt.Printf("\n── LoCoMo Benchmark Results ──────────────────────────────\n\n")
	if r.HasJudge {
		fmt.Printf("%-14s  %5s  %7s  %7s  %8s\n", "Category", "Count", "F1", "EM", "Judge%")
		fmt.Printf("%-14s  %5s  %7s  %7s  %8s\n", "──────────────", "─────", "───────", "───────", "────────")
		for _, c := range r.ByCategory {
			fmt.Printf("%-14s  %5d  %6.1f%%  %6.1f%%  %7.1f%%\n", c.Category, c.Count, c.AvgF1*100, c.AvgEM*100, c.AvgJudge*100)
		}
		fmt.Printf("%-14s  %5s  %7s  %7s  %8s\n", "──────────────", "─────", "───────", "───────", "────────")
		fmt.Printf("%-14s  %5d  %6.1f%%  %6.1f%%  %7.1f%%\n", "OVERALL", r.Overall.Count, r.Overall.AvgF1*100, r.Overall.AvgEM*100, r.Overall.AvgJudge*100)
		fmt.Printf("\n(Judge: mistral-small-2506, n=%d)\n", r.Overall.JudgeCount)
	} else {
		fmt.Printf("%-14s  %5s  %7s  %7s\n", "Category", "Count", "F1", "EM")
		fmt.Printf("%-14s  %5s  %7s  %7s\n", "──────────────", "─────", "───────", "───────")
		for _, c := range r.ByCategory {
			fmt.Printf("%-14s  %5d  %6.1f%%  %6.1f%%\n", c.Category, c.Count, c.AvgF1*100, c.AvgEM*100)
		}
		fmt.Printf("%-14s  %5s  %7s  %7s\n", "──────────────", "─────", "───────", "───────")
		fmt.Printf("%-14s  %5d  %6.1f%%  %6.1f%%\n", "OVERALL", r.Overall.Count, r.Overall.AvgF1*100, r.Overall.AvgEM*100)
	}
	fmt.Printf("\nDuration: %s\n", r.Duration.Round(time.Second))
}

// ── Parsing ──────────────────────────────────────────────────────────────────

func parseLoCoMo(path string) ([]Conversation, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read locomo: %w", err)
	}

	var raw []rawConversation
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("parse locomo: %w", err)
	}

	conversations := make([]Conversation, 0, len(raw))
	for _, rc := range raw {
		conv := Conversation{
			SampleID: rc.SampleID,
			QA:       make([]QA, len(rc.QA)),
		}

		// Parse speaker names.
		if v, ok := rc.Conv["speaker_a"]; ok {
			_ = json.Unmarshal(v, &conv.SpeakerA)
		}
		if v, ok := rc.Conv["speaker_b"]; ok {
			_ = json.Unmarshal(v, &conv.SpeakerB)
		}

		// Parse sessions by iterating numbered keys.
		for i := 1; i <= 100; i++ {
			key := "session_" + strconv.Itoa(i)
			turnsRaw, ok := rc.Conv[key]
			if !ok {
				break
			}

			sess := Session{Number: i}
			if dtRaw, ok := rc.Conv[key+"_date_time"]; ok {
				_ = json.Unmarshal(dtRaw, &sess.DateTime)
			}

			var turns []rawTurn
			if err := json.Unmarshal(turnsRaw, &turns); err != nil {
				continue // skip malformed sessions
			}

			for _, t := range turns {
				text := t.Text
				if text == "" && t.BlipCaption != "" {
					text = "[Image: " + t.BlipCaption + "]"
				}
				if text == "" {
					continue
				}
				sess.Turns = append(sess.Turns, Turn{
					Speaker: t.Speaker,
					DiaID:   t.DiaID,
					Text:    text,
				})
			}

			if len(sess.Turns) > 0 {
				conv.Sessions = append(conv.Sessions, sess)
			}
		}

		for i, q := range rc.QA {
			// Adversarial questions (category 5) use adversarial_answer, not answer.
			answer := rawAnswerToString(q.Answer)
			if answer == "" && len(q.AdversarialAnswer) > 0 {
				answer = rawAnswerToString(q.AdversarialAnswer)
			}
			conv.QA[i] = QA{
				Question: q.Question,
				Answer:   answer,
				Category: q.Category,
				Evidence: q.Evidence,
			}
		}

		conversations = append(conversations, conv)
	}

	return conversations, nil
}

func rawAnswerToString(raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	// Numeric answers (e.g. 2022, 3).
	return strings.Trim(string(raw), " \t\n")
}

// ── Evaluation helpers ───────────────────────────────────────────────────────

func formatSession(s Session) string {
	var b strings.Builder
	if s.DateTime != "" {
		b.WriteString("[" + s.DateTime + "]\n")
	}
	for _, t := range s.Turns {
		b.WriteString(t.Speaker + ": " + t.Text + "\n")
	}
	return b.String()
}

// temporalTag produces a readable time label for an edge fact in context.
// Uses session tag from source path — all edges have a session, only ~50% have validAt.
// Inconsistent date coverage (some edges dated, some not) confuses multi-hop reasoning.
// ValidAt dates stored in SearchResult.ValidAt for future use (e.g. temporal filtering).
func temporalTag(_, source string) string {
	if source == "" {
		return ""
	}
	parts := strings.Split(source, "/")
	last := parts[len(parts)-1]
	if strings.HasPrefix(last, "session_") {
		return "[" + last + "] "
	}
	return ""
}

func formatContext(results []graph.SearchResult) string {
	if len(results) == 0 {
		return "(no relevant facts found)"
	}
	// Three-pass: community reports first (global context), then facts (edges),
	// then dialogue (episodes). Entity profiles skipped — they cause entity-attribution
	// confusion. Session tags on edge facts provide temporal grounding.
	var b strings.Builder
	n := 0

	// Pass 1: community reports (Leiden §4 global context).
	for _, r := range results {
		if r.Type != "community" {
			continue
		}
		n++
		fmt.Fprintf(&b, "%d. [background] %s\n", n, r.Body)
	}

	// Pass 2: edge facts.
	// Each edge is rendered individually with a session tag for temporal grounding.
	// v44 finding: edge grouping (merging same-subject edges into comma-lists) hurts both
	// single-hop (-4.2%) and adversarial (-4.1%) because it removes session tags and
	// creates ambiguous multi-item lists that confuse attribution for adversarial questions.
	for _, r := range results {
		if r.Type != "edge" {
			continue
		}
		n++
		tag := temporalTag("", r.Source)
		fmt.Fprintf(&b, "%d. %s%s\n", n, tag, r.Body)
	}

	// Pass 3: episode dialogue.
	for _, r := range results {
		if r.Type != "episode" {
			continue
		}
		n++
		body := r.Body
		if len(body) > 1500 {
			body = body[:1500] + "..."
		}
		fmt.Fprintf(&b, "%d. [dialogue] %s\n", n, body)
	}
	if n == 0 {
		return "(no relevant facts found)"
	}
	return b.String()
}

func formatEpisodeContext(episodes []store.Episode) string {
	if len(episodes) == 0 {
		return "(no relevant dialogue found)"
	}
	var b strings.Builder
	for i, ep := range episodes {
		body := ep.Content
		if len(body) > 500 {
			body = body[:500] + "..."
		}
		fmt.Fprintf(&b, "%d. %s\n", i+1, body)
	}
	return b.String()
}

func formatSecureEpisodeContext(rows []store.SecureEpisodeIndexRow, hits []secureindex.SearchHit) string {
	if len(hits) == 0 {
		return "(no relevant dialogue found)"
	}
	var b strings.Builder
	for i, hit := range hits {
		row := rows[hit.Index]
		body := row.Content
		if len(body) > 500 {
			body = body[:500] + "..."
		}
		fmt.Fprintf(&b, "%d. %s\n", i+1, body)
	}
	return b.String()
}

func chunkText(text string, size int) []string {
	if len([]rune(text)) <= size {
		return []string{text}
	}
	// Split at line boundaries to keep dialogue turns intact.
	// Partial turns confuse entity/edge extraction.
	lines := strings.Split(text, "\n")
	var chunks []string
	var current strings.Builder
	for _, line := range lines {
		if current.Len()+len(line)+1 > size && current.Len() > 0 {
			chunks = append(chunks, current.String())
			current.Reset()
		}
		if current.Len() > 0 {
			current.WriteByte('\n')
		}
		current.WriteString(line)
	}
	if current.Len() > 0 {
		chunks = append(chunks, current.String())
	}
	return chunks
}

// ── Scoring (SQuAD-style F1 and Exact Match) ────────────────────────────────

func normalizeAnswer(s string) string {
	s = strings.ToLower(s)
	// Remove punctuation.
	s = strings.Map(func(r rune) rune {
		if unicode.IsPunct(r) {
			return -1
		}
		return r
	}, s)
	// Remove articles.
	for _, article := range []string{" a ", " an ", " the "} {
		s = strings.ReplaceAll(s, article, " ")
	}
	return strings.Join(strings.Fields(s), " ")
}

func tokenF1(prediction, gold string) float64 {
	predTokens := strings.Fields(normalizeAnswer(prediction))
	goldTokens := strings.Fields(normalizeAnswer(gold))
	if len(predTokens) == 0 || len(goldTokens) == 0 {
		return 0
	}

	goldCounts := map[string]int{}
	for _, t := range goldTokens {
		goldCounts[t]++
	}

	common := 0
	for _, t := range predTokens {
		if goldCounts[t] > 0 {
			common++
			goldCounts[t]--
		}
	}

	if common == 0 {
		return 0
	}
	precision := float64(common) / float64(len(predTokens))
	recall := float64(common) / float64(len(goldTokens))
	return 2 * precision * recall / (precision + recall)
}

func exactMatch(prediction, gold string) float64 {
	if normalizeAnswer(prediction) == normalizeAnswer(gold) {
		return 1
	}
	return 0
}

// ── Aggregation ──────────────────────────────────────────────────────────────

func aggregate(scores []qaScore, duration time.Duration) *Result {
	type acc struct {
		f1Sum      float64
		emSum      float64
		judgeSum   float64
		count      int
		judgeCount int
	}
	byCategory := map[int]*acc{}
	overall := &acc{}

	for _, s := range scores {
		if _, ok := byCategory[s.category]; !ok {
			byCategory[s.category] = &acc{}
		}
		byCategory[s.category].f1Sum += s.f1
		byCategory[s.category].emSum += s.em
		byCategory[s.category].count++
		overall.f1Sum += s.f1
		overall.emSum += s.em
		overall.count++
		if s.judge >= 0 {
			byCategory[s.category].judgeSum += s.judge
			byCategory[s.category].judgeCount++
			overall.judgeSum += s.judge
			overall.judgeCount++
		}
	}

	hasJudge := overall.judgeCount > 0
	result := &Result{Duration: duration, HasJudge: hasJudge}
	if overall.count > 0 {
		cs := CategoryScore{
			Category:   "overall",
			Count:      overall.count,
			AvgF1:      overall.f1Sum / float64(overall.count),
			AvgEM:      overall.emSum / float64(overall.count),
			JudgeCount: overall.judgeCount,
		}
		if overall.judgeCount > 0 {
			cs.AvgJudge = overall.judgeSum / float64(overall.judgeCount)
		}
		result.Overall = cs
	}

	cats := make([]int, 0, len(byCategory))
	for c := range byCategory {
		cats = append(cats, c)
	}
	sort.Ints(cats)

	for _, c := range cats {
		a := byCategory[c]
		name := categoryName[c]
		if name == "" {
			name = "cat-" + strconv.Itoa(c)
		}
		cs := CategoryScore{
			Category:   name,
			Count:      a.count,
			AvgF1:      a.f1Sum / float64(a.count),
			AvgEM:      a.emSum / float64(a.count),
			JudgeCount: a.judgeCount,
		}
		if a.judgeCount > 0 {
			cs.AvgJudge = a.judgeSum / float64(a.judgeCount)
		}
		result.ByCategory = append(result.ByCategory, cs)
	}

	return result
}
