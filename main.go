// memory-local — portable knowledge graph builder using SQLite + pluggable LLM backends.
//
// Dependencies: Go, SQLite (embedded), plus the active build profile runtime.
//
// Usage:
//
//	memory-local ingest <path>          # walk dir, queue all text files
//	memory-local search <query>         # hybrid search (FTS + vector)
//	memory-local communities            # list detected communities
//	memory-local status                 # queue + graph stats
//	memory-local worker                 # run the extraction worker (blocking)
//	memory-local run <path>             # ingest + worker in one shot
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/sharpner/ultramemory/bench"
	"github.com/sharpner/ultramemory/graph"
	"github.com/sharpner/ultramemory/ingest"
	"github.com/sharpner/ultramemory/llm"
	"github.com/sharpner/ultramemory/secureindex"
	"github.com/sharpner/ultramemory/store"
)

// version is set at build time via -ldflags "-X main.version=..."
// Falls back to "(dev)" for go run / untagged builds.
var version = "(dev)"

const (
	defaultDB    = "memory-local.db"
	defaultGroup = "default"
	pollInterval = 200 * time.Millisecond
	secureMethod = "kpt-v1"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}
	if os.Args[1] == "--version" || os.Args[1] == "-version" || os.Args[1] == "version" {
		fmt.Println("ultramemory " + version)
		return
	}

	// Config from env (overridable).
	dbPath := envOr("MEMORY_DB", defaultDB)
	extractModel := envOr("MEMORY_MODEL", defaultExtractModel)
	embedModel := envOr("MEMORY_EMBED_MODEL", defaultEmbeddingModel)
	groupID := envOr("MEMORY_GROUP", defaultGroup)
	resolveThreshold := 0.92
	if v := os.Getenv("MEMORY_RESOLVE_THRESHOLD"); v != "" {
		t, err := strconv.ParseFloat(v, 64)
		if err != nil || t <= 0 || t > 1 {
			fatalf("MEMORY_RESOLVE_THRESHOLD must be a float in (0, 1], got %q", v)
		}
		resolveThreshold = t
	}
	llmParallel := defaultLLMParallel
	if v := os.Getenv("MEMORY_LLM_PARALLEL"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 {
			fatalf("MEMORY_LLM_PARALLEL must be a positive integer, got %q", v)
		}
		llmParallel = n
	}

	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	})))

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	db, err := store.Open(dbPath)
	must(err, "open db")
	defer db.Close() //nolint:errcheck

	runtimeProfile, err := newDefaultRuntime(extractModel, embedModel)
	must(err, "configure runtime")
	slog.Info("runtime configured", "provider", buildProviderName, "extract_model", extractModel, "embed_model", embedModel)
	personalKey := secureKeyOrEnv("")

	switch os.Args[1] {
	case "ingest":
		fs := flag.NewFlagSet("ingest", flag.ExitOnError)
		source := fs.String("source", "", "source label (e.g. arXiv URL) instead of file path")
		_ = fs.Parse(os.Args[2:])
		if fs.NArg() < 1 {
			fatalf("usage: ultramemory ingest [-source URL] <path>")
		}
		rejectTrailingFlags(fs)
		must(runtimeProfile.ping(ctx), "ping runtime")
		w := ingest.New(db, groupID).WithOCR(runtimeProfile.ocr).WithPersonalKey(personalKey)
		if *source != "" {
			w = w.WithSource(*source)
		}
		n, err := w.Walk(ctx, fs.Arg(0))
		must(err, "walk")
		fmt.Fprintf(os.Stderr, "✓ Queued %d chunks from %s\n", n, fs.Arg(0))

	case "worker":
		must(runtimeProfile.ping(ctx), "ping runtime")
		fmt.Fprintln(os.Stderr, "Warming up model…")
		if err := runtimeProfile.warmup(ctx); err != nil {
			slog.Warn("warmup failed", "err", err)
		}
		runWorker(ctx, db, runtimeProfile.extractor, runtimeProfile.embedder, resolveThreshold, llmParallel, groupID, personalKey)

	case "run":
		fs := flag.NewFlagSet("run", flag.ExitOnError)
		source := fs.String("source", "", "source label (e.g. arXiv URL) instead of file path")
		_ = fs.Parse(os.Args[2:])
		if fs.NArg() < 1 {
			fatalf("usage: ultramemory run [-source URL] <path>")
		}
		rejectTrailingFlags(fs)
		must(runtimeProfile.ping(ctx), "ping runtime")
		fmt.Fprintln(os.Stderr, "Warming up model…")
		if err := runtimeProfile.warmup(ctx); err != nil {
			slog.Warn("warmup failed", "err", err)
		}

		w := ingest.New(db, groupID).WithOCR(runtimeProfile.ocr).WithPersonalKey(personalKey)
		if *source != "" {
			w = w.WithSource(*source)
		}
		n, err := w.Walk(ctx, fs.Arg(0))
		must(err, "walk")
		fmt.Fprintf(os.Stderr, "✓ Queued %d chunks — starting worker (Ctrl+C to stop)\n", n)
		runWorker(ctx, db, runtimeProfile.extractor, runtimeProfile.embedder, resolveThreshold, llmParallel, groupID, personalKey)

	case "search":
		fs := flag.NewFlagSet("search", flag.ExitOnError)
		format := fs.String("format", "text", "output format: text|json")
		maxTokens := fs.Int("max-tokens", 0, "token budget for output (0 = unlimited)")
		_ = fs.Parse(os.Args[2:])
		if fs.NArg() < 1 {
			fatalf("usage: ultramemory search [-format text|json] [-max-tokens N] <query>")
		}
		must(runtimeProfile.ping(ctx), "ping runtime")
		query := strings.Join(fs.Args(), " ")
		if personalKey != "" {
			results, err := graph.SecureSearch(ctx, db, runtimeProfile.embedder, personalKey, groupID, secureMethod, query, 10)
			must(err, "secure search")
			printSearch(results, query, *format, *maxTokens)
			break
		}
		results, err := graph.Search(ctx, db, runtimeProfile.embedder, query, groupID, 10)
		must(err, "search")
		printSearch(results, query, *format, *maxTokens)

	case "personal-index":
		fs := flag.NewFlagSet("personal-index", flag.ExitOnError)
		methodName := fs.String("method", secureMethod, "secure index method name")
		key := fs.String("key", "", "personal key (fallback: MEMORY_PERSONAL_KEY)")
		_ = fs.Parse(os.Args[2:])
		secureKey := secureKeyOrEnv(*key)
		if secureKey == "" {
			fatalf("usage: ultramemory personal-index [-method %s] -key <secret>", secureMethod)
		}
		existing, err := db.CountSecureEpisodes(ctx, groupID, *methodName)
		must(err, "count secure index")
		if existing > 0 {
			fmt.Fprintf(os.Stderr, "✓ Secure episode index already present for %d episodes in group %q\n", existing, groupID)
			break
		}
		episodes, err := db.AllEpisodesWithEmbeddings(ctx, groupID)
		must(err, "load episodes")
		if len(episodes) == 0 {
			fatalf("no episodes with embeddings found in group %q; run ingest/worker first", groupID)
		}
		method := secureindex.NewMethod(secureKey, len(episodes[0].Embedding))
		rows := make([]store.SecureEpisodeIndexRow, 0, len(episodes))
		for _, episode := range episodes {
			state := method.EncodeDoc(episode.Embedding)
			rows = append(rows, store.SecureEpisodeIndexRow{
				EpisodeUUID:  episode.UUID,
				GroupID:      groupID,
				Method:       *methodName,
				Public:       state.Public,
				BaseWaveReal: state.BaseWaveReal,
				BaseWaveImag: state.BaseWaveImag,
				WaveReal:     state.WaveReal,
				WaveImag:     state.WaveImag,
				ModeWeight:   state.ModeWeight,
				ModeEnergy:   state.ModeEnergy,
			})
		}
		must(db.ReplaceSecureEpisodeIndex(ctx, groupID, *methodName, rows), "store secure index")
		fmt.Fprintf(os.Stderr, "✓ Built %s secure episode index for %d episodes in group %q\n", *methodName, len(rows), groupID)

	case "personal-search":
		fs := flag.NewFlagSet("personal-search", flag.ExitOnError)
		methodName := fs.String("method", secureMethod, "secure index method name")
		key := fs.String("key", "", "personal key (fallback: MEMORY_PERSONAL_KEY)")
		format := fs.String("format", "text", "output format: text|json")
		limit := fs.Int("limit", 10, "maximum number of hits")
		_ = fs.Parse(os.Args[2:])
		if fs.NArg() < 1 {
			fatalf("usage: ultramemory personal-search [-method %s] [-limit N] -key <secret> <query>", secureMethod)
		}
		secureKey := secureKeyOrEnv(*key)
		if secureKey == "" {
			fatalf("usage: ultramemory personal-search [-method %s] [-limit N] -key <secret> <query>", secureMethod)
		}
		rows, err := db.AllSecureEpisodes(ctx, groupID, *methodName)
		must(err, "load secure index")
		if len(rows) == 0 {
			fatalf("no secure episode index found for group %q and method %q; run ultramemory personal-index first", groupID, *methodName)
		}
		must(runtimeProfile.ping(ctx), "ping runtime")
		query := strings.Join(fs.Args(), " ")
		queryEmb, err := runtimeProfile.embedder.Embed(ctx, query)
		must(err, "embed query")
		method := secureindex.NewMethod(secureKey, len(queryEmb))
		queryState := method.EncodeQuery(queryEmb)
		rows, err = decryptSecureRows(rows, secureKey)
		must(err, "decrypt secure rows")
		docs := secureStates(rows)
		hits := method.Search(docs, queryState, *limit)
		printPersonalSearch(rows, hits, query, *format, 0)

	case "personal-cluster":
		fs := flag.NewFlagSet("personal-cluster", flag.ExitOnError)
		methodName := fs.String("method", secureMethod, "secure index method name")
		key := fs.String("key", "", "personal key (fallback: MEMORY_PERSONAL_KEY)")
		format := fs.String("format", "text", "output format: text|json")
		k := fs.Int("k", 10, "neighbors per episode in the keyed similarity graph")
		minMembers := fs.Int("min", 2, "minimum members per cluster")
		minScore := fs.Float64("min-score", 0.45, "minimum keyed edge score to keep")
		resolution := fs.Float64("resolution", 1.0, "Louvain resolution")
		_ = fs.Parse(os.Args[2:])
		secureKey := secureKeyOrEnv(*key)
		if secureKey == "" {
			fatalf("usage: ultramemory personal-cluster [-method %s] [-k N] -key <secret>", secureMethod)
		}
		rows, err := db.AllSecureEpisodes(ctx, groupID, *methodName)
		must(err, "load secure index")
		if len(rows) == 0 {
			fatalf("no secure episode index found for group %q and method %q; run ultramemory personal-index first", groupID, *methodName)
		}
		method := secureindex.NewMethod(secureKey, len(rows[0].BaseWaveReal))
		clusters := method.Cluster(secureStates(rows), *k, *resolution, *minScore)
		rows, err = decryptSecureRows(rows, secureKey)
		must(err, "decrypt secure rows")
		printPersonalClusters(rows, clusters, *format, *minMembers)

	case "bench":
		fs := flag.NewFlagSet("bench", flag.ExitOnError)
		limit := fs.Int("limit", 0, "max conversations to evaluate (0 = all)")
		baseline := fs.Bool("baseline", false, "baseline mode: episode FTS only, no graph extraction")
		personalKey := fs.String("personal-key", "", "enable KPT retrieval in LoCoMo with this personal key")
		qaModel := fs.String("qa-model", "", "override QA answering model: 'mistral-small-2506' etc (default: same as extraction model)")
		qaOnly := fs.Bool("qa-only", false, "skip ingestion, run QA on existing DB (use with -qa-model for fast model swapping)")
		judgeModel := fs.String("judge", "", "LLM judge model for semantic evaluation: 'mistral-small-2506' (requires MISTRAL_API_KEY)")
		_ = fs.Parse(os.Args[2:])
		if fs.NArg() < 1 {
			fatalf("usage: ultramemory bench [-limit N] [-baseline] [-personal-key KEY] [-qa-model MODEL] [-qa-only] [-judge MODEL] <locomo10.json>")
		}
		if !*qaOnly {
			must(runtimeProfile.ping(ctx), "ping runtime")
			fmt.Fprintln(os.Stderr, "Warming up model…")
			if err := runtimeProfile.warmup(ctx); err != nil {
				slog.Warn("warmup failed", "err", err)
			}
		}

		qaAnswerer := runtimeProfile.answerer
		if *qaModel != "" {
			qaAnswerer = mustNewMistralClient(*qaModel, "-qa-model "+*qaModel)
			slog.Info("QA answerer", "model", *qaModel, "provider", "mistral")
		}

		var judge bench.Judge
		if *judgeModel != "" {
			judge = mustNewMistralClient(*judgeModel, "-judge "+*judgeModel)
			slog.Info("LLM judge", "model", *judgeModel, "provider", "mistral")
		}

		result, err := bench.RunLoCoMo(ctx, fs.Arg(0), db, runtimeProfile.extractor, runtimeProfile.embedder, qaAnswerer, judge, resolveThreshold, *limit, *baseline, *qaOnly, *personalKey)
		must(err, "bench")
		bench.PrintResult(result)

	case "resolve":
		fs := flag.NewFlagSet("resolve", flag.ExitOnError)
		dryRun := fs.Bool("dry-run", false, "print planned merges without writing")
		threshold := fs.Float64("threshold", 0.85, "cosine similarity threshold (0–1]")
		_ = fs.Parse(os.Args[2:])
		if *threshold <= 0 || *threshold > 1 {
			fatalf("--threshold must be in (0, 1], got %g", *threshold)
		}
		if personalKey != "" {
			result, err := db.ResolveSecureEntities(ctx, groupID, personalKey, store.ResolveConfig{
				Threshold: *threshold,
				DryRun:    *dryRun,
			})
			must(err, "resolve secure entities")
			printResolveResult(result, *dryRun)
			break
		}
		result, err := db.ResolveEntities(ctx, groupID, store.ResolveConfig{
			Threshold: *threshold,
			DryRun:    *dryRun,
		})
		must(err, "resolve")
		printResolveResult(result, *dryRun)

	case "retry":
		n, err := db.RequeueFailed(ctx)
		must(err, "requeue failed jobs")
		fmt.Fprintf(os.Stderr, "✓ Requeued %d failed jobs\n", n)

	case "communities":
		fs := flag.NewFlagSet("communities", flag.ExitOnError)
		format := fs.String("format", "text", "output format: text|json")
		minMembers := fs.Int("min", 2, "minimum members to show a community")
		detect := fs.Bool("detect", false, "run community detection (writes to DB)")
		ricci := fs.Bool("ricci", false, "use Mutual-kNN + ORC instead of Louvain")
		resolution := fs.Float64("resolution", 1.0, "Louvain resolution (higher = more, smaller communities)")
		_ = fs.Parse(os.Args[2:])
		if *detect && *ricci && personalKey == "" {
			fmt.Fprintln(os.Stderr, "Running Mutual-kNN + ORC + Louvain…")
			cr, err := db.MutualKNNCommunities(ctx, groupID, 20, *resolution)
			must(err, "mutual-knn communities")
			fmt.Fprintf(os.Stderr, "✓ %d communities across %d entities (Mutual-kNN + ORC)\n",
				cr.Communities, cr.Entities)
		}
		if personalKey != "" {
			if *detect {
				cr := store.CommunityResult{}
				err := error(nil)
				if *ricci {
					fmt.Fprintln(os.Stderr, "Running secure ORC + Louvain community detection on the encrypted entity graph…")
					cr, err = db.SecureRicciCommunities(ctx, groupID, *resolution)
				}
				if !*ricci {
					fmt.Fprintln(os.Stderr, "Running secure community detection on the encrypted entity graph…")
					cr, err = db.DetectCommunities(ctx, groupID, *resolution)
				}
				must(err, "detect secure communities")
				fmt.Fprintf(os.Stderr, "✓ %d secure communities across %d entities\n", cr.Communities, cr.Entities)
				must(graph.GenerateSecureCommunityReports(ctx, db, groupID, personalKey), "generate secure community reports")
			}
			communities, err := db.ListSecureCommunities(ctx, groupID, personalKey)
			must(err, "list secure communities")
			printCommunitySummaries(communities, *format, *minMembers)
			break
		}
		if *detect && !*ricci {
			fmt.Fprintln(os.Stderr, "Running Louvain community detection…")
			cr, err := db.DetectCommunities(ctx, groupID, *resolution)
			must(err, "detect communities")
			fmt.Fprintf(os.Stderr, "✓ %d communities across %d entities\n", cr.Communities, cr.Entities)
		}
		if *detect {
			if err := graph.GenerateCommunityReports(ctx, db, groupID); err != nil {
				fmt.Fprintf(os.Stderr, "warning: community report generation failed: %v\n", err)
			}
		}
		if !*detect {
			printCommunities(ctx, db, groupID, *format, *minMembers)
		}

	case "curvature":
		fs := flag.NewFlagSet("curvature", flag.ExitOnError)
		persist := fs.Bool("store", false, "persist curvatures to edge_curvatures table")
		bridges := fs.Int("bridges", 0, "show top N bridge edges (most negative curvature)")
		format := fs.String("format", "text", "output format: text|json")
		_ = fs.Parse(os.Args[2:])
		if personalKey != "" {
			curvatures, stats, err := db.ComputeSecureGraphCurvatures(ctx, groupID, personalKey)
			must(err, "compute secure curvatures")
			printCurvatureStats(stats, *format)
			if *persist {
				must(db.StoreCurvatures(ctx, groupID, curvatures), "store secure curvatures")
			}
			if *bridges > 0 {
				if *persist {
					bridges, err := db.TopSecureBridges(ctx, groupID, *bridges, personalKey)
					must(err, "top secure bridges")
					printBridges(bridges, *format)
					break
				}
				slices.SortFunc(curvatures, func(a, b store.EdgeCurvature) int {
					if a.Curvature < b.Curvature {
						return -1
					}
					if a.Curvature > b.Curvature {
						return 1
					}
					return 0
				})
				n := min(*bridges, len(curvatures))
				printBridges(curvatures[:n], *format)
			}
			break
		}

		// If -bridges without computation, just query stored curvatures.
		if *bridges > 0 && !*persist {
			top, err := db.TopBridges(ctx, groupID, *bridges)
			if err == nil && len(top) > 0 {
				printBridges(top, *format)
				break
			}
		}

		fmt.Fprintln(os.Stderr, "Computing Ollivier-Ricci curvatures…")
		curvatures, stats, err := db.ComputeCurvatures(ctx, groupID, 0)
		must(err, "compute curvatures")
		printCurvatureStats(stats, *format)

		if *persist {
			fmt.Fprintln(os.Stderr, "Storing curvatures…")
			must(db.StoreCurvatures(ctx, groupID, curvatures), "store curvatures")
			fmt.Fprintf(os.Stderr, "✓ %d curvatures stored\n", len(curvatures))
		}

		if *bridges > 0 {
			top, err := db.TopBridges(ctx, groupID, *bridges)
			if err == nil {
				printBridges(top, *format)
				break
			}
			// Fall back to in-memory sort.
			slices.SortFunc(curvatures, func(a, b store.EdgeCurvature) int {
				if a.Curvature < b.Curvature {
					return -1
				}
				if a.Curvature > b.Curvature {
					return 1
				}
				return 0
			})
			n := min(*bridges, len(curvatures))
			printBridges(curvatures[:n], *format)
		}

	case "status":
		fs := flag.NewFlagSet("status", flag.ExitOnError)
		format := fs.String("format", "text", "output format: text|json")
		_ = fs.Parse(os.Args[2:])
		printStatus(ctx, db, groupID, secureMethod, personalKey != "", *format)

	default:
		usage()
		os.Exit(1)
	}
}

const staleJobTimeout = 5 * time.Minute

// runWorker polls the SQLite queue and processes jobs with max 1 concurrent LLM call.
// extractor handles entity/edge extraction; embedder handles all embeddings for the active build.
func runWorker(ctx context.Context, db *store.DB, extractor llm.EntityExtractor, embedder llm.Embedder, resolveThreshold float64, llmParallel int, groupID, personalKey string) {
	ext := graph.New(db, extractor, embedder, resolveThreshold, llmParallel, personalKey)
	secureMode := personalKey != ""
	concurrency := runtime.NumCPU()
	if concurrency > 4 {
		concurrency = 4
	}

	recoverStaleJobs(ctx, db)

	// Worker pool — but LLM semaphore in Extractor limits actual LLM calls to 1.
	jobs := make(chan *store.Job, concurrency)
	var workerWG sync.WaitGroup
	var inFlight atomic.Int64

	for i := 0; i < concurrency; i++ {
		workerWG.Add(1)
		go func() {
			defer workerWG.Done()
			for job := range jobs {
				if err := ext.ProcessJob(ctx, job.Payload, job.Attempts); err != nil {
					slog.Error("job failed", "id", job.ID, "err", err)
					if err := db.FailJob(context.Background(), job.ID, err.Error()); err != nil {
						slog.Error("fail job", "err", err)
					}
					inFlight.Add(-1)
					continue
				}
				if err := db.CompleteJob(context.Background(), job.ID); err != nil {
					slog.Error("complete job", "err", err)
				}
				inFlight.Add(-1)
			}
		}()
	}

	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()

	lastLog := time.Now()
	lastRecover := time.Now()
	processed := 0
	communityDirty := false // true when new jobs processed since last community detection

	for {
		select {
		case <-ctx.Done():
			close(jobs)
			workerWG.Wait()
			ext.Wait()
			fmt.Fprintf(os.Stderr, "\n✓ Worker stopped. Processed %d chunks.\n", processed)
			return
		case <-ticker.C:
			// Periodically recover stale jobs in case a worker goroutine panicked.
			if time.Since(lastRecover) > staleJobTimeout {
				recoverStaleJobs(ctx, db)
				lastRecover = time.Now()
			}

			queueEmpty := true
			for {
				job, err := db.NextJob(ctx)
				if err != nil {
					slog.Error("poll queue", "err", err)
					break
				}
				if job == nil {
					break // queue empty
				}
				queueEmpty = false
				communityDirty = true
				inFlight.Add(1)
				jobs <- job
				processed++

				// Progress log every 10s.
				if time.Since(lastLog) > 10*time.Second {
					stats, _ := db.QueueStats(ctx)
					slog.Info("progress",
						"processed", processed,
						"pending", stats["pending"],
						"done", stats["done"],
						"failed", stats["failed"],
					)
					lastLog = time.Now()
				}
			}

			// Run community detection once when queue drains after processing.
			if queueEmpty && communityDirty && inFlight.Load() == 0 {
				ext.Wait()
				slog.Info("queue drained — running community detection")
				cr, err := db.DetectCommunities(ctx, groupID, 1.0)
				if err != nil {
					slog.Error("community detection failed — will retry on next drain", "err", err)
				} else {
					communityDirty = false
					slog.Info("communities detected", "communities", cr.Communities, "entities", cr.Entities)
					if secureMode {
						continue
					}
					if err := graph.GenerateCommunityReports(ctx, db, groupID); err != nil {
						slog.Error("community reports failed", "err", err)
					}
				}
			}
		}
	}
}

// searchHit is the JSON-serialisable presentation of one search result.
type searchHit struct {
	Rank   int     `json:"rank"`
	Type   string  `json:"type"`
	Title  string  `json:"title"`
	Body   string  `json:"body,omitempty"`
	Score  float64 `json:"score"`
	Source string  `json:"source,omitempty"`
}

type personalSearchHit struct {
	Rank    int     `json:"rank"`
	UUID    string  `json:"uuid"`
	Score   float64 `json:"score"`
	Source  string  `json:"source,omitempty"`
	Content string  `json:"content"`
}

type personalClusterHit struct {
	ClusterID int      `json:"cluster_id"`
	Size      int      `json:"size"`
	Members   []string `json:"members"`
}

// approxTokens estimates token count using the standard 1 token ≈ 4 chars heuristic.
func approxTokens(s string) int {
	return (len(s) + 3) / 4
}

func printSearch(results []graph.SearchResult, query, format string, maxTokens int) {
	if format == "json" {
		enc := json.NewEncoder(os.Stdout)
		used := 0
		for i, r := range results {
			cost := approxTokens(r.Title + r.Body)
			if maxTokens > 0 && used+cost > maxTokens {
				break
			}
			_ = enc.Encode(searchHit{i + 1, r.Type, r.Title, r.Body, r.Score, r.Source})
			used += cost
		}
		return
	}
	if len(results) == 0 {
		fmt.Println("(no results)")
		return
	}
	fmt.Printf("Results for %q:\n\n", query)
	used := 0
	for i, r := range results {
		cost := approxTokens(r.Title + r.Body)
		if maxTokens > 0 && used+cost > maxTokens {
			fmt.Printf("-- token budget reached (%d/%d tokens used, %d result(s) omitted) --\n",
				used, maxTokens, len(results)-i)
			break
		}
		src := ""
		if r.Source != "" {
			src = "   source=" + filepath.Base(r.Source) + "\n"
		}
		fmt.Printf("%d. [%s] %s\n   %s\n   score=%.4f\n%s\n",
			i+1, r.Type, r.Title, r.Body, r.Score, src)
		used += cost
	}
}

func printPersonalSearch(rows []store.SecureEpisodeIndexRow, hits []secureindex.SearchHit, query, format string, maxTokens int) {
	if format == "json" {
		enc := json.NewEncoder(os.Stdout)
		used := 0
		for rank, hit := range hits {
			row := rows[hit.Index]
			cost := approxTokens(row.Source + row.Content)
			if maxTokens > 0 && used+cost > maxTokens {
				break
			}
			_ = enc.Encode(personalSearchHit{
				Rank:    rank + 1,
				UUID:    row.EpisodeUUID,
				Score:   hit.Score,
				Source:  row.Source,
				Content: row.Content,
			})
			used += cost
		}
		return
	}
	if len(hits) == 0 {
		fmt.Println("(no secure results)")
		return
	}
	fmt.Printf("Secure results for %q:\n\n", query)
	used := 0
	for rank, hit := range hits {
		row := rows[hit.Index]
		cost := approxTokens(row.Source + row.Content)
		if maxTokens > 0 && used+cost > maxTokens {
			fmt.Printf("-- token budget reached (%d/%d tokens used, %d result(s) omitted) --\n",
				used, maxTokens, len(hits)-rank)
			break
		}
		fmt.Printf("%d. [%0.4f] %s\n   %s\n\n", rank+1, hit.Score, filepath.Base(row.Source), truncateLine(row.Content, 180))
		used += cost
	}
}

func printPersonalClusters(rows []store.SecureEpisodeIndexRow, clusters []secureindex.Cluster, format string, minMembers int) {
	filtered := clusters[:0]
	for _, cluster := range clusters {
		if len(cluster.Members) < minMembers {
			continue
		}
		filtered = append(filtered, cluster)
	}

	if format == "json" {
		enc := json.NewEncoder(os.Stdout)
		for _, cluster := range filtered {
			members := make([]string, 0, len(cluster.Members))
			for _, member := range cluster.Members {
				row := rows[member]
				members = append(members, truncateLine(row.Content, 120))
			}
			_ = enc.Encode(personalClusterHit{
				ClusterID: cluster.ID,
				Size:      len(cluster.Members),
				Members:   members,
			})
		}
		return
	}

	if len(filtered) == 0 {
		fmt.Printf("(no personal clusters with >= %d members)\n", minMembers)
		return
	}

	fmt.Printf("── Personal Secure Clusters (%d shown) ────────\n\n", len(filtered))
	for _, cluster := range filtered {
		fmt.Printf("  Cluster %d (%d members)\n", cluster.ID, len(cluster.Members))
		for _, member := range cluster.Members {
			row := rows[member]
			fmt.Printf("    - %s\n", truncateLine(row.Content, 120))
		}
		fmt.Println()
	}
}

func printStatus(ctx context.Context, db *store.DB, groupID, method string, secureMode bool, format string) {
	stats, err := db.QueueStats(ctx)
	must(err, "queue stats")

	episodes, _ := db.CountEpisodes(ctx, groupID)
	entities, _ := db.CountEntities(ctx, groupID)
	edges, _ := db.CountEdges(ctx, groupID)
	secureEpisodes, _ := db.CountSecureEpisodes(ctx, groupID, method)

	if format == "json" {
		for _, s := range []string{"pending", "processing", "done", "failed"} {
			if _, ok := stats[s]; !ok {
				stats[s] = 0
			}
		}
		curvTotal, curvBridges, curvInternal, curvMean := db.CurvatureStatus(ctx, groupID)
		out := struct {
			Graph struct {
				Episodes       int `json:"episodes"`
				Entities       int `json:"entities"`
				Edges          int `json:"edges"`
				SecureEpisodes int `json:"secure_episodes"`
			} `json:"graph"`
			Mode      string `json:"mode"`
			Curvature *struct {
				Edges    int     `json:"edges"`
				Bridges  int     `json:"bridges"`
				Internal int     `json:"internal"`
				Mean     float64 `json:"mean"`
			} `json:"curvature,omitempty"`
			Queue map[string]int `json:"queue"`
		}{}
		out.Graph.Episodes = episodes
		out.Graph.Entities = entities
		out.Graph.Edges = edges
		out.Graph.SecureEpisodes = secureEpisodes
		if secureMode {
			out.Mode = "kpt"
		} else {
			out.Mode = "graph"
		}
		out.Queue = stats
		if curvTotal > 0 {
			out.Curvature = &struct {
				Edges    int     `json:"edges"`
				Bridges  int     `json:"bridges"`
				Internal int     `json:"internal"`
				Mean     float64 `json:"mean"`
			}{curvTotal, curvBridges, curvInternal, curvMean}
		}
		_ = json.NewEncoder(os.Stdout).Encode(out)
		return
	}

	fmt.Printf("── Graph ──────────────────\n")
	fmt.Printf("  episodes : %d\n", episodes)
	fmt.Printf("  entities : %d\n", entities)
	fmt.Printf("  edges    : %d\n", edges)
	fmt.Printf("  secure   : %d\n", secureEpisodes)
	if secureMode {
		fmt.Printf("  mode     : kpt\n")
	} else {
		fmt.Printf("  mode     : graph\n")
	}

	curvTotal, curvBridges, curvInternal, curvMean := db.CurvatureStatus(ctx, groupID)
	if curvTotal > 0 {
		fmt.Printf("\n── Curvature ──────────────\n")
		fmt.Printf("  edges    : %d\n", curvTotal)
		fmt.Printf("  bridges  : %d (%.0f%%)\n", curvBridges, pct(curvBridges, curvTotal))
		fmt.Printf("  internal : %d (%.0f%%)\n", curvInternal, pct(curvInternal, curvTotal))
		fmt.Printf("  mean κ   : %.3f\n", curvMean)
	}

	fmt.Printf("\n── Queue ──────────────────\n")
	for _, s := range []string{"pending", "processing", "done", "failed"} {
		fmt.Printf("  %-10s : %d\n", s, stats[s])
	}
}

func printCommunities(ctx context.Context, db *store.DB, groupID, format string, minMembers int) {
	communities, err := db.ListCommunities(ctx, groupID)
	must(err, "list communities")
	printCommunitySummaries(communities, format, minMembers)
}

func printCommunitySummaries(communities []store.CommunitySummary, format string, minMembers int) {
	// Filter by minimum member count.
	filtered := communities[:0]
	for _, c := range communities {
		if len(c.Members) < minMembers {
			continue
		}
		filtered = append(filtered, c)
	}

	if format == "json" {
		enc := json.NewEncoder(os.Stdout)
		for _, c := range filtered {
			_ = enc.Encode(c)
		}
		return
	}

	if len(filtered) == 0 {
		fmt.Printf("(no communities with >= %d members)\n", minMembers)
		return
	}

	fmt.Printf("── Communities (%d shown, %d total, min %d members) ────────\n\n",
		len(filtered), len(communities), minMembers)
	for _, c := range filtered {
		fmt.Printf("  Community %d  (%d members)\n", c.CommunityID, len(c.Members))
		fmt.Printf("    Members: %s\n", strings.Join(c.Members, ", "))
		if c.Report != "" {
			fmt.Printf("    Report:  %s\n", c.Report)
		}
		fmt.Println()
	}
}

func printResolveResult(r store.ResolveResult, dryRun bool) {
	mode := ""
	if dryRun {
		mode = " (dry-run)"
	}
	fmt.Printf("Entity resolution%s:\n", mode)
	fmt.Printf("  clusters found    : %d\n", r.ClustersFound)
	fmt.Printf("  entities merged   : %d\n", r.EntitiesMerged)
	fmt.Printf("  edges retargeted  : %d\n", r.EdgesRetargeted)
	fmt.Printf("  episodes relinked : %d\n", r.EpisodesRelinked)
}

func printSecureResolveResult(r store.SecureResolveResult, dryRun bool) {
	mode := ""
	if dryRun {
		mode = " (dry-run)"
	}
	fmt.Printf("Secure episode resolution%s:\n", mode)
	fmt.Printf("  clusters found  : %d\n", r.ClustersFound)
	fmt.Printf("  episodes merged : %d\n", r.EpisodesMerged)
}

func usage() {
	fmt.Fprintf(os.Stderr, "ultramemory %s — local knowledge graph (%s build)\n\n", version, buildProviderName)
	fmt.Fprintf(os.Stderr, `Commands:
  run     <path>   ingest directory + start worker (all-in-one)  [-source URL]
  ingest  <path>   queue all text files for processing         [-source URL]
  worker           process queued jobs (blocking)
  search  <query>  graph search, or KPT search when MEMORY_PERSONAL_KEY is set
  personal-index   build keyed KPT episode index from existing embeddings     (-key, -method)
  personal-search  keyed KPT search over episodes                             (-key, -method, -limit, -format)
  personal-cluster keyed KPT clustering over episode semantics                (-key, -method, -k, -min-score)
  retry            requeue all failed jobs for reprocessing
  resolve          merge near-duplicates; entities in graph mode, episodes in KPT mode
  communities      graph communities, or KPT communities when MEMORY_PERSONAL_KEY is set
  curvature        compute ORC on graph edges, or KPT episode graph in secure mode
  bench   <json>   evaluate against LoCoMo benchmark (flags: -limit N, -baseline)
  status           show queue and graph statistics

Environment:
  MEMORY_DB                  path to SQLite file          (default: memory-local.db)
  MEMORY_GROUP               namespace/group              (default: default)
  MEMORY_PERSONAL_KEY        enables KPT-first ingest/search/clustering and encrypts queue + episode payloads at rest
  MEMORY_RESOLVE_THRESHOLD   resolve similarity           (default: 0.92)
  MEMORY_LLM_PARALLEL        concurrent LLM calls         (default: %d)
%s`, defaultLLMParallel, buildEnvironmentHelp)
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func secureKeyOrEnv(explicit string) string {
	if explicit != "" {
		return explicit
	}
	return os.Getenv("MEMORY_PERSONAL_KEY")
}

func mustNewMistralClient(model, usage string) *llm.MistralClient {
	apiKey := os.Getenv("MISTRAL_API_KEY")
	if apiKey == "" {
		fatalf("MISTRAL_API_KEY not set — required for %s", usage)
	}
	return llm.NewMistral(apiKey, model)
}

func must(err error, msg string) {
	if err != nil {
		fatalf("%s: %v", msg, err)
	}
}

func runSecureSearch(ctx context.Context, db *store.DB, embedder llm.Embedder, personalKey, groupID, methodName, query, format string, maxTokens int) {
	rows, err := db.AllSecureEpisodes(ctx, groupID, methodName)
	must(err, "load secure index")
	if len(rows) == 0 {
		fatalf("no secure episode index found for group %q and method %q", groupID, methodName)
	}

	queryEmb, err := embedder.Embed(ctx, query)
	must(err, "embed query")
	rows, err = decryptSecureRows(rows, personalKey)
	must(err, "decrypt secure rows")

	method := secureindex.NewMethod(personalKey, len(queryEmb))
	queryState := method.EncodeQuery(queryEmb)
	hits := method.Search(secureStates(rows), queryState, 10)
	printPersonalSearch(rows, hits, query, format, maxTokens)
}

func decryptSecureRows(rows []store.SecureEpisodeIndexRow, personalKey string) ([]store.SecureEpisodeIndexRow, error) {
	out := make([]store.SecureEpisodeIndexRow, 0, len(rows))
	for _, row := range rows {
		content, err := secureindex.DecryptString(personalKey, "episode-content", row.Content)
		if err != nil {
			row.Content = "<entschlüsselung fehlgeschlagen>"
		} else {
			row.Content = content
		}
		source, err := secureindex.DecryptString(personalKey, "episode-source", row.Source)
		if err != nil {
			row.Source = "<locked>"
		} else {
			row.Source = source
		}
		out = append(out, row)
	}
	return out, nil
}

func secureStates(rows []store.SecureEpisodeIndexRow) []secureindex.State {
	states := make([]secureindex.State, 0, len(rows))
	for _, row := range rows {
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
	return states
}

func truncateLine(value string, max int) string {
	value = strings.TrimSpace(strings.ReplaceAll(value, "\n", " "))
	if len(value) <= max {
		return value
	}
	if max <= 3 {
		return value[:max]
	}
	return value[:max-3] + "..."
}

func recoverStaleJobs(ctx context.Context, db *store.DB) {
	n, err := db.RecoverStaleJobs(ctx, staleJobTimeout)
	if err != nil {
		slog.Error("recover stale jobs", "err", err)
	}
	if n > 0 {
		slog.Info("recovered stale jobs", "count", n)
	}
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "error: "+format+"\n", args...)
	os.Exit(1)
}

func printCurvatureStats(stats store.CurvatureStats, format string) {
	if format == "json" {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		enc.Encode(stats) //nolint:errcheck
		return
	}
	fmt.Printf("\nOllivier-Ricci Curvature Statistics\n")
	fmt.Printf("───────────────────────────────────\n")
	fmt.Printf("  Total edges:   %d\n", stats.TotalEdges)
	fmt.Printf("  Bridges (κ<0): %d (%.1f%%)\n", stats.Bridges, pct(stats.Bridges, stats.TotalEdges))
	fmt.Printf("  Internal (κ>0):%d (%.1f%%)\n", stats.Internal, pct(stats.Internal, stats.TotalEdges))
	fmt.Printf("  Flat (|κ|<.05):%d (%.1f%%)\n", stats.Flat, pct(stats.Flat, stats.TotalEdges))
	fmt.Printf("  Mean κ:        %.4f\n", stats.Mean)
	fmt.Printf("  Min κ:         %.4f\n", stats.Min)
	fmt.Printf("  Max κ:         %.4f\n", stats.Max)
	fmt.Printf("  Elapsed:       %s\n", stats.Elapsed)
}

func printBridges(bridges []store.EdgeCurvature, format string) {
	if format == "json" {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		enc.Encode(bridges) //nolint:errcheck
		return
	}
	fmt.Printf("\nTop Bridge Edges (most negative curvature)\n")
	fmt.Printf("───────────────────────────────────────────\n")
	for i, b := range bridges {
		src := b.SourceName
		if src == "" {
			src = b.SourceUUID[:min(8, len(b.SourceUUID))]
		}
		tgt := b.TargetName
		if tgt == "" {
			tgt = b.TargetUUID[:min(8, len(b.TargetUUID))]
		}
		fmt.Printf("  %3d. κ=%+.4f  %s ↔ %s\n", i+1, b.Curvature, src, tgt)
	}
}

func pct(n, total int) float64 {
	if total == 0 {
		return 0
	}
	return float64(n) / float64(total) * 100
}

// rejectTrailingFlags detects flags placed after the positional path argument
// (e.g. "ingest ./path -source URL") which Go's flag package silently ignores.
func rejectTrailingFlags(fs *flag.FlagSet) {
	for _, arg := range fs.Args()[1:] {
		if strings.HasPrefix(arg, "-") {
			fatalf("flags must come before the path argument: %q\nusage: ultramemory %s [-flags] <path>", arg, fs.Name())
		}
	}
}
