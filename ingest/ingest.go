// Package ingest walks a directory and enqueues text chunks for graph extraction.
package ingest

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/sharpner/ultramemory/graph"
	"github.com/sharpner/ultramemory/llm"
	"github.com/sharpner/ultramemory/secureindex"
	"github.com/sharpner/ultramemory/store"
)

const (
	chunkSize    = 1500
	chunkOverlap = 150
)

// Walker ingests files into the job queue.
type Walker struct {
	db             *store.DB
	groupID        string
	personalKey    string
	sourceOverride string  // if set, used as source instead of file path
	pdftotextBin   string  // optional: poppler pdftotext
	pdftoppmBin    string  // optional: poppler pdftoppm (needed for OCR fallback)
	tesseractBin   string  // optional: Tesseract OCR
	detexBin       string  // optional: opendetex for LaTeX stripping
	ocrClient      llm.OCR // optional: model OCR fallback
}

// New creates a Walker, detecting available PDF tools on PATH.
func New(db *store.DB, groupID string) *Walker {
	w := &Walker{db: db, groupID: groupID}
	w.pdftotextBin, _ = exec.LookPath("pdftotext")
	w.pdftoppmBin, _ = exec.LookPath("pdftoppm")
	w.tesseractBin, _ = exec.LookPath("tesseract")
	w.detexBin, _ = exec.LookPath("detex")

	if w.pdftotextBin == "" {
		slog.Warn("pdftotext not found — digital PDFs will be skipped (install poppler)")
	}
	if w.tesseractBin == "" {
		slog.Warn("tesseract not found — scanned PDFs will fall back to model OCR when configured")
	}
	return w
}

// WithSource sets a source override (e.g. an arXiv URL) used instead of the file path.
func (w *Walker) WithSource(source string) *Walker {
	w.sourceOverride = source
	return w
}

// WithOCR attaches a model OCR client as last-resort fallback for scanned PDFs.
// Only used when both pdftotext and tesseract produce no output.
func (w *Walker) WithOCR(client llm.OCR) *Walker {
	w.ocrClient = client
	return w
}

// WithPersonalKey enables encrypted queue payloads for KPT-first ingestion.
func (w *Walker) WithPersonalKey(personalKey string) *Walker {
	w.personalKey = personalKey
	return w
}

// Walk recursively ingests all readable text files under root.
func (w *Walker) Walk(ctx context.Context, root string) (int, error) {
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return 0, fmt.Errorf("abs root: %w", err)
	}

	total := 0
	err = filepath.WalkDir(absRoot, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			slog.Warn("skip unreadable path", "path", path, "err", err)
			return nil
		}
		if d.IsDir() {
			if path == absRoot {
				return nil
			}
			return skipDir(d.Name())
		}
		if isPDF(path) {
			text, err := w.extractPDF(ctx, path)
			if err != nil {
				slog.Warn("skip pdf", "path", path, "err", err)
				return nil
			}
			return w.enqueueChunks(ctx, text, path, &total)
		}
		if isTeX(path) {
			text, err := w.sanitizeTeX(ctx, path)
			if err != nil {
				slog.Warn("skip tex", "path", path, "err", err)
				return nil
			}
			return w.enqueueChunks(ctx, text, path, &total)
		}
		if !isText(path) {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			slog.Warn("skip unreadable file", "path", path, "err", err)
			return nil
		}
		if !utf8.Valid(data) {
			return nil
		}
		return w.enqueueChunks(ctx, string(data), path, &total)
	})
	return total, err
}

// extractPDF tries pdftotext → tesseract → model OCR in order.
func (w *Walker) extractPDF(ctx context.Context, path string) (string, error) {
	// 1. pdftotext
	if w.pdftotextBin != "" {
		text, err := extractPDFText(ctx, w.pdftotextBin, path)
		if err == nil && text != "" {
			return text, nil
		}
	}

	// Scanned PDF — need image-based OCR.
	if w.pdftoppmBin == "" {
		return "", fmt.Errorf("scanned PDF requires pdftoppm (install poppler)")
	}

	// 2. Tesseract
	if w.tesseractBin != "" {
		text, err := w.ocrPDFTesseract(ctx, path)
		if err == nil && text != "" {
			return text, nil
		}
		slog.Warn("tesseract failed", "path", path, "err", err)
	}

	// 3. model OCR fallback
	if w.ocrClient != nil {
		slog.Warn("using model OCR fallback — accuracy is lower than Tesseract; install tesseract for better results",
			"path", path)
		return w.ocrPDFModel(ctx, path)
	}

	return "", fmt.Errorf("scanned PDF: no OCR available (install tesseract)")
}

func (w *Walker) ocrPDFTesseract(ctx context.Context, pdfPath string) (string, error) {
	pages, cleanup, err := w.pdfToImages(ctx, pdfPath)
	if err != nil {
		return "", err
	}
	defer cleanup()

	var texts []string
	for _, page := range pages {
		out, err := w.runTesseract(ctx, page)
		if err != nil {
			slog.Warn("tesseract page failed", "page", page, "err", err)
			continue
		}
		if out != "" {
			texts = append(texts, out)
		}
	}
	if len(texts) == 0 {
		return "", fmt.Errorf("tesseract produced no output")
	}
	return strings.Join(texts, "\n\n"), nil
}

func (w *Walker) ocrPDFModel(ctx context.Context, pdfPath string) (string, error) {
	pages, cleanup, err := w.pdfToImages(ctx, pdfPath)
	if err != nil {
		return "", err
	}
	defer cleanup()

	var texts []string
	for _, page := range pages {
		data, err := os.ReadFile(page)
		if err != nil {
			slog.Warn("read page image failed", "page", page, "err", err)
			continue
		}
		out, err := w.ocrClient.OCR(ctx, data)
		if err != nil {
			slog.Warn("model OCR page failed", "page", page, "err", err)
			continue
		}
		if out != "" {
			texts = append(texts, out)
		}
	}
	if len(texts) == 0 {
		return "", fmt.Errorf("model OCR produced no output")
	}
	return strings.Join(texts, "\n\n"), nil
}

func (w *Walker) pdfToImages(ctx context.Context, pdfPath string) ([]string, func(), error) {
	tmpDir, err := os.MkdirTemp("", "ultramemory-ocr-*")
	if err != nil {
		return nil, func() {}, fmt.Errorf("temp dir: %w", err)
	}
	cleanup := func() { _ = os.RemoveAll(tmpDir) }

	prefix := filepath.Join(tmpDir, "page")
	if _, err := exec.CommandContext(ctx, w.pdftoppmBin,
		"-r", "200", "-jpeg", pdfPath, prefix,
	).Output(); err != nil {
		cleanup()
		return nil, func() {}, fmt.Errorf("pdftoppm: %w", err)
	}

	pages, err := filepath.Glob(prefix + "-*.jpg")
	if err != nil || len(pages) == 0 {
		cleanup()
		return nil, func() {}, fmt.Errorf("pdftoppm produced no pages")
	}
	sort.Strings(pages)
	return pages, cleanup, nil
}

// runTesseract must run from the image's directory — tesseract rejects absolute paths.
func (w *Walker) runTesseract(ctx context.Context, imagePath string) (string, error) {
	cmd := exec.CommandContext(ctx, w.tesseractBin, filepath.Base(imagePath), "stdout")
	cmd.Dir = filepath.Dir(imagePath)
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

func (w *Walker) enqueueChunks(ctx context.Context, text, source string, total *int) error {
	if w.sourceOverride != "" {
		source = w.sourceOverride
	}
	for _, c := range chunk(text, chunkSize, chunkOverlap) {
		c = strings.TrimSpace(c)
		if len(c) < 50 {
			continue
		}
		p := graph.IngestPayload{
			Content: c,
			Source:  source,
			GroupID: w.groupID,
		}
		if w.personalKey != "" {
			contentCiphertext, err := secureindex.EncryptString(w.personalKey, "ingest-content", c)
			if err != nil {
				return fmt.Errorf("encrypt content payload: %w", err)
			}
			sourceCiphertext, err := secureindex.EncryptString(w.personalKey, "ingest-source", source)
			if err != nil {
				return fmt.Errorf("encrypt source payload: %w", err)
			}
			p.Content = ""
			p.Source = ""
			p.ContentCiphertext = contentCiphertext
			p.SourceCiphertext = sourceCiphertext
		}
		payload, err := json.Marshal(p)
		if err != nil {
			return fmt.Errorf("marshal payload: %w", err)
		}
		if err := w.db.PushJob(ctx, store.JobTypeIngest, string(payload)); err != nil {
			return fmt.Errorf("push job: %w", err)
		}
		*total++
	}
	return nil
}

func skipDir(name string) error {
	skip := map[string]bool{
		".git": true, "node_modules": true, "vendor": true,
		".next": true, "dist": true, "build": true, "__pycache__": true,
		".cache": true, "coverage": true, "testdata": true,
	}
	if skip[name] || strings.HasPrefix(name, ".") {
		return filepath.SkipDir
	}
	return nil
}

func isText(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	ok := map[string]bool{
		".go": true, ".md": true, ".txt": true, ".yaml": true,
		".yml": true, ".json": true, ".toml": true, ".sh": true,
		".ts": true, ".tsx": true, ".js": true, ".jsx": true,
		".py": true, ".rs": true, ".sql": true, ".env": true,
		".proto": true, ".graphql": true, ".html": true, ".css": true,
		".bib": true,
	}
	return ok[ext]
}

func isPDF(path string) bool {
	return strings.ToLower(filepath.Ext(path)) == ".pdf"
}

func isTeX(path string) bool {
	return strings.ToLower(filepath.Ext(path)) == ".tex"
}

func extractPDFText(ctx context.Context, bin, path string) (string, error) {
	out, err := exec.CommandContext(ctx, bin, path, "-").Output()
	if err != nil {
		return "", fmt.Errorf("pdftotext: %w", err)
	}
	return strings.TrimSpace(string(out)), nil
}

func chunk(text string, size, overlap int) []string {
	runes := []rune(text)
	if len(runes) <= size {
		return []string{string(runes)}
	}

	var chunks []string
	step := size - overlap
	if step <= 0 {
		step = size
	}
	for i := 0; i < len(runes); i += step {
		end := i + size
		if end > len(runes) {
			end = len(runes)
		}
		chunks = append(chunks, string(runes[i:end]))
		if end == len(runes) {
			break
		}
	}
	return chunks
}
