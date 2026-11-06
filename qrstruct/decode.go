package qrstruct

import (
	"errors"
	"fmt"
	"image"
	"math"
	"runtime"
	"sort"
	"sync"
)

// ErrNoDecode is returned when no hypothesis yields a valid symbol.
var ErrNoDecode = errors.New("qrstruct: no decodable symbol found")

// Options tunes the decoder. Zero values select the defaults.
type Options struct {
	// MinVersion and MaxVersion bound the version hypotheses (default 1..10).
	MinVersion int
	MaxVersion int

	// MinModulePx skips versions whose modules would be smaller than this
	// many pixels given the quad size (default 2).
	MinModulePx float64

	// TryMirror also reads each fitted grid through its diagonal, which
	// decodes mirrored symbols (default true).
	TryMirror bool

	// TrySurface enables the cylinder stage when the planar fit fails
	// (default true).
	TrySurface bool

	// MaxCandidates is the number of best-ranked hypotheses refined and
	// decoded with a planar grid (default 3).
	MaxCandidates int

	// MaxSurfaceCandidates is the number of best-ranked hypotheses passed to
	// the cylinder stage (default 4).
	MaxSurfaceCandidates int

	// Workers bounds the goroutines used by the searches
	// (default runtime.NumCPU()).
	Workers int
}

// DefaultOptions returns the recommended settings.
func DefaultOptions() Options {
	return Options{
		MinVersion:           1,
		MaxVersion:           10,
		MinModulePx:          2,
		TryMirror:            true,
		TrySurface:           true,
		MaxCandidates:        3,
		MaxSurfaceCandidates: 4,
		Workers:              runtime.NumCPU(),
	}
}

func (o Options) normalise() Options {
	d := DefaultOptions()
	if o.MinVersion < MinVersion {
		o.MinVersion = d.MinVersion
	}
	if o.MaxVersion < o.MinVersion || o.MaxVersion > MaxVersion {
		o.MaxVersion = max(d.MaxVersion, o.MinVersion)
	}
	if o.MinModulePx <= 0 {
		o.MinModulePx = d.MinModulePx
	}
	if o.MaxCandidates <= 0 {
		o.MaxCandidates = d.MaxCandidates
	}
	if o.MaxSurfaceCandidates <= 0 {
		o.MaxSurfaceCandidates = d.MaxSurfaceCandidates
	}
	if o.Workers <= 0 {
		o.Workers = d.Workers
	}
	return o
}

// Result describes a decoded symbol.
type Result struct {
	// Content is the payload converted to a Go string (UTF-8).
	Content string
	// Payload holds the raw decoded bytes.
	Payload []byte

	Version int
	Level   Level
	Mask    int
	ECI     int

	// Corners are the fitted lattice corners in image pixels (TL, TR, BR, BL
	// of the symbol).
	Corners Quad
	// Surface is the cylinder model used, zero for a planar fit.
	Surface Surface
	// Mirrored reports whether the symbol was read through its diagonal.
	Mirrored bool
	// Rotation is the number of corner-order shifts applied to the input quad.
	Rotation int

	// Errors and Erasures count the codewords corrected by Reed-Solomon.
	Errors   int
	Erasures int
	// Score is the final fit objective (known-module mismatches plus format
	// distance, plus the ambiguity term for a cylinder fit).
	Score float64
	// Stage names the fit that succeeded: "homography" or "cylinder".
	Stage string
}

type hypothesis struct {
	version  int
	rotation int
	quad     Quad
	score    float64
	norm     float64
}

// Decode fits a sampling grid inside quad and decodes the symbol. The quad
// corners are TL, TR, BR, BL in image order; the symbol may be rotated or
// mirrored with respect to that order.
func Decode(img *image.Gray, quad Quad, opts Options) (*Result, error) {
	if img == nil || img.Rect.Dx() < 2 || img.Rect.Dy() < 2 {
		return nil, errors.New("qrstruct: image too small")
	}
	opts = opts.normalise()
	side := sideLength(quad)
	if side < 8 {
		return nil, fmt.Errorf("qrstruct: quad side %.1f px too small", side)
	}

	cands := identify(img, quad, side, opts)
	if len(cands) == 0 {
		return nil, ErrNoDecode
	}

	// Planar stage.
	for i, h := range cands {
		if i >= opts.MaxCandidates {
			break
		}
		ev := newEvaluator(img, h.version, 3)
		q, score := ev.descent(h.quad, Surface{}, fineSteps, planarLambda)
		if r, ok := ev.decodeBoth(q, Surface{}, opts.TryMirror); ok {
			r.fill(h, score, "homography")
			return r, nil
		}
	}
	if !opts.TrySurface {
		return nil, ErrNoDecode
	}

	// Cylinder stage.
	for i, h := range cands {
		if i >= opts.MaxSurfaceCandidates {
			break
		}
		if r, ok := surfaceStage(img, h, opts); ok {
			return r, nil
		}
	}
	return nil, ErrNoDecode
}

func (r *Result) fill(h hypothesis, score float64, stage string) {
	r.Version = h.version
	r.Rotation = h.rotation
	r.Score = score
	r.Stage = stage
}

// decodeBoth tries the lattice as fitted and, optionally, mirrored.
func (e *evaluator) decodeBoth(q Quad, surf Surface, tryMirror bool) (*Result, bool) {
	if r, err := e.decode(q, surf); err == nil {
		return r, true
	}
	if tryMirror {
		if r, err := e.decode(q.mirror(), surf); err == nil {
			r.Mirrored = true
			return r, true
		}
	}
	return nil, false
}

// parallelFor runs fn(i) for i in [0, n) on a bounded worker pool.
func parallelFor(n, workers int, fn func(i int)) {
	var wg sync.WaitGroup
	sem := make(chan struct{}, max(1, workers))
	for i := 0; i < n; i++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int) {
			defer wg.Done()
			defer func() { <-sem }()
			fn(i)
		}(i)
	}
	wg.Wait()
}

// identify ranks (version, orientation) hypotheses by their normalised
// known-module score after a coarse single-sample fit.
func identify(img *image.Gray, quad Quad, side float64, opts Options) []hypothesis {
	var hyps []hypothesis
	for v := opts.MinVersion; v <= opts.MaxVersion; v++ {
		if side/float64(SymbolSize(v)) < opts.MinModulePx {
			break
		}
		for rot := 0; rot < 4; rot++ {
			hyps = append(hyps, hypothesis{version: v, rotation: rot, quad: quad.rotate(rot)})
		}
	}
	if len(hyps) == 0 {
		return nil
	}
	parallelFor(len(hyps), opts.Workers, func(i int) {
		h := &hyps[i]
		ev := newEvaluator(img, h.version, 1)
		q, _ := ev.globalSearch(h.quad, planarLambda)
		q, s := ev.descent(q, Surface{}, coarseSteps, planarLambda)
		h.quad, h.score, h.norm = q, s, ev.normalised(s)
	})
	sort.SliceStable(hyps, func(i, j int) bool { return hyps[i].norm < hyps[j].norm })
	return hyps
}

// Cylinder parameter grid for the surface stage.
var (
	surfaceThetasDeg = []float64{0, 30, 60, 90, 120, 150}
	surfaceBends     = []float64{0.3, 0.5, 0.7}
	surfacePhases    = []float64{-0.5, 0, 0.5}
	surfaceLambda    = 60.0
	surfaceRefine    = 6
)

// surfaceStage sweeps the cylinder parameters for one hypothesis: a cheap
// single-sample descent ranks every parameter set, then the best few are
// refined with full sampling and decoded.
func surfaceStage(img *image.Gray, h hypothesis, opts Options) (*Result, bool) {
	type trial struct {
		surf  Surface
		quad  Quad
		score float64
	}
	var trials []trial
	for _, th := range surfaceThetasDeg {
		for _, b := range surfaceBends {
			for _, p := range surfacePhases {
				trials = append(trials, trial{surf: Surface{Bend: b, Phase: p, Theta: th * math.Pi / 180}})
			}
		}
	}
	parallelFor(len(trials), opts.Workers, func(i int) {
		ev := newEvaluator(img, h.version, 1)
		q, s := ev.descent(h.quad, trials[i].surf, fineSteps[:2], surfaceLambda)
		trials[i].quad, trials[i].score = q, s
	})
	sort.SliceStable(trials, func(i, j int) bool { return trials[i].score < trials[j].score })

	ev := newEvaluator(img, h.version, 3)
	for i, tr := range trials {
		if i >= surfaceRefine {
			break
		}
		q, s := ev.descent(tr.quad, tr.surf, fineSteps[1:], surfaceLambda)
		if r, ok := ev.decodeBoth(q, tr.surf, opts.TryMirror); ok {
			r.fill(h, s, "cylinder")
			return r, true
		}
	}
	return nil, false
}

// decode samples the grid, reads the format, de-interleaves and corrects the
// codewords and parses the payload.
func (e *evaluator) decode(q Quad, surf Surface) (*Result, error) {
	l := e.layout
	vals := e.sampler.values(q, surf)
	t := binarize(vals, e.bits)
	level, mask, _ := l.readFormat(e.bits)
	cw := l.extractCodewords(e.bits, mask)

	// Per-codeword reliability: the least confident of its eight modules.
	n := l.Size
	rel := make([]float32, len(cw))
	for i := range cw {
		m := float32(math.MaxFloat32)
		for b := 0; b < 8; b++ {
			p := l.dataPos[i*8+b]
			d := vals[p[0]*n+p[1]] - t
			if d < 0 {
				d = -d
			}
			if d < m {
				m = d
			}
		}
		rel[i] = m
	}

	blocks := BlockLayout(l.Version, level)
	index := deinterleaveIndex(blocks)
	data, nerr, nerase, err := correctBlocks(cw, rel, blocks, index)
	if err != nil {
		return nil, err
	}
	payload, eci, endBit, err := parseSegments(data, l.Version)
	if err != nil {
		return nil, fmt.Errorf("parsing segments: %w", err)
	}
	if nerase > 0 && !validPadding(data, endBit) {
		return nil, errors.New("erasure decode rejected: padding bytes do not match")
	}
	return &Result{
		Content:  payloadToString(payload, eci),
		Payload:  payload,
		Version:  l.Version,
		Level:    level,
		Mask:     mask,
		ECI:      eci,
		Corners:  q,
		Surface:  surf,
		Errors:   nerr,
		Erasures: nerase,
	}, nil
}

// correctBlocks de-interleaves the codewords and runs Reed-Solomon on every
// block, erasing progressively more of the least reliable codewords until the
// block decodes. It returns the concatenated data codewords.
func correctBlocks(cw []byte, rel []float32, blocks []Block, index [][]int) ([]byte, int, int, error) {
	last := index[len(index)-1]
	if last[len(last)-1] >= len(cw) {
		return nil, 0, 0, errors.New("codeword stream shorter than the block layout")
	}
	var data []byte
	totalErr, totalErase := 0, 0
	for b, blk := range blocks {
		src := index[b]
		msg := make([]byte, len(src))
		order := make([]int, len(src))
		for k, p := range src {
			msg[k] = cw[p]
			order[k] = k
		}
		if rel != nil {
			sort.SliceStable(order, func(i, j int) bool { return rel[src[order[i]]] < rel[src[order[j]]] })
		}
		// Erasures are capped at half the EC codewords so that at least half
		// of the redundancy remains to validate the correction: with EC
		// erasures the decoder would always "succeed" by interpolation.
		fracs := []float64{0, 0.25, 0.5}
		if rel == nil {
			fracs = fracs[:1]
		}
		var ok bool
		for _, frac := range fracs {
			k := int(math.Round(frac * float64(blk.EC)))
			work := make([]byte, len(msg))
			copy(work, msg)
			nerr, err := RSDecode(work, blk.EC, order[:k])
			if err != nil {
				continue
			}
			data = append(data, work[:blk.Data]...)
			totalErr += nerr
			totalErase += k
			ok = true
			break
		}
		if !ok {
			return nil, 0, 0, fmt.Errorf("block %d: reed-solomon failed", b)
		}
	}
	return data, totalErr, totalErase, nil
}

// DecodeMatrix decodes a perfectly sampled module matrix (1 = dark), as
// produced by an encoder. It is mostly useful for tests and tooling.
func DecodeMatrix(bits []uint8, version int) (*Result, error) {
	if version < MinVersion || version > MaxVersion {
		return nil, fmt.Errorf("version %d out of range", version)
	}
	l := LayoutFor(version)
	if len(bits) != l.Size*l.Size {
		return nil, fmt.Errorf("matrix has %d modules, version %d needs %d", len(bits), version, l.Size*l.Size)
	}
	level, mask, _ := l.readFormat(bits)
	cw := l.extractCodewords(bits, mask)
	blocks := BlockLayout(version, level)
	data, nerr, _, err := correctBlocks(cw, nil, blocks, deinterleaveIndex(blocks))
	if err != nil {
		return nil, err
	}
	payload, eci, _, err := parseSegments(data, version)
	if err != nil {
		return nil, err
	}
	return &Result{Content: payloadToString(payload, eci), Payload: payload, Version: version, Level: level, Mask: mask, ECI: eci, Errors: nerr}, nil
}
