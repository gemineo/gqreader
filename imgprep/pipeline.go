package imgprep

import (
	"context"
	"fmt"
	"sync"
	"time"

	"gocv.io/x/gocv"
)

// DetectFunc is the signature for barcode detection functions.
// It takes a preprocessed image and returns detected barcodes.
// The generic type T allows any barcode result type.
type DetectFunc[T any] func(img gocv.Mat) ([]T, error)

// StrategyResult holds the results from a single preprocessing strategy.
type StrategyResult[T any] struct {
	// Strategy is the name of the preprocessing strategy used.
	Strategy string

	// Variant is the specific variant name (e.g., rotation angle).
	// Empty for strategies that don't generate variants.
	Variant string

	// Detections contains all barcodes found with this strategy.
	Detections []T

	// Duration is the time taken for preprocessing + detection.
	Duration time.Duration

	// Error is non-nil if the strategy failed.
	Error error
}

// FullName returns the complete strategy identifier including variant.
func (r *StrategyResult[T]) FullName() string {
	if r.Variant == "" {
		return r.Strategy
	}
	return r.Strategy + "/" + r.Variant
}

// Pipeline orchestrates parallel execution of preprocessing strategies.
type Pipeline[T any] struct {
	strategies []Strategy
	maxWorkers int
}

// PipelineOption configures a Pipeline.
type PipelineOption[T any] func(*Pipeline[T])

// WithStrategies adds preprocessing strategies to the pipeline.
func WithStrategies[T any](strategies ...Strategy) PipelineOption[T] {
	return func(p *Pipeline[T]) {
		p.strategies = append(p.strategies, strategies...)
	}
}

// WithMaxWorkers sets the maximum number of concurrent workers.
// Default is 0 (unlimited - one goroutine per strategy).
func WithMaxWorkers[T any](n int) PipelineOption[T] {
	return func(p *Pipeline[T]) {
		p.maxWorkers = n
	}
}

// NewPipeline creates a new preprocessing pipeline.
func NewPipeline[T any](opts ...PipelineOption[T]) *Pipeline[T] {
	p := &Pipeline[T]{
		strategies: make([]Strategy, 0),
		maxWorkers: 0,
	}
	for _, opt := range opts {
		opt(p)
	}
	return p
}

// Process runs all preprocessing strategies in parallel and aggregates results.
// Each strategy preprocesses the input image, then runs detection on the result.
// All results are collected, even from strategies that find no barcodes.
//
// The input image is NOT modified. Each strategy receives a read-only view.
// The context can be used to cancel processing early.
func (p *Pipeline[T]) Process(ctx context.Context, src gocv.Mat, detect DetectFunc[T]) ([]StrategyResult[T], error) {
	if src.Empty() {
		return nil, fmt.Errorf("input image is empty")
	}

	if len(p.strategies) == 0 {
		return nil, fmt.Errorf("no strategies configured")
	}

	// Expand variant strategies into individual tasks
	type task struct {
		strategy Strategy
		variant  string // Empty for non-variant strategies
		variantFn func() gocv.Mat // Returns preprocessed image for variant
	}

	var tasks []task
	for _, s := range p.strategies {
		if _, ok := s.(VariantGenerator); ok {
			// Capture variants - we need to generate them during task execution
			tasks = append(tasks, task{
				strategy: s,
				variant:  "__variant__", // Marker for variant expansion
			})
		} else {
			tasks = append(tasks, task{strategy: s})
		}
	}

	// Result channel
	results := make(chan StrategyResult[T], len(tasks)*10) // Buffer for variants

	// Worker pool
	var wg sync.WaitGroup
	semaphore := make(chan struct{}, p.workerLimit(len(tasks)))

	for _, t := range tasks {
		select {
		case <-ctx.Done():
			// Context cancelled, stop launching new tasks
			break
		default:
		}

		wg.Add(1)
		go func(t task) {
			defer wg.Done()

			semaphore <- struct{}{}        // Acquire
			defer func() { <-semaphore }() // Release

			if t.variant == "__variant__" {
				// Expand and process variants
				p.processVariants(ctx, src, t.strategy.(VariantGenerator), detect, results)
			} else {
				// Process single strategy
				result := p.processStrategy(ctx, src, t.strategy, detect)
				select {
				case results <- result:
				case <-ctx.Done():
				}
			}
		}(t)
	}

	// Close results channel when all workers complete
	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results
	var allResults []StrategyResult[T]
	for result := range results {
		allResults = append(allResults, result)
	}

	return allResults, nil
}

// processStrategy handles a single preprocessing strategy.
func (p *Pipeline[T]) processStrategy(ctx context.Context, src gocv.Mat, strategy Strategy, detect DetectFunc[T]) StrategyResult[T] {
	start := time.Now()
	result := StrategyResult[T]{
		Strategy: strategy.Name(),
	}

	// Check context
	select {
	case <-ctx.Done():
		result.Error = ctx.Err()
		result.Duration = time.Since(start)
		return result
	default:
	}

	// Apply preprocessing
	processed, err := strategy.Apply(src)
	if err != nil {
		result.Error = fmt.Errorf("preprocessing failed: %w", err)
		result.Duration = time.Since(start)
		return result
	}
	defer processed.Close()

	// Check context again
	select {
	case <-ctx.Done():
		result.Error = ctx.Err()
		result.Duration = time.Since(start)
		return result
	default:
	}

	// Run detection
	detections, err := detect(processed)
	if err != nil {
		result.Error = fmt.Errorf("detection failed: %w", err)
		result.Duration = time.Since(start)
		return result
	}

	result.Detections = detections
	result.Duration = time.Since(start)
	return result
}

// processVariants handles variant-generating strategies.
func (p *Pipeline[T]) processVariants(ctx context.Context, src gocv.Mat, vg VariantGenerator, detect DetectFunc[T], results chan<- StrategyResult[T]) {
	for variantName, processed := range vg.Variants(src) {
		select {
		case <-ctx.Done():
			processed.Close()
			return
		default:
		}

		start := time.Now()
		result := StrategyResult[T]{
			Strategy: vg.Name(),
			Variant:  variantName,
		}

		detections, err := detect(processed)
		processed.Close()

		if err != nil {
			result.Error = fmt.Errorf("detection failed: %w", err)
		} else {
			result.Detections = detections
		}
		result.Duration = time.Since(start)

		select {
		case results <- result:
		case <-ctx.Done():
			return
		}
	}
}

// workerLimit returns the effective worker limit.
func (p *Pipeline[T]) workerLimit(taskCount int) int {
	if p.maxWorkers <= 0 {
		return taskCount // Unlimited
	}
	if p.maxWorkers > taskCount {
		return taskCount
	}
	return p.maxWorkers
}

// Strategies returns the configured strategies.
func (p *Pipeline[T]) Strategies() []Strategy {
	return p.strategies
}
