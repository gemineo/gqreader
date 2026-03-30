package imgprep

import (
	"cmp"
	"slices"
)

// AggregatedResults holds the combined results from all preprocessing strategies.
type AggregatedResults[T any] struct {
	// AllDetections contains every detection from all strategies, potentially with duplicates.
	AllDetections []T

	// ByStrategy maps strategy names to their detections.
	ByStrategy map[string][]T

	// StrategyStats contains timing and error information per strategy.
	StrategyStats []StrategyResult[T]

	// SuccessfulStrategies lists strategies that found at least one barcode.
	SuccessfulStrategies []string

	// FailedStrategies lists strategies that encountered errors.
	FailedStrategies []string
}

// Aggregate combines results from multiple strategies into a single summary.
func Aggregate[T any](results []StrategyResult[T]) *AggregatedResults[T] {
	agg := &AggregatedResults[T]{
		AllDetections:        make([]T, 0),
		ByStrategy:           make(map[string][]T),
		StrategyStats:        results,
		SuccessfulStrategies: make([]string, 0),
		FailedStrategies:     make([]string, 0),
	}

	for _, r := range results {
		fullName := r.FullName()

		if r.Error != nil {
			agg.FailedStrategies = append(agg.FailedStrategies, fullName)
			continue
		}

		agg.ByStrategy[fullName] = r.Detections
		agg.AllDetections = append(agg.AllDetections, r.Detections...)

		if len(r.Detections) > 0 {
			agg.SuccessfulStrategies = append(agg.SuccessfulStrategies, fullName)
		}
	}

	return agg
}

// TotalDetections returns the total number of detections (may include duplicates).
func (a *AggregatedResults[T]) TotalDetections() int {
	return len(a.AllDetections)
}

// UniqueCount returns the number of unique detections after deduplication.
// Requires a Deduplicator to be applied first.
func (a *AggregatedResults[T]) HasResults() bool {
	return len(a.AllDetections) > 0
}

// BestStrategy returns the strategy that found the most detections.
func (a *AggregatedResults[T]) BestStrategy() string {
	var best string
	var bestCount int

	for strategy, detections := range a.ByStrategy {
		if len(detections) > bestCount {
			best = strategy
			bestCount = len(detections)
		}
	}

	return best
}

// Deduplicator defines how to identify and merge duplicate detections.
type Deduplicator[T any] interface {
	// Key returns a unique identifier for a detection.
	// Detections with the same key are considered duplicates.
	Key(detection T) string

	// Merge combines multiple duplicate detections into one.
	// The input slice contains all detections with the same key.
	// Return the "best" detection to keep.
	Merge(detections []T) T
}

// Deduplicate removes duplicate detections using the provided Deduplicator.
func Deduplicate[T any](detections []T, dedup Deduplicator[T]) []T {
	if len(detections) == 0 {
		return nil
	}

	// Group by key
	groups := make(map[string][]T)
	keyOrder := make([]string, 0) // Preserve order

	for _, d := range detections {
		key := dedup.Key(d)
		if _, exists := groups[key]; !exists {
			keyOrder = append(keyOrder, key)
		}
		groups[key] = append(groups[key], d)
	}

	// Merge each group
	result := make([]T, 0, len(groups))
	for _, key := range keyOrder {
		merged := dedup.Merge(groups[key])
		result = append(result, merged)
	}

	return result
}

// DeduplicateFunc is a function-based Deduplicator.
type DeduplicateFunc[T any] struct {
	KeyFn   func(T) string
	MergeFn func([]T) T
}

func (d *DeduplicateFunc[T]) Key(detection T) string {
	return d.KeyFn(detection)
}

func (d *DeduplicateFunc[T]) Merge(detections []T) T {
	return d.MergeFn(detections)
}

// SortByConfidence sorts detections by confidence score in descending order.
// Requires detections to have a Confidence field accessible via the getter function.
func SortByConfidence[T any](detections []T, getConfidence func(T) float32) []T {
	result := make([]T, len(detections))
	copy(result, detections)

	slices.SortFunc(result, func(a, b T) int {
		// Descending order
		return cmp.Compare(getConfidence(b), getConfidence(a))
	})

	return result
}

// FilterByConfidence removes detections below the threshold.
func FilterByConfidence[T any](detections []T, threshold float32, getConfidence func(T) float32) []T {
	result := make([]T, 0, len(detections))
	for _, d := range detections {
		if getConfidence(d) >= threshold {
			result = append(result, d)
		}
	}
	return result
}

// TopN returns the top N detections by confidence.
func TopN[T any](detections []T, n int, getConfidence func(T) float32) []T {
	if len(detections) <= n {
		return SortByConfidence(detections, getConfidence)
	}

	sorted := SortByConfidence(detections, getConfidence)
	return sorted[:n]
}
