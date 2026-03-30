package imgprep

import (
	"iter"

	"gocv.io/x/gocv"
)

// Strategy defines a preprocessing strategy for image enhancement.
// Implementations should be stateless and safe for concurrent use.
//
// The returned Mat is owned by the caller and must be closed when no longer needed.
// If an error is returned, no Mat is returned (nil).
type Strategy interface {
	// Name returns a unique identifier for this strategy.
	// Used for logging, metrics, and result attribution.
	Name() string

	// Apply preprocesses the source image and returns a new processed image.
	// The source Mat must not be modified.
	// The returned Mat is a new allocation owned by the caller.
	Apply(src gocv.Mat) (gocv.Mat, error)
}

// VariantGenerator is a strategy that can generate multiple image variants.
// This interface extends Strategy for strategies that want to explore
// parameter spaces (e.g., multiple rotation angles).
type VariantGenerator interface {
	Strategy

	// Variants returns an iterator over all generated variants.
	// Each variant is a (name suffix, processed image) pair.
	// The caller is responsible for closing each returned Mat.
	Variants(src gocv.Mat) iter.Seq2[string, gocv.Mat]
}

// StrategyFunc is an adapter to allow ordinary functions to be used as strategies.
type StrategyFunc struct {
	name    string
	applyFn func(gocv.Mat) (gocv.Mat, error)
}

// NewStrategyFunc creates a Strategy from a function.
func NewStrategyFunc(name string, fn func(gocv.Mat) (gocv.Mat, error)) *StrategyFunc {
	return &StrategyFunc{
		name:    name,
		applyFn: fn,
	}
}

func (s *StrategyFunc) Name() string {
	return s.name
}

func (s *StrategyFunc) Apply(src gocv.Mat) (gocv.Mat, error) {
	return s.applyFn(src)
}

// CompositeStrategy chains multiple strategies sequentially.
// Each strategy's output becomes the next strategy's input.
type CompositeStrategy struct {
	name       string
	strategies []Strategy
}

// NewCompositeStrategy creates a strategy that applies multiple strategies in sequence.
func NewCompositeStrategy(name string, strategies ...Strategy) *CompositeStrategy {
	return &CompositeStrategy{
		name:       name,
		strategies: strategies,
	}
}

func (c *CompositeStrategy) Name() string {
	return c.name
}

func (c *CompositeStrategy) Apply(src gocv.Mat) (gocv.Mat, error) {
	if len(c.strategies) == 0 {
		// Return a clone of the source
		dst := gocv.NewMat()
		src.CopyTo(&dst)
		return dst, nil
	}

	// Apply first strategy
	current, err := c.strategies[0].Apply(src)
	if err != nil {
		return gocv.Mat{}, err
	}

	// Chain remaining strategies
	for _, strategy := range c.strategies[1:] {
		next, err := strategy.Apply(current)
		current.Close() // Close intermediate result
		if err != nil {
			return gocv.Mat{}, err
		}
		current = next
	}

	return current, nil
}

// variantStrategy generates multiple variants of an image.
// Useful for rotation search or parameter sweep.
type variantStrategy struct {
	name      string
	generator func(src gocv.Mat) iter.Seq2[string, gocv.Mat]
}

// NewVariantStrategy creates a strategy that generates multiple image variants.
// The generator function yields (variantName, variantMat) pairs.
// Each yielded Mat must be a new allocation.
func NewVariantStrategy(name string, gen func(src gocv.Mat) iter.Seq2[string, gocv.Mat]) VariantGenerator {
	return &variantStrategy{
		name:      name,
		generator: gen,
	}
}

func (v *variantStrategy) Name() string {
	return v.name
}

// Variants returns an iterator over all generated variants.
func (v *variantStrategy) Variants(src gocv.Mat) iter.Seq2[string, gocv.Mat] {
	return v.generator(src)
}

// Apply returns the first variant (for Strategy interface compatibility).
// For full variant iteration, use Variants() instead.
func (v *variantStrategy) Apply(src gocv.Mat) (gocv.Mat, error) {
	for _, mat := range v.generator(src) {
		return mat, nil
	}
	// No variants generated, return a copy
	dst := gocv.NewMat()
	src.CopyTo(&dst)
	return dst, nil
}
