# imgprep - Image Preprocessing for Barcode Detection

Package `imgprep` provides image preprocessing strategies optimized for barcode detection on degraded documents (fax, low-quality scans, folded paper, etc.).

## Features

- **Parallel Pipeline Architecture**: Run multiple preprocessing strategies concurrently
- **Pluggable Strategies**: Easy to add custom preprocessing strategies
- **Result Aggregation**: Combine and deduplicate results from all strategies
- **Battle-tested Algorithms**: Includes Sauvola, Wolf, Niblack binarization and more
- **Go 1.23+ Iterator Support**: Uses modern Go iteration patterns

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"

    "gocv.io/x/gocv"
    "github.com/gemineo/gqreader/imgprep"
    "github.com/gemineo/gqreader/qrdet"
)

func main() {
    // Create detector
    detector, err := qrdet.NewDetector(qrdet.DefaultConfig())
    if err != nil {
        log.Fatal(err)
    }
    defer detector.Close()

    // Create preprocessing pipeline
    pipeline := imgprep.NewPipeline[qrdet.Detection](
        imgprep.WithStrategies[qrdet.Detection](imgprep.DefaultStrategies()...),
    )

    // Load image
    img := gocv.IMRead("document.png", gocv.IMReadColor)
    if img.Empty() {
        log.Fatal("Failed to load image")
    }
    defer img.Close()

    // Process with all strategies in parallel
    ctx := context.Background()
    results, err := pipeline.Process(ctx, img, func(preprocessed gocv.Mat) ([]qrdet.Detection, error) {
        imgSrc := qrdet.NewImageFromMat(preprocessed)
        defer imgSrc.Close()
        return detector.Detect(imgSrc)
    })
    if err != nil {
        log.Fatal(err)
    }

    // Aggregate and analyze results
    agg := imgprep.Aggregate(results)
    fmt.Printf("Found %d total detections\n", agg.TotalDetections())
    fmt.Printf("Best strategy: %s\n", agg.BestStrategy())
}
```

## Available Strategies

### Preset Pipelines

| Strategy | Use Case | Pipeline |
|----------|----------|----------|
| `FastStrategy()` | High-quality scans | Grayscale → Gaussian(3) → Otsu |
| `StandardStrategy()` | General documents | Grayscale → CLAHE → AdaptiveThreshold |
| `DegradedStrategy()` | Old/copied documents | Grayscale → Median(5) → CLAHE(3.0) → AdaptiveThreshold(31) |
| `FaxStrategy()` | Fax documents | Grayscale → Median² → CLAHE → AdaptiveThreshold → MorphOpen |
| `SauvolaStrategy()` | Historical documents | Grayscale → Median → Sauvola |
| `RotationStrategy()` | Skewed documents | Multiple rotation angles with Standard preprocessing |

### Elementary Operations

- `Grayscale` - Color to grayscale conversion
- `GaussianBlur` - Gaussian smoothing
- `MedianBlur` - Salt-and-pepper noise removal
- `FastNlMeansDenoising` - Non-local means denoising
- `CLAHE` - Contrast Limited Adaptive Histogram Equalization
- `OtsuThreshold` - Otsu's automatic binarization
- `AdaptiveThreshold` - Local adaptive binarization
- `SauvolaThreshold` - Sauvola binarization (document-optimized)
- `WolfThreshold` - Wolf binarization
- `NiblackThreshold` - Niblack binarization
- `MorphOpen` / `MorphClose` - Morphological operations
- `Deskew` - Automatic skew correction
- `BlueChannel` - Blue channel extraction

## Creating Custom Strategies

### Using StrategyFunc

```go
invert := imgprep.NewStrategyFunc("invert", func(src gocv.Mat) (gocv.Mat, error) {
    dst := gocv.NewMat()
    gocv.BitwiseNot(src, &dst)
    return dst, nil
})
```

### Using CompositeStrategy

```go
custom := imgprep.NewCompositeStrategy("my_pipeline",
    &imgprep.Grayscale{},
    &imgprep.MedianBlur{KernelSize: 5},
    &imgprep.SauvolaThreshold{WindowSize: 21, K: 0.4},
)
```

### Implementing Strategy Interface

```go
type MyStrategy struct {
    param float64
}

func (s *MyStrategy) Name() string {
    return fmt.Sprintf("my_strategy_%.2f", s.param)
}

func (s *MyStrategy) Apply(src gocv.Mat) (gocv.Mat, error) {
    // Your preprocessing logic
    dst := gocv.NewMat()
    // ...
    return dst, nil
}
```

## Deduplication

When running multiple strategies, the same barcode may be detected multiple times. Use deduplication to merge results:

```go
dedup := &imgprep.DeduplicateFunc[qrdet.Detection]{
    KeyFn: func(d qrdet.Detection) string {
        // Group by approximate position
        return fmt.Sprintf("%.0f,%.0f", d.CenterXY[0]/10, d.CenterXY[1]/10)
    },
    MergeFn: func(dups []qrdet.Detection) qrdet.Detection {
        // Keep highest confidence
        best := dups[0]
        for _, d := range dups[1:] {
            if d.Confidence > best.Confidence {
                best = d
            }
        }
        return best
    },
}

unique := imgprep.Deduplicate(agg.AllDetections, dedup)
```

## Performance Tips

1. **Order strategies by speed**: Default strategies are ordered fast → slow
2. **Limit workers for memory-constrained systems**: `WithMaxWorkers[T](4)`
3. **Use context for timeouts**: `context.WithTimeout(ctx, 30*time.Second)`
4. **Profile with benchmarks**: `go test -bench=. -benchmem`

## Algorithm References

- **Sauvola**: Sauvola, J., & Pietikäinen, M. (2000). Adaptive document image binarization.
- **Wolf**: Wolf, C., & Jolion, J. M. (2004). Extraction and recognition of artificial text in multimedia documents.
- **Niblack**: Niblack, W. (1986). An introduction to digital image processing.
