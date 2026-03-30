// Package imgprep provides image preprocessing strategies for barcode detection
// on degraded documents (fax, low-quality scans, folded paper, etc.).
//
// The package implements a parallel pipeline architecture where multiple
// preprocessing strategies run concurrently, and their results are aggregated
// and deduplicated to maximize barcode detection recall.
//
// # Architecture
//
// The pipeline executes multiple preprocessing strategies in parallel:
//
//	┌─────────────┐
//	│   Input     │
//	│   Image     │
//	└──────┬──────┘
//	       │
//	       ├──────────────┬──────────────┬──────────────┐
//	       ▼              ▼              ▼              ▼
//	┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
//	│   Fast      │ │  Standard   │ │  Degraded   │ │  Rotation   │
//	│  Strategy   │ │  Strategy   │ │  Strategy   │ │   Search    │
//	└──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
//	       │              │              │              │
//	       └──────────────┴──────────────┴──────────────┘
//	                           │
//	                           ▼
//	                  ┌─────────────────┐
//	                  │   Aggregation   │
//	                  │ & Deduplication │
//	                  └─────────────────┘
//
// # Basic Usage
//
//	pipeline := imgprep.NewPipeline(
//	    imgprep.WithStrategies(imgprep.DefaultStrategies()...),
//	)
//
//	img := gocv.IMRead("document.png", gocv.IMReadColor)
//	defer img.Close()
//
//	results, err := pipeline.Process(ctx, img, myDetector.Detect)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	for _, r := range results {
//	    fmt.Printf("Found %d barcodes with strategy %s\n", len(r.Detections), r.Strategy)
//	}
//
// # Custom Strategies
//
// You can implement custom preprocessing strategies by implementing the Strategy interface:
//
//	type MyStrategy struct{}
//
//	func (s *MyStrategy) Name() string { return "my-custom" }
//
//	func (s *MyStrategy) Apply(src gocv.Mat) (gocv.Mat, error) {
//	    // Apply your preprocessing...
//	    return processed, nil
//	}
//
//	pipeline := imgprep.NewPipeline(
//	    imgprep.WithStrategies(&MyStrategy{}),
//	)
package imgprep
