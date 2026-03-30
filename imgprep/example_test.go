package imgprep_test

import (
	"context"
	"fmt"
	"time"

	"gocv.io/x/gocv"

	"github.com/gemineo/gqreader/imgprep"
)

// MockDetection simulates a barcode detection result.
type MockDetection struct {
	Content    string
	Confidence float32
	BBox       [4]float32
}

// Example demonstrates the basic usage of the preprocessing pipeline.
func Example() {
	// Create pipeline with default strategies
	pipeline := imgprep.NewPipeline[MockDetection](
		imgprep.WithStrategies[MockDetection](imgprep.DefaultStrategies()...),
	)

	// Load image
	img := gocv.IMRead("document.png", gocv.IMReadColor)
	if img.Empty() {
		fmt.Println("Failed to load image")
		return
	}
	defer img.Close()

	// Define detection function (replace with actual detector)
	detect := func(preprocessed gocv.Mat) ([]MockDetection, error) {
		// This would call your actual barcode detector
		return []MockDetection{
			{Content: "ABC123", Confidence: 0.95},
		}, nil
	}

	// Process with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	results, err := pipeline.Process(ctx, img, detect)
	if err != nil {
		fmt.Printf("Processing failed: %v\n", err)
		return
	}

	// Aggregate results
	agg := imgprep.Aggregate(results)

	fmt.Printf("Found %d total detections\n", agg.TotalDetections())
	fmt.Printf("Successful strategies: %v\n", agg.SuccessfulStrategies)
	fmt.Printf("Best strategy: %s\n", agg.BestStrategy())
}

// Example_customStrategy demonstrates creating a custom preprocessing strategy.
func Example_customStrategy() {
	// Create a custom strategy using StrategyFunc
	invertColors := imgprep.NewStrategyFunc("invert", func(src gocv.Mat) (gocv.Mat, error) {
		dst := gocv.NewMat()
		gocv.BitwiseNot(src, &dst)
		return dst, nil
	})

	// Combine with existing strategies
	pipeline := imgprep.NewPipeline[MockDetection](
		imgprep.WithStrategies[MockDetection](
			imgprep.StandardStrategy(),
			invertColors,
		),
	)

	_ = pipeline // Use pipeline...
}

// Example_compositeStrategy demonstrates chaining multiple operations.
func Example_compositeStrategy() {
	// Create a custom composite strategy
	aggressive := imgprep.NewCompositeStrategy("aggressive",
		&imgprep.Grayscale{},
		&imgprep.MedianBlur{KernelSize: 5},
		&imgprep.CLAHE{ClipLimit: 4.0},
		&imgprep.SauvolaThreshold{WindowSize: 21, K: 0.4},
		&imgprep.MorphOpen{KernelSize: 2},
		&imgprep.MorphClose{KernelSize: 2},
	)

	pipeline := imgprep.NewPipeline[MockDetection](
		imgprep.WithStrategies[MockDetection](aggressive),
	)

	_ = pipeline
}

// Example_deduplication demonstrates deduplicating results.
func Example_deduplication() {
	// Assume we have results from pipeline processing
	detections := []MockDetection{
		{Content: "ABC123", Confidence: 0.95, BBox: [4]float32{10, 10, 100, 100}},
		{Content: "ABC123", Confidence: 0.87, BBox: [4]float32{12, 11, 98, 99}}, // Duplicate
		{Content: "XYZ789", Confidence: 0.92, BBox: [4]float32{200, 50, 300, 150}},
	}

	// Create deduplicator based on content
	dedup := &imgprep.DeduplicateFunc[MockDetection]{
		KeyFn: func(d MockDetection) string {
			return d.Content // Same content = same barcode
		},
		MergeFn: func(dups []MockDetection) MockDetection {
			// Keep the one with highest confidence
			best := dups[0]
			for _, d := range dups[1:] {
				if d.Confidence > best.Confidence {
					best = d
				}
			}
			return best
		},
	}

	unique := imgprep.Deduplicate(detections, dedup)
	fmt.Printf("Unique detections: %d\n", len(unique))
	// Output: Unique detections: 2
}

// Example_withQRDet shows integration with the qrdet package.
func Example_withQRDet() {
	/*
		// This is how you would integrate with the qrdet detector:

		import "github.com/gemineo/gqreader/qrdet"

		// Create detector
		detector, err := qrdet.NewDetector(qrdet.DefaultConfig())
		if err != nil {
			log.Fatal(err)
		}
		defer detector.Close()

		// Create pipeline
		pipeline := imgprep.NewPipeline[qrdet.Detection](
			imgprep.WithStrategies[qrdet.Detection](imgprep.AllStrategies()...),
			imgprep.WithMaxWorkers[qrdet.Detection](4), // Limit concurrency
		)

		// Load image
		img, err := qrdet.NewImageFromFile("document.png")
		if err != nil {
			log.Fatal(err)
		}
		defer img.Close()

		// Adapter: wrap qrdet.Detect to work with gocv.Mat
		detectFn := func(preprocessed gocv.Mat) ([]qrdet.Detection, error) {
			// Convert gocv.Mat back to qrdet.ImageSource
			imgSrc := qrdet.NewImageFromMat(preprocessed)
			defer imgSrc.Close()
			return detector.Detect(imgSrc)
		}

		// Process
		ctx := context.Background()
		results, err := pipeline.Process(ctx, img.ToMat(), detectFn)
		if err != nil {
			log.Fatal(err)
		}

		// Deduplicate based on decoded content
		dedup := &imgprep.DeduplicateFunc[qrdet.Detection]{
			KeyFn: func(d qrdet.Detection) string {
				// You would need the decoded content from pyzbar or similar
				return fmt.Sprintf("%.0f,%.0f", d.CenterXY[0], d.CenterXY[1])
			},
			MergeFn: func(dups []qrdet.Detection) qrdet.Detection {
				best := dups[0]
				for _, d := range dups[1:] {
					if d.Confidence > best.Confidence {
						best = d
					}
				}
				return best
			},
		}

		agg := imgprep.Aggregate(results)
		unique := imgprep.Deduplicate(agg.AllDetections, dedup)

		fmt.Printf("Found %d unique QR codes\n", len(unique))
	*/
}
