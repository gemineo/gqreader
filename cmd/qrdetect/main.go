// Command qrdetect detects QR codes in images using YOLOv8 segmentation.
//
// Usage:
//
//	qrdetect -input image.png -output result.png
//	qrdetect -model path/to/model.onnx -input image.png -output result.png
//	qrdetect -input image.png -json
//
// The command loads an image, runs QR code detection, and outputs an annotated
// image showing detected QR codes with bounding boxes, quadrilaterals, and
// confidence scores.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"os"

	"gocv.io/x/gocv"

	"github.com/gemineo/gqreader/qrdet"
)

func main() {
	// Parse command-line flags
	inputPath := flag.String("input", "", "Input image path (required)")
	outputPath := flag.String("output", "output.png", "Output annotated image path")
	modelPath := flag.String("model", "", "ONNX model path (optional, uses embedded model if not specified)")
	confThreshold := flag.Float64("conf", 0.5, "Confidence threshold (0.0-1.0)")
	nmsThreshold := flag.Float64("nms", 0.3, "NMS IoU threshold (0.0-1.0)")
	jsonOutput := flag.Bool("json", false, "Output detection results as JSON instead of image")
	verbose := flag.Bool("v", false, "Verbose output")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s -input <image> [options]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Detect QR codes in images using YOLOv8 segmentation.\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  %s -input photo.jpg -output result.png\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -input photo.jpg -json\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -model models/qrdet-s.onnx -input photo.jpg\n", os.Args[0])
	}

	flag.Parse()

	// Validate required arguments
	if *inputPath == "" {
		fmt.Fprintln(os.Stderr, "Error: -input is required")
		flag.Usage()
		os.Exit(1)
	}

	// Check if input file exists
	if _, err := os.Stat(*inputPath); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "Error: input file not found: %s\n", *inputPath)
		os.Exit(1)
	}

	// Create detector configuration
	config := qrdet.DefaultConfig().
		WithConfidence(float32(*confThreshold)).
		WithNMSIoU(float32(*nmsThreshold))

	if *verbose {
		fmt.Printf("Configuration: conf=%.2f, nms=%.2f\n", config.ConfidenceThreshold, config.NMSIoUThreshold)
	}

	// Create detector
	var detector *qrdet.Detector
	var err error

	if *modelPath != "" {
		// Load from file
		if *verbose {
			fmt.Printf("Loading model from: %s\n", *modelPath)
		}
		detector, err = qrdet.NewDetectorFromFile(*modelPath, config)
	} else if qrdet.HasEmbeddedModel() {
		// Use embedded model
		if *verbose {
			fmt.Printf("Using embedded model (%d bytes)\n", qrdet.EmbeddedModelSize())
		}
		detector, err = qrdet.NewDetector(config)
	} else {
		fmt.Fprintln(os.Stderr, "Error: no model available")
		fmt.Fprintln(os.Stderr, "Either:")
		fmt.Fprintln(os.Stderr, "  - Specify -model path/to/model.onnx")
		fmt.Fprintln(os.Stderr, "  - Build with embedded model: go build -tags embed_model")
		os.Exit(1)
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating detector: %v\n", err)
		os.Exit(1)
	}
	defer detector.Close()

	// Load input image
	if *verbose {
		fmt.Printf("Loading image: %s\n", *inputPath)
	}

	// Load the image using GoCV (we need the Mat for drawing later)
	mat := gocv.IMRead(*inputPath, gocv.IMReadColor)
	if mat.Empty() {
		fmt.Fprintf(os.Stderr, "Error: failed to load image: %s\n", *inputPath)
		os.Exit(1)
	}
	defer mat.Close()

	if *verbose {
		fmt.Printf("Image size: %dx%d\n", mat.Cols(), mat.Rows())
	}

	// Create ImageSource for detection (no-copy wrapper)
	img := qrdet.NewImageFromMat(mat)

	// Run detection
	if *verbose {
		fmt.Println("Running detection...")
	}

	detections, err := detector.Detect(img)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error during detection: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Found %d QR code(s)\n", len(detections))

	// Output results
	if *jsonOutput {
		// JSON output mode
		outputJSON(detections)
	} else {
		// Image output mode
		drawDetections(&mat, detections)

		if *verbose {
			fmt.Printf("Saving annotated image: %s\n", *outputPath)
		}

		if !gocv.IMWrite(*outputPath, mat) {
			fmt.Fprintf(os.Stderr, "Error: failed to save image: %s\n", *outputPath)
			os.Exit(1)
		}

		fmt.Printf("Saved annotated image: %s\n", *outputPath)
	}

	// Print summary
	for i, det := range detections {
		fmt.Printf("  [%d] confidence=%.2f bbox=[%.0f, %.0f, %.0f, %.0f]\n",
			i+1, det.Confidence,
			det.BBoxXYXY[0], det.BBoxXYXY[1], det.BBoxXYXY[2], det.BBoxXYXY[3])
	}
}

// outputJSON prints detection results as JSON to stdout.
func outputJSON(detections []qrdet.Detection) {
	type jsonDetection struct {
		Index       int          `json:"index"`
		Confidence  float32      `json:"confidence"`
		BBoxXYXY    [4]float32   `json:"bbox_xyxy"`
		CenterXY    [2]float32   `json:"center_xy"`
		WidthHeight [2]float32   `json:"width_height"`
		QuadXY      [4][2]float32 `json:"quad_xy"`
	}

	output := make([]jsonDetection, len(detections))
	for i, det := range detections {
		output[i] = jsonDetection{
			Index:       i + 1,
			Confidence:  det.Confidence,
			BBoxXYXY:    det.BBoxXYXY,
			CenterXY:    det.CenterXY,
			WidthHeight: det.WidthHeight,
			QuadXY:      det.QuadXY,
		}
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(output); err != nil {
		fmt.Fprintf(os.Stderr, "Error encoding JSON: %v\n", err)
		os.Exit(1)
	}
}

// drawDetections draws bounding boxes and quadrilaterals on the image.
func drawDetections(img *gocv.Mat, detections []qrdet.Detection) {
	// Colors
	bboxColor := color.RGBA{R: 0, G: 255, B: 0, A: 255}      // Green for bbox
	quadColor := color.RGBA{R: 0, G: 128, B: 255, A: 255}    // Blue for quad
	textColor := color.RGBA{R: 255, G: 255, B: 255, A: 255}  // White for text
	bgColor := color.RGBA{R: 0, G: 0, B: 0, A: 255}          // Black background for text

	for i, det := range detections {
		// Draw bounding box
		rect := image.Rect(
			int(det.BBoxXYXY[0]), int(det.BBoxXYXY[1]),
			int(det.BBoxXYXY[2]), int(det.BBoxXYXY[3]),
		)
		gocv.Rectangle(img, rect, bboxColor, 2)

		// Draw quadrilateral
		for j := 0; j < 4; j++ {
			k := (j + 1) % 4
			p1 := image.Pt(int(det.QuadXY[j][0]), int(det.QuadXY[j][1]))
			p2 := image.Pt(int(det.QuadXY[k][0]), int(det.QuadXY[k][1]))
			gocv.Line(img, p1, p2, quadColor, 2)
		}

		// Draw corner points of quad
		for j := 0; j < 4; j++ {
			center := image.Pt(int(det.QuadXY[j][0]), int(det.QuadXY[j][1]))
			gocv.Circle(img, center, 4, quadColor, -1) // Filled circle
		}

		// Draw confidence label
		label := fmt.Sprintf("#%d: %.0f%%", i+1, det.Confidence*100)
		labelPos := image.Pt(int(det.BBoxXYXY[0]), int(det.BBoxXYXY[1])-8)

		// Ensure label is visible (not above image)
		if labelPos.Y < 15 {
			labelPos.Y = int(det.BBoxXYXY[3]) + 15
		}

		// Draw text background
		textSize := gocv.GetTextSize(label, gocv.FontHersheyPlain, 1.2, 2)
		bgRect := image.Rect(
			labelPos.X-2, labelPos.Y-textSize.Y-2,
			labelPos.X+textSize.X+2, labelPos.Y+4,
		)
		gocv.Rectangle(img, bgRect, bgColor, -1) // Filled rectangle

		// Draw text
		gocv.PutText(img, label, labelPos, gocv.FontHersheyPlain, 1.2, textColor, 2)
	}

	// Draw legend in top-left corner
	legendY := 20
	gocv.PutText(img, fmt.Sprintf("Detected: %d QR code(s)", len(detections)),
		image.Pt(10, legendY), gocv.FontHersheyPlain, 1.0, textColor, 1)
}
