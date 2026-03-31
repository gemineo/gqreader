// Command qreader detects and decodes QR codes in images.
//
// Usage:
//
//	qreader -input image.png
//	qreader -model path/to/model.onnx -input image.png
//	qreader -input image.png -json
//	qreader -input image.png -output annotated.png
//
// The command runs the full QReader pipeline: YOLOv8 detection followed by
// brute-force decode with multiple preprocessing strategies.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"os"

	"gocv.io/x/gocv"

	gqreader "github.com/gemineo/gqreader"
	"github.com/gemineo/gqreader/qrdet"
)

// appendWeChatFallback is defined in wechat.go (built with -tags wechat) or
// wechat_stub.go (default build). It appends WeChatDecoder as a fallback decoder.

func main() {
	inputPath := flag.String("input", "", "Input image path (required)")
	outputPath := flag.String("output", "", "Output annotated image path (optional)")
	modelPath := flag.String("model", "", "ONNX model path (optional, uses embedded model if not specified)")
	confThreshold := flag.Float64("conf", 0.5, "Confidence threshold (0.0-1.0)")
	jsonOutput := flag.Bool("json", false, "Output results as JSON")
	minimal := flag.Bool("minimal", false, "Use minimal config (faster, skip scaling/perspective)")
	fast := flag.Bool("fast", false, "Disable fallback decoder chain (ZBar only). Faster, but may mis-encode Latin-1 QR codes.")
	wechatModels := flag.String("wechat-models", os.Getenv("QREADER_WECHAT_MODELS"), "Path to WeChatQR Caffe model directory (3rd fallback for curved/tiny QR; requires -tags wechat build)")
	verbose := flag.Bool("v", false, "Verbose output")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s -input <image> [options]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Detect and decode QR codes in images.\n\n")
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  %s -input photo.jpg\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -input photo.jpg -json\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -input photo.jpg -output result.png\n", os.Args[0])
	}
	flag.Parse()

	if *inputPath == "" {
		fmt.Fprintln(os.Stderr, "Error: -input is required")
		flag.Usage()
		os.Exit(1)
	}

	if _, err := os.Stat(*inputPath); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "Error: input file not found: %s\n", *inputPath)
		os.Exit(1)
	}

	// Build config
	var config gqreader.Config
	switch {
	case *minimal:
		config = gqreader.MinimalConfig()
	case *fast:
		config = gqreader.FastConfig()
	default:
		config = gqreader.DefaultConfig()
	}
	config.DetectionConfig = config.DetectionConfig.WithConfidence(float32(*confThreshold))
	if !*fast && *wechatModels != "" {
		config = appendWeChatFallback(config, *wechatModels)
	}

	if *verbose {
		fmt.Printf("Configuration: conf=%.2f, perspective=%v, scales=%v\n",
			config.DetectionConfig.ConfidenceThreshold, config.TryPerspective, config.ScaleFactors)
	}

	// Create QReader
	var qr *gqreader.QReader
	var err error

	if *modelPath != "" {
		qr, err = gqreader.NewFromFile(*modelPath, config)
	} else if qrdet.HasEmbeddedModel() {
		qr, err = gqreader.New(config)
	} else {
		fmt.Fprintln(os.Stderr, "Error: no model available")
		fmt.Fprintln(os.Stderr, "Either:")
		fmt.Fprintln(os.Stderr, "  - Specify -model path/to/model.onnx")
		fmt.Fprintln(os.Stderr, "  - Build with embedded model: go build -tags embed_model")
		os.Exit(1)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating QReader: %v\n", err)
		os.Exit(1)
	}
	defer qr.Close()

	if *verbose {
		fmt.Printf("Loading image: %s\n", *inputPath)
	}

	// Run detection + decoding
	results, err := qr.DetectAndDecodeFile(*inputPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	// Count decoded
	decoded := 0
	for _, r := range results {
		if r.Decoded {
			decoded++
		}
	}
	fmt.Printf("Found %d QR code(s), decoded %d\n", len(results), decoded)

	if *jsonOutput {
		printJSON(results)
	} else {
		printResults(results)
	}

	// Optionally save annotated image
	if *outputPath != "" {
		saveAnnotated(*inputPath, *outputPath, results)
	}
}

func printResults(results []gqreader.QRResult) {
	for i, r := range results {
		if r.Decoded {
			fmt.Printf("  [%d] \"%s\" (confidence=%.0f%%, strategy=%s)\n",
				i+1, r.Content, r.Detection.Confidence*100, r.Strategy)
		} else {
			fmt.Printf("  [%d] <not decoded> (confidence=%.0f%%)\n",
				i+1, r.Detection.Confidence*100)
		}
	}
}

func printJSON(results []gqreader.QRResult) {
	type jsonResult struct {
		Index      int          `json:"index"`
		Content    string       `json:"content,omitempty"`
		Decoded    bool         `json:"decoded"`
		Confidence float32      `json:"confidence"`
		Strategy   string       `json:"strategy,omitempty"`
		BBoxXYXY   [4]float32   `json:"bbox_xyxy"`
		QuadXY     [4][2]float32 `json:"quad_xy"`
	}

	output := make([]jsonResult, len(results))
	for i, r := range results {
		output[i] = jsonResult{
			Index:      i + 1,
			Content:    r.Content,
			Decoded:    r.Decoded,
			Confidence: r.Detection.Confidence,
			Strategy:   r.Strategy,
			BBoxXYXY:   r.Detection.BBoxXYXY,
			QuadXY:     r.Detection.QuadXY,
		}
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(output); err != nil {
		fmt.Fprintf(os.Stderr, "Error encoding JSON: %v\n", err)
		os.Exit(1)
	}
}

func saveAnnotated(inputPath, outputPath string, results []gqreader.QRResult) {
	img := gocv.IMRead(inputPath, gocv.IMReadColor)
	if img.Empty() {
		fmt.Fprintf(os.Stderr, "Error: failed to reload image for annotation\n")
		return
	}
	defer img.Close()

	bboxColor := color.RGBA{R: 0, G: 255, B: 0, A: 255}
	quadColor := color.RGBA{R: 0, G: 128, B: 255, A: 255}
	textColor := color.RGBA{R: 255, G: 255, B: 255, A: 255}
	bgColor := color.RGBA{R: 0, G: 0, B: 0, A: 255}
	decodedColor := color.RGBA{R: 0, G: 255, B: 128, A: 255}

	for i, r := range results {
		det := r.Detection

		// Draw bbox
		rect := image.Rect(
			int(det.BBoxXYXY[0]), int(det.BBoxXYXY[1]),
			int(det.BBoxXYXY[2]), int(det.BBoxXYXY[3]),
		)
		bc := bboxColor
		if r.Decoded {
			bc = decodedColor
		}
		gocv.Rectangle(&img, rect, bc, 2)

		// Draw quad
		for j := 0; j < 4; j++ {
			k := (j + 1) % 4
			p1 := image.Pt(int(det.QuadXY[j][0]), int(det.QuadXY[j][1]))
			p2 := image.Pt(int(det.QuadXY[k][0]), int(det.QuadXY[k][1]))
			gocv.Line(&img, p1, p2, quadColor, 2)
		}

		// Draw label
		label := fmt.Sprintf("#%d: %.0f%%", i+1, det.Confidence*100)
		if r.Decoded && len(r.Content) <= 40 {
			label += " " + r.Content
		} else if r.Decoded {
			label += " " + r.Content[:37] + "..."
		}

		labelPos := image.Pt(int(det.BBoxXYXY[0]), int(det.BBoxXYXY[1])-8)
		if labelPos.Y < 15 {
			labelPos.Y = int(det.BBoxXYXY[3]) + 15
		}

		textSize := gocv.GetTextSize(label, gocv.FontHersheyPlain, 1.0, 1)
		bgRect := image.Rect(
			labelPos.X-2, labelPos.Y-textSize.Y-2,
			labelPos.X+textSize.X+2, labelPos.Y+4,
		)
		gocv.Rectangle(&img, bgRect, bgColor, -1)
		gocv.PutText(&img, label, labelPos, gocv.FontHersheyPlain, 1.0, textColor, 1)
	}

	if !gocv.IMWrite(outputPath, img) {
		fmt.Fprintf(os.Stderr, "Error: failed to save annotated image: %s\n", outputPath)
		return
	}
	fmt.Printf("Saved annotated image: %s\n", outputPath)
}
