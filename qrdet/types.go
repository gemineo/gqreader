// Package qrdet provides QR code detection using YOLOv8 segmentation.
//
// This package is a Go port of the Python QRdet library, providing full parity
// with the original implementation. It uses ONNX Runtime for model inference
// and GoCV for image processing.
//
// Basic usage:
//
//	detector, err := qrdet.NewDetector(qrdet.DefaultConfig())
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer detector.Close()
//
//	detections, err := detector.DetectFile("image.png")
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	for _, det := range detections {
//	    fmt.Printf("QR code at %v with confidence %.2f\n", det.BBoxXYXY, det.Confidence)
//	}
package qrdet

// Detection represents a single detected QR code in an image.
// All coordinate fields have both absolute (pixel) and normalized (0-1) variants.
type Detection struct {
	// Confidence is the detection confidence score (0.0 to 1.0).
	Confidence float32

	// ImageWidth is the width of the source image in pixels.
	ImageWidth int

	// ImageHeight is the height of the source image in pixels.
	ImageHeight int

	// BBoxXYXY is the bounding box in [x1, y1, x2, y2] format (pixels).
	// (x1, y1) is top-left, (x2, y2) is bottom-right.
	BBoxXYXY [4]float32

	// BBoxXYXYN is the bounding box normalized to [0, 1] range.
	BBoxXYXYN [4]float32

	// CenterXY is the center point of the bounding box (pixels).
	CenterXY [2]float32

	// CenterXYN is the center point normalized to [0, 1] range.
	CenterXYN [2]float32

	// WidthHeight is the width and height of the bounding box (pixels).
	WidthHeight [2]float32

	// WidthHeightN is the width and height normalized to [0, 1] range.
	WidthHeightN [2]float32

	// PolygonXY is the precise segmentation polygon as a list of (x, y) points.
	// This provides the accurate boundary of the detected QR code.
	PolygonXY [][2]float32

	// PolygonXYN is the segmentation polygon normalized to [0, 1] range.
	PolygonXYN [][2]float32

	// QuadXY is a fitted 4-point quadrilateral approximating the QR code boundary.
	// Points are ordered clockwise starting from top-left.
	QuadXY [4][2]float32

	// QuadXYN is the quadrilateral normalized to [0, 1] range.
	QuadXYN [4][2]float32

	// PaddedQuadXY is an expanded quadrilateral that fully contains the polygon.
	// Useful for cropping with margin.
	PaddedQuadXY [4][2]float32

	// PaddedQuadXYN is the padded quadrilateral normalized to [0, 1] range.
	PaddedQuadXYN [4][2]float32
}

// Config holds configuration for the QR detector.
// Use DefaultConfig() to get the default configuration matching Python QRdet.
type Config struct {
	// ConfidenceThreshold is the minimum confidence score for a detection.
	// Detections below this threshold are filtered out.
	// Default: 0.5 (matching Python QRdet)
	ConfidenceThreshold float32

	// NMSIoUThreshold is the IoU threshold for Non-Maximum Suppression.
	// Higher values allow more overlapping detections.
	// Default: 0.3 (matching Python QRdet)
	NMSIoUThreshold float32

	// MaxDetections is the maximum number of detections to return.
	// Default: 100 (matching Python QRdet)
	MaxDetections int

	// InputSize is the model input size (width and height).
	// The model was trained with 640x640 images.
	// Default: 640
	InputSize int

	// Provider configures the ONNX Runtime execution provider and session options.
	// Use CPUProviderConfig() or CoreMLProviderConfig() to configure.
	// The zero value selects CPU with full graph optimization.
	Provider ProviderConfig
}

// DefaultConfig returns the default configuration matching Python QRdet.
func DefaultConfig() Config {
	return Config{
		ConfidenceThreshold: 0.5,
		NMSIoUThreshold:     0.3,
		MaxDetections:       100,
		InputSize:           640,
		Provider:            CPUProviderConfig(),
	}
}

// WithConfidence returns a copy of the config with the given confidence threshold.
func (c Config) WithConfidence(threshold float32) Config {
	c.ConfidenceThreshold = threshold
	return c
}

// WithNMSIoU returns a copy of the config with the given NMS IoU threshold.
func (c Config) WithNMSIoU(threshold float32) Config {
	c.NMSIoUThreshold = threshold
	return c
}

// WithMaxDetections returns a copy of the config with the given max detections.
func (c Config) WithMaxDetections(max int) Config {
	c.MaxDetections = max
	return c
}

// WithProvider returns a copy of the config with the given execution provider.
func (c Config) WithProvider(p ProviderConfig) Config {
	c.Provider = p
	return c
}

// modelInputSize is the fixed input size for the ONNX model.
const modelInputSize = 640

// modelOutputPredictions is the number of predictions in the model output.
// YOLOv8 produces 8400 predictions (80x80 + 40x40 + 20x20 grid cells * 1 anchor).
const modelOutputPredictions = 8400

// modelOutputChannels is the number of channels per prediction.
// For QRdet (single-class detection):
// 4 (bbox) + 1 (confidence) + 32 (mask coefficients) = 37
// (vs. standard YOLO: 4 + 80 + 32 = 116)
const modelOutputChannels = 37

// modelMaskCoefficients is the number of mask coefficients per prediction.
const modelMaskCoefficients = 32

// modelPrototypeSize is the spatial size of prototype masks.
const modelPrototypeSize = 160

// modelNumClasses is the number of classes (QR code is single class).
const modelNumClasses = 1

// qrClassIndex is the class index for QR codes in the model output.
// QRdet is trained for single-class detection, so confidence is at index 4.
// Layout: [cx, cy, w, h, confidence, mask_coeff_0, ..., mask_coeff_31]
const qrClassIndex = 4
