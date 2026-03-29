package qrdet

import (
	"fmt"
)

// Detector is the main interface for QR code detection.
// It wraps an ONNX Runtime session and provides a simple API for detecting
// QR codes in images.
//
// A Detector should be created once and reused for multiple detections.
// Call Close() when done to release resources.
type Detector struct {
	session *session
	config  Config
}

// NewDetector creates a new QR detector using the embedded ONNX model.
// The model is compiled into the binary using go:embed.
//
// Example:
//
//	detector, err := qrdet.NewDetector(qrdet.DefaultConfig())
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer detector.Close()
func NewDetector(config Config) (*Detector, error) {
	modelData := GetEmbeddedModel()
	if len(modelData) == 0 {
		return nil, fmt.Errorf("embedded model is empty; ensure the model was built with go:embed")
	}

	return newDetectorFromData(modelData, config)
}

// NewDetectorFromFile creates a new QR detector from an external ONNX model file.
// This is useful during development or when the model is too large to embed.
//
// Example:
//
//	detector, err := qrdet.NewDetectorFromFile("models/qrdet-s.onnx", qrdet.DefaultConfig())
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer detector.Close()
func NewDetectorFromFile(modelPath string, config Config) (*Detector, error) {
	if config.InputSize == 0 {
		config.InputSize = modelInputSize
	}

	sess, err := newSessionFromFile(modelPath, config.InputSize, config.Provider)
	if err != nil {
		return nil, fmt.Errorf("failed to create session from file: %w", err)
	}

	return &Detector{
		session: sess,
		config:  config,
	}, nil
}

// newDetectorFromData creates a detector from raw ONNX model bytes.
func newDetectorFromData(modelData []byte, config Config) (*Detector, error) {
	if config.InputSize == 0 {
		config.InputSize = modelInputSize
	}

	sess, err := newSession(modelData, config.InputSize, config.Provider)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	return &Detector{
		session: sess,
		config:  config,
	}, nil
}

// Detect runs QR code detection on an image.
// Returns a slice of Detection structs, one for each detected QR code.
//
// The input must be a valid ImageSource created via NewImageFromFile, NewImageFromBytes,
// or NewImageFromRaw. Images of any size are accepted; they will be letterboxed to the
// model input size.
//
// Example:
//
//	img, err := qrdet.NewImageFromFile("photo.jpg")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer img.Close()
//
//	detections, err := detector.Detect(img)
//	for _, det := range detections {
//	    fmt.Printf("QR at %v (confidence: %.2f)\n", det.BBoxXYXY, det.Confidence)
//	}
func (d *Detector) Detect(img *ImageSource) ([]Detection, error) {
	if img == nil {
		return nil, fmt.Errorf("image source is nil")
	}

	// Convert ImageSource to internal gocv.Mat
	mat := img.toMat()
	if mat.Empty() {
		return nil, fmt.Errorf("input image is empty")
	}

	// Preprocess: letterbox, normalize, convert to CHW
	inputData, letterboxInfo, err := preprocessImage(mat, d.config.InputSize)
	if err != nil {
		return nil, fmt.Errorf("preprocessing failed: %w", err)
	}

	// Set input tensor
	if err := d.session.setInput(inputData); err != nil {
		return nil, fmt.Errorf("failed to set input: %w", err)
	}

	// Run inference
	if err := d.session.run(); err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}

	// Get outputs
	output0 := d.session.getOutput0()
	output1 := d.session.getOutput1()

	// Post-process: parse detections, apply NMS, generate masks
	detections, err := processOutput(output0, output1, letterboxInfo, d.config)
	if err != nil {
		return nil, fmt.Errorf("post-processing failed: %w", err)
	}

	return detections, nil
}

// DetectFile loads an image from a file and runs detection.
// This is a convenience method that handles image loading and resource cleanup.
//
// Example:
//
//	detections, err := detector.DetectFile("photo.jpg")
func (d *Detector) DetectFile(imagePath string) ([]Detection, error) {
	img, err := NewImageFromFile(imagePath)
	if err != nil {
		return nil, err
	}
	defer img.Close()

	return d.Detect(img)
}

// Close releases all resources associated with the detector.
// Always call Close() when done with the detector.
func (d *Detector) Close() error {
	if d.session != nil {
		return d.session.close()
	}
	return nil
}

// Config returns the current configuration.
func (d *Detector) Config() Config {
	return d.config
}

// SetConfidenceThreshold updates the confidence threshold for future detections.
func (d *Detector) SetConfidenceThreshold(threshold float32) {
	d.config.ConfidenceThreshold = threshold
}

// SetNMSIoUThreshold updates the NMS IoU threshold for future detections.
func (d *Detector) SetNMSIoUThreshold(threshold float32) {
	d.config.NMSIoUThreshold = threshold
}
