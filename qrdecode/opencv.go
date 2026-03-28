package qrdecode

import (
	"fmt"

	"gocv.io/x/gocv"
)

// OpenCVDecoder decodes QR codes using GoCV's QRCodeDetector.
//
// It wraps gocv.QRCodeDetector and uses DetectAndDecode for single-QR detection
// per image crop. This is the default decoder for the QReader pipeline.
type OpenCVDecoder struct {
	detector gocv.QRCodeDetector
}

// NewOpenCVDecoder creates a new OpenCV-based QR code decoder.
func NewOpenCVDecoder() (*OpenCVDecoder, error) {
	det := gocv.NewQRCodeDetector()
	return &OpenCVDecoder{detector: det}, nil
}

// Decode attempts to decode a QR code from the given image.
//
// Uses DetectAndDecode which handles both detection and decoding in one step.
// Returns at most one decoded QR code per call (OpenCV's single-QR API).
func (d *OpenCVDecoder) Decode(img gocv.Mat) ([]DecodedQR, error) {
	if img.Empty() {
		return nil, fmt.Errorf("input image is empty")
	}

	points := gocv.NewMat()
	defer points.Close()

	straight := gocv.NewMat()
	defer straight.Close()

	content := d.detector.DetectAndDecode(img, &points, &straight)
	if content == "" {
		return nil, nil
	}

	result := DecodedQR{Content: content}

	// Extract corner points if available.
	// Points Mat is typically 1x4 CV_32FC2 (4 points, 2 floats each = 8 floats total).
	if !points.Empty() {
		if data, err := points.DataPtrFloat32(); err == nil && len(data) >= 8 {
			numPoints := len(data) / 2
			if numPoints > 4 {
				numPoints = 4
			}
			result.Points = make([][2]float32, numPoints)
			for i := 0; i < numPoints; i++ {
				result.Points[i] = [2]float32{data[i*2], data[i*2+1]}
			}
		}
	}

	return []DecodedQR{result}, nil
}

// Close releases the QRCodeDetector resources.
func (d *OpenCVDecoder) Close() error {
	d.detector.Close()
	return nil
}

// Ensure OpenCVDecoder implements Decoder.
var _ Decoder = (*OpenCVDecoder)(nil)
