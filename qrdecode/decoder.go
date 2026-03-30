// Package qrdecode provides QR code decoding abstractions.
//
// It defines a Decoder interface that can be implemented by different backends
// (OpenCV, zbar, etc.) and a default OpenCV-based implementation.
package qrdecode

import "gocv.io/x/gocv"

// DecodedQR represents a successfully decoded QR code.
type DecodedQR struct {
	// Content is the decoded UTF-8 text.
	Content string

	// Points contains the corner points detected by the decoder, if available.
	// May be nil if the decoder does not provide point information.
	Points [][2]float32
}

// Decoder decodes QR codes from images.
// Implementations must be safe for sequential use but need not be concurrent-safe.
// Call Close() when done to release resources.
type Decoder interface {
	// Decode attempts to find and decode QR codes in the given image.
	// Returns decoded results and any error encountered.
	// An empty slice with nil error means no QR codes were found.
	Decode(img gocv.Mat) ([]DecodedQR, error)

	// Close releases resources held by the decoder.
	Close() error
}
