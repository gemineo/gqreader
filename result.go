// Package gqreader provides end-to-end QR code detection and decoding.
//
// It combines YOLOv8-based detection (qrdet) with a brute-force decode pipeline
// that tries multiple image preprocessing strategies to maximize decode success rate.
//
// Basic usage:
//
//	qr, err := gqreader.New(gqreader.DefaultConfig())
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer qr.Close()
//
//	results, err := qr.DetectAndDecode("image.png")
//	for _, r := range results {
//	    if r.Decoded {
//	        fmt.Printf("QR: %s (confidence: %.0f%%)\n", r.Content, r.Detection.Confidence*100)
//	    }
//	}
package gqreader

import "github.com/gemineo/gqreader/qrdet"

// QRResult represents a detected QR code with optional decoded content.
type QRResult struct {
	// Detection contains the YOLOv8 detection data (bbox, quad, polygon, confidence).
	Detection qrdet.Detection

	// Content is the decoded text. Empty if decoding failed.
	Content string

	// Decoded is true if the QR code was successfully decoded.
	Decoded bool

	// Strategy identifies which preprocessing strategy succeeded.
	// Format: "crop/scale/flavor" (e.g., "bbox/scale_2.00/inversion").
	// Empty if decoding failed.
	Strategy string
}
