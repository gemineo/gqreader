//go:build !embed_model

package qrdet

// This file provides a stub for development when the ONNX model is not embedded.
// When building for production with embedded model, use: go build -tags embed_model
//
// Build tags:
//   - Default (no tag): Uses this stub, model must be loaded from file
//   - embed_model: Uses model.go with embedded ONNX file
//
// During development, use NewDetectorFromFile() to load the model:
//
//	detector, err := qrdet.NewDetectorFromFile("models/qrdet-s.onnx", qrdet.DefaultConfig())

// GetEmbeddedModel returns nil when not built with embed_model tag.
// Use NewDetectorFromFile() instead during development.
func GetEmbeddedModel() []byte {
	return nil
}

// HasEmbeddedModel returns false when not built with embed_model tag.
func HasEmbeddedModel() bool {
	return false
}

// EmbeddedModelSize returns 0 when not built with embed_model tag.
func EmbeddedModelSize() int {
	return 0
}
