//go:build embed_model

package qrdet

import (
	_ "embed"
)

// embeddedModel contains the ONNX model data embedded at compile time.
// The model file must exist at build time for the embed directive to work.
//
// To build with the embedded model:
//  1. First export the model: make model
//  2. Then build with tag: go build -tags embed_model ./cmd/qrdetect
//
// The model file is expected at: qrdet/models/qrdet-s.onnx
// Size is approximately 20-50MB depending on the model variant.
//
//go:embed models/qrdet-s.onnx
var embeddedModel []byte

// GetEmbeddedModel returns the embedded ONNX model data.
// Returns the full model when built with -tags embed_model.
func GetEmbeddedModel() []byte {
	return embeddedModel
}

// HasEmbeddedModel returns true if a model is embedded in the binary.
func HasEmbeddedModel() bool {
	return len(embeddedModel) > 0
}

// EmbeddedModelSize returns the size of the embedded model in bytes.
func EmbeddedModelSize() int {
	return len(embeddedModel)
}
