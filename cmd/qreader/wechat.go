//go:build wechat

package main

import (
	"path/filepath"

	gqreader "github.com/gemineo/gqreader"
	"github.com/gemineo/gqreader/qrdecode"
)

// appendWeChatFallback appends WeChatDecoder as an additional fallback decoder.
// modelDir must contain detect.prototxt, detect.caffemodel, sr.prototxt, sr.caffemodel.
func appendWeChatFallback(config gqreader.Config, modelDir string) gqreader.Config {
	return config.WithFallbackDecoderFactories(
		func() (qrdecode.Decoder, error) {
			return qrdecode.NewWeChatDecoder(
				filepath.Join(modelDir, "detect.prototxt"),
				filepath.Join(modelDir, "detect.caffemodel"),
				filepath.Join(modelDir, "sr.prototxt"),
				filepath.Join(modelDir, "sr.caffemodel"),
			)
		},
	)
}
