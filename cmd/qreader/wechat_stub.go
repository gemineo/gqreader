//go:build !wechat

package main

import gqreader "github.com/gemineo/gqreader"

// appendWeChatFallback is a no-op when built without -tags wechat.
// To enable WeChatQR: build with -tags wechat and provide --wechat-models.
func appendWeChatFallback(config gqreader.Config, _ string) gqreader.Config {
	return config
}
