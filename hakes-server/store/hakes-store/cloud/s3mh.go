/*
 * Copyright 2024 The HAKES Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package cloud

import (
	"log"

	"hakes-store/hakes-store/kv"

	"github.com/aws/aws-sdk-go-v2/service/s3/types"
)

// s3 backed manifest handler

var _ kv.ManifestHandler = (*SeqS3ManifestHandler)(nil)

type SeqS3ManifestHandler struct {
	curManifestId int
	bucket        string
	cli           s3svc
	stats         *S3Stats
}

// a client that connected to an s3 bucket is required
// it is recommended to use a separate bucket such that reopen is fast. (minimal effect)
func NewSeqS3ManifestHandler(bucket string, cli *S3C) *SeqS3ManifestHandler {
	return &SeqS3ManifestHandler{
		bucket: bucket,
		cli:    cli,
		stats:  &cli.stats,
	}
}

func (h *SeqS3ManifestHandler) GetStats() string {
	return h.stats.Encode()
}

func (h *SeqS3ManifestHandler) openLatestManifest() ([]byte, int, error) {
	var objects []string
	collectManifestObjects := func(o types.Object) {
		objects = append(objects, *o.Key)
	}
	if err := h.cli.ListObject(h.bucket, collectManifestObjects); err != nil {
		return nil, -1, err
	}
	maxID := -1
	toOpen := ""
	for _, m := range objects {
		if cur, err := kv.GetSeqManifestID(m); err == nil && cur > maxID {
			maxID = cur
			toOpen = m
		}
	}
	if maxID == -1 {
		return nil, -1, kv.ErrEmptyManifest
	}

	// load the manifest
	if contents, err := h.cli.GetObject(h.bucket, toOpen); err != nil {
		log.Printf("failed to download manifest %v", toOpen)
		return nil, -1, err
	} else {
		return contents, maxID, nil
	}
	// s3 object put is atomic. So we should not reach here.
}

func (h *SeqS3ManifestHandler) Reopen() ([]byte, error) {
	latestManifest, manifestID, err := h.openLatestManifest()
	h.curManifestId = manifestID
	return latestManifest, err
}

// input ownership is transferred to manifest handler, the caller no longer modify it afterwards
func (h *SeqS3ManifestHandler) Update(data []byte) error {
	targetName := kv.CreateSeqManifestName(h.curManifestId + 1)
	if err := h.cli.PutObject(h.bucket, targetName, data); err != nil {
		log.Printf("failed to update manifest to s3: %v", targetName)
		return err
	}
	oldName := kv.CreateSeqManifestName(h.curManifestId)
	h.cli.DeleteObject(h.bucket, oldName)
	h.curManifestId++
	return nil
}

// not used
func (h *SeqS3ManifestHandler) Sync() {
	// no-op
}

// not used
func (h *SeqS3ManifestHandler) Close() error {
	return nil
}
