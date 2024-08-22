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

package hakesstoreutil

import (
	"fmt"
	"log"
	"os"

	"hakes-store/cloud"
	io "hakes-store/hakes-store/io"
	kv "hakes-store/hakes-store/kv"
	"hakes-store/hakes-store/table"
	cli "hakes-store/store-daemon-cli"

	"github.com/dgraph-io/badger/v3/options"
)

func PrepareHakesKVOptFromLsmConfig(cfg *LSMConfig) kv.Options {
	opts := kv.DefaultOptions()
	opts.SyncWrites = cfg.SyncWrites
	// default info level logging
	loggingLevel := kv.INFO
	switch cfg.LogType {
	case "debug":
		loggingLevel = kv.DEBUG
	case "warning":
		loggingLevel = kv.WARNING
	case "error":
		loggingLevel = kv.ERROR
	}
	if len(cfg.LogPath) > 0 {
		opts = opts.SetDefaultLoggerToFile(cfg.LogPath, loggingLevel)
	} else {
		opts = opts.WithLoggingLevel(loggingLevel)
	}

	// default snappy
	switch cfg.CompressionType {
	case "None":
		opts = opts.WithCompression(options.None)
	case "ZSTD":
		opts = opts.WithCompression(options.ZSTD)
	}
	if cfg.MemtableSize > 0 {
		opts.MemTableSize = int64(cfg.MemtableSize) << 20
	}
	if cfg.BasetableSize > 0 {
		opts.BaseTableSize = int64(cfg.BasetableSize) << 20
	}
	if cfg.BaselevelSize > 0 {
		opts.BaseLevelSize = int64(cfg.BaselevelSize) << 20
	}
	if cfg.LevelsizeMultiplier > 0 {
		opts.LevelSizeMultiplier = cfg.LevelsizeMultiplier
	}
	if cfg.TablesizeMultiplier > 0 {
		opts.TableSizeMultiplier = cfg.TablesizeMultiplier
	}
	if cfg.Maxlevel > 0 {
		opts.MaxLevels = cfg.Maxlevel
	}
	if cfg.ValueThreshold > 0 {
		opts.ValueThreshold = int64(cfg.ValueThreshold) << 10
	}
	if cfg.NumMemtable > 0 {
		opts.NumMemtables = cfg.NumMemtable
	}
	if cfg.BlockSize > 0 {
		opts.BlockSize = cfg.BlockSize << 10
	}
	if cfg.BloomFPR > 0 {
		opts.BloomFalsePositive = cfg.BloomFPR
	}
	if cfg.BlockCacheSize > 0 {
		opts.BlockCacheSize = int64(cfg.BlockCacheSize) << 20
	}
	if cfg.IndexCacheSize > 0 {
		opts.IndexCacheSize = int64(cfg.IndexCacheSize) << 20
	}
	if cfg.NumLevelZeroTable > 0 {
		opts.NumLevelZeroTables = cfg.NumLevelZeroTable
	}
	if cfg.NumLZTStall > 0 {
		opts.NumLevelZeroTablesStall = cfg.NumLZTStall
	}
	if cfg.NumCompactor > 0 {
		opts.NumCompactors = cfg.NumCompactor
	}
	if cfg.LMaxCompaction {
		opts.LmaxCompaction = true
	}
	if cfg.DisableWAL {
		opts.DisableWAL = true
	}
	if cfg.LogCompactionStats {
		opts.AlwaysLogCompactionStats = true
	}
	if cfg.LogFlushStats {
		opts.AlwaysLogFlushStats = true
	}
	switch cfg.PrefetchMode {
	case "Sync":
		opts = opts.WithCompactionPrefetch(table.SyncPrefetch, cfg.PrefetchSize<<10)
	case "Async":
		opts = opts.WithCompactionPrefetch(table.AsyncPrefetch, cfg.PrefetchSize<<10)
	case "None":
		fallthrough
	default:
		opts = opts.WithCompactionPrefetch(table.NoPrefetch, 0)
	}
	return opts
}

func NewSstCacheFromConfig(cfg SstCacheConfig) io.SstCache {
	switch cfg.Type {
	case "local":
		log.Printf("local sst cache at %s of size %d MB used", cfg.Css.Path, cfg.Capacity)
		return io.NewLocalSstCache(uint64(cfg.Capacity)<<20, NewCSSCliFromConfig("", &cfg.Css))
	case "remote":
		ndc := cli.NewStoreDaemonCli(fmt.Sprintf("%s:%d", cfg.RemoteIP, cfg.RemotePort))
		if err := ndc.Connect(); err != nil {
			log.Println("failed to connect to remote sst cache (store-daemon)")
			return nil
		}
		log.Printf("connected to remote sst cache (store-daemon) on port %d", cfg.RemotePort)
		return io.NewRemoteSstCache(NewCSSCliFromConfig("", &cfg.Css), ndc)
	default:
		log.Printf("unknown mode: %s, sst cache disabled", cfg.Type)
		return nil
	}
}

func NewCompactionSchedulerFromConfig(name string, cfg *RemoteCompactionConfig) (kv.CompactionScheduler, func()) {
	prepareNdc := func() *cli.StoreDaemonCli {
		ip := "127.0.0.1"
		if len(cfg.StoreDaemonIP) > 0 {
			log.Printf("addr: %s", cfg.StoreDaemonIP)
			ip = cfg.StoreDaemonIP
		}
		ndc := cli.NewStoreDaemonCli(fmt.Sprintf("%s:%d", ip, cfg.StoreDaemonPort))
		if err := ndc.Connect(); err != nil {
			log.Println("fail to connect to store-daemon")
			return nil
		}
		return ndc
	}
	if !cfg.Use {
		log.Println("remote compaction scheduling disabled")
		return nil, nil
	}
	switch cfg.Type {
	case "always":
		if ndc := prepareNdc(); ndc == nil {
			log.Printf("no store-daemon connection for always remote scheduler: falling back to local only")
			return nil, nil
		} else {
			log.Printf("connected to store-daemon on port %d, always remote policy used", cfg.StoreDaemonPort)
			return kv.NewAlwaysCheckRemoteScheduler(ndc), func() { ndc.Close() }
		}
	default:
		log.Println("unknown compaction scheduler type: falling back to local only")
		return nil, nil
	}
}

func NewCSSCliFromConfig(name string, cfg *CSSConfig) io.CSSCli {
	prepareLF := func() io.CSSCli {
		if err := os.MkdirAll(cfg.Path, 0755); err != nil {
			log.Fatal("cannot create css path")
			panic(err)
		}
		cli := &io.FSCli{}
		cli.Connect(cfg.Path)
		return cli
	}

	prepareS3 := func() io.CSSCli {
		cli := cloud.NewS3CCS()
		if err := cli.Connect(cfg.Path); err != nil {
			panic(err)
		}
		return cli
	}
	switch cfg.Type {
	case "local":
		return prepareLF()
	case "s3":
		return prepareS3()
	default:
		return prepareLF()
	}
}

func NewDLogHandlerFromConfig(name string, estLogSize uint64, cfg *DLogHandlerConfig) io.DLogHandler {
	prepareFDlh := func() io.DLogHandler {
		if err := os.MkdirAll(cfg.Path, 0755); err != nil {
			log.Fatal("cannot create css path")
			panic(err)
		}
		cli := &io.FDLogHandler{}
		cli.Connect(cfg.Path)
		return cli
	}

	prepareKinesis := func() io.DLogHandler {
		cli := cloud.NewKinesisLogHandler(cfg.Path, estLogSize)
		cli.Connect(cfg.Path)
		return cli
	}

	switch cfg.Type {
	case "local":
		return prepareFDlh()
	case "kinesis":
		return prepareKinesis()
	default:
		return prepareFDlh()
	}
}

func NewManifestHandlerFromConfig(name string, cfg *ManifestHandlerConfig) kv.ManifestHandler {
	prepareSeqFileMh := func() kv.ManifestHandler {
		if err := os.MkdirAll(cfg.Path, 0755); err != nil {
			log.Fatal("cannot create css path")
		}
		return kv.NewSeqLFManifestHandler(cfg.Path)
	}
	prepareS3Mh := func() kv.ManifestHandler {
		s3c, err := cloud.ConnectToS3()
		if err != nil {
			log.Fatal("cannot connect to s3 during manifest handler init")
		}
		return cloud.NewSeqS3ManifestHandler(cfg.Path, s3c)
	}

	switch cfg.Type {
	case "local":
		return prepareSeqFileMh()
	case "s3":
		return prepareS3Mh()
	default:
		return prepareSeqFileMh()
	}
}

// return the option and a cleaner callback
func PrepareHakesKVOptFromConfig(cfg *HakesKVConfig) (*kv.Options, func()) {
	log.Println("prepare hakes kv options")
	cssCli := NewCSSCliFromConfig(cfg.Name, &cfg.Css)
	dlh := NewDLogHandlerFromConfig(cfg.Name, uint64(cfg.Lsm.MemtableSize<<20), &cfg.Dlh)
	mh := NewManifestHandlerFromConfig(cfg.Name, &cfg.Mh)
	rcs, rcsCleanCB := NewCompactionSchedulerFromConfig(cfg.Name, &cfg.Rc)
	sc := NewSstCacheFromConfig(cfg.Sc)
	opts := PrepareHakesKVOptFromLsmConfig(&cfg.Lsm).WithCSSCli(cssCli).WithDLogHandler(dlh).WithManifestHandler(mh).WithCompactionScheduler(rcs)
	opts = opts.WithSstCache(sc, kv.NewLevelSstCachePolicy(opts.MaxLevels, opts.LevelSizeMultiplier))
	return &opts, func() {
		if rcsCleanCB != nil {
			rcsCleanCB()
		}
		mh.Close()
		cssCli.Disconnect()
		dlh.Disconnect()
	}
}
