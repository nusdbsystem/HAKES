/*
 * Copyright 2024 The HAKES Authors
 * Copyright 2017 Dgraph Labs, Inc. and Contributors
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

package kv

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/dgraph-io/ristretto/z"
	"github.com/pkg/errors"

	"hakes-store/hakes-store/io"
	"hakes-store/hakes-store/table"

	"github.com/dgraph-io/badger/v3/options"
	"github.com/dgraph-io/badger/v3/y"
)

// Options are params for creating DB object.
//
// This package provides DefaultOptions which contains options that should
// work for most applications. Consider using that as a starting point before
// customizing it for your own needs.
//
// Each option X is documented on the WithX method.
type Options struct {
	SyncWrites        bool
	NumVersionsToKeep int
	Logger            Logger

	// compression
	Compression          options.CompressionType
	ZSTDCompressionLevel int

	MetricsEnabled bool
	NumGoroutines  int

	// Fine tuning options.

	MemTableSize        int64
	BaseTableSize       int64
	BaseLevelSize       int64
	LevelSizeMultiplier int
	TableSizeMultiplier int
	MaxLevels           int

	ValueThreshold     int64
	NumMemtables       int
	BlockSize          int
	BloomFalsePositive float64
	BlockCacheSize     int64
	IndexCacheSize     int64

	NumLevelZeroTables      int
	NumLevelZeroTablesStall int

	ValueLogFileSize   int64
	ValueLogMaxEntries uint32

	NumCompactors    int
	CompactL0OnClose bool
	LmaxCompaction   bool

	// When set, checksum will be validated for each entry read from the value log file.
	VerifyValueChecksum bool

	// Encryption related options.
	EncryptionKey                 []byte        // encryption key
	EncryptionKeyRotationDuration time.Duration // key rotation duration

	// ChecksumVerificationMode decides when db should verify checksums for SSTable blocks.
	ChecksumVerificationMode options.ChecksumVerificationMode

	// DetectConflicts determines whether the transactions would be checked for
	// conflicts. The transactions can be processed at a higher rate when
	// conflict detection is disabled.
	DetectConflicts bool

	// NamespaceOffset specifies the offset from where the next 8 bytes contains the namespace.
	NamespaceOffset int

	// 4. Flags for testing purposes
	// ------------------------------
	maxBatchCount int64 // max entries in batch
	maxBatchSize  int64 // max batch size in bytes

	// hakes addition
	cssSetup             bool
	cssCli               io.CSSCli
	dlsSetup             bool
	dlogHandler          io.DLogHandler
	manifestHandlerSetup bool
	manifestHandler      ManifestHandler

	compscheduler CompactionScheduler

	enableTrivialMove  bool
	DisableWAL         bool
	compactionPrefetch table.PrefetchOptions

	uid string // an uid to differentiate different kv instances

	AlwaysLogCompactionStats bool
	AlwaysLogFlushStats      bool

	sstCachePolicy SstCachePolicy
	sstCache       io.SstCache
}

// DefaultOptions sets a list of recommended options for good performance.
// Feel free to modify these to suit your needs with the WithX methods.
func DefaultOptions() Options {
	return Options{

		MemTableSize:        64 << 20,
		BaseTableSize:       2 << 20,
		BaseLevelSize:       10 << 20,
		TableSizeMultiplier: 2,
		LevelSizeMultiplier: 10,
		MaxLevels:           7,
		NumGoroutines:       8,
		MetricsEnabled:      true,

		NumCompactors:           4, // Run at least 2 compactors. Zero-th compactor prioritizes L0.
		NumLevelZeroTables:      5,
		NumLevelZeroTablesStall: 15,
		NumMemtables:            5,
		BloomFalsePositive:      0.01,
		BlockSize:               4 * 1024,
		SyncWrites:              false,
		NumVersionsToKeep:       1,
		CompactL0OnClose:        false,
		VerifyValueChecksum:     false,
		Compression:             options.Snappy,
		BlockCacheSize:          256 << 20,
		IndexCacheSize:          0,

		ZSTDCompressionLevel: 1,

		ValueLogFileSize: 1<<30 - 1,

		ValueLogMaxEntries: 1000000,

		ValueThreshold: defaultValueThreshold,

		Logger:                        defaultLogger(INFO),
		EncryptionKey:                 []byte{},
		EncryptionKeyRotationDuration: 10 * 24 * time.Hour, // Default 10 days.
		DetectConflicts:               true,
		NamespaceOffset:               -1,

		cssSetup:             false,
		cssCli:               nil,
		dlsSetup:             false,
		dlogHandler:          nil,
		manifestHandlerSetup: false,
		manifestHandler:      nil,

		compscheduler: nil,

		enableTrivialMove:  true,
		DisableWAL:         false,
		compactionPrefetch: table.PrefetchOptions{Type: table.NoPrefetch, PrefetchSize: 0},

		AlwaysLogCompactionStats: false,
		AlwaysLogFlushStats:      false,
	}
}

func buildTableOptions(db *DB) table.Options {
	opt := db.opt
	return table.Options{
		MetricsEnabled:       db.opt.MetricsEnabled,
		TableSize:            uint64(opt.BaseTableSize),
		BlockSize:            opt.BlockSize,
		BloomFalsePositive:   opt.BloomFalsePositive,
		ChkMode:              opt.ChecksumVerificationMode,
		Compression:          opt.Compression,
		ZSTDCompressionLevel: opt.ZSTDCompressionLevel,
		BlockCache:           db.blockCache,
		IndexCache:           db.indexCache,
		AllocPool:            db.allocPool,
	}
}

// After checking, there is no hardware requirement for the value threshold to be capped at 1MB.
// change its name to defaultValueThreshold
const (
	defaultValueThreshold = (1 << 20) // 1 MB
)

// of format compression-type:compression-level
func parseCompression(cStr string) (options.CompressionType, int, error) {
	cStrSplit := strings.Split(cStr, ":")
	cType := cStrSplit[0]
	level := 3

	var err error
	if len(cStrSplit) == 2 {
		level, err = strconv.Atoi(cStrSplit[1])
		y.Check(err)
		if level <= 0 {
			return 0, 0,
				errors.Errorf("ERROR: compression level(%v) must be greater than zero", level)
		}
	} else if len(cStrSplit) > 2 {
		return 0, 0, errors.Errorf("ERROR: Invalid hakeskv.compression argument")
	}
	switch cType {
	case "zstd":
		return options.ZSTD, level, nil
	case "snappy":
		return options.Snappy, 0, nil
	case "none":
		return options.None, 0, nil
	}
	return 0, 0, errors.Errorf("ERROR: compression type (%s) invalid", cType)
}

// generateSuperFlag generates an identical SuperFlag string from the provided Options.
func generateSuperFlag(options Options) string {
	superflag := ""
	v := reflect.ValueOf(&options).Elem()
	optionsStruct := v.Type()
	for i := 0; i < v.NumField(); i++ {
		if field := v.Field(i); field.CanInterface() {
			name := strings.ToLower(optionsStruct.Field(i).Name)
			kind := v.Field(i).Kind()
			switch kind {
			case reflect.Bool:
				superflag += name + "="
				superflag += fmt.Sprintf("%v; ", field.Bool())
			case reflect.Int, reflect.Int64:
				superflag += name + "="
				superflag += fmt.Sprintf("%v; ", field.Int())
			case reflect.Uint32, reflect.Uint64:
				superflag += name + "="
				superflag += fmt.Sprintf("%v; ", field.Uint())
			case reflect.Float64:
				superflag += name + "="
				superflag += fmt.Sprintf("%v; ", field.Float())
			case reflect.String:
				superflag += name + "="
				superflag += fmt.Sprintf("%v; ", field.String())
			default:
				continue
			}
		}
	}
	return superflag
}

// FromSuperFlag fills Options fields for each flag within the superflag. For
// example, replacing the default Options.NumGoroutines:
//
//	options := FromSuperFlag("numgoroutines=4", DefaultOptions(""))
//
// It's important to note that if you pass an empty Options struct, FromSuperFlag
// will not fill it with default values. FromSuperFlag only writes to the fields
// present within the superflag string (case insensitive).
//
// It specially handles compression subflag.
// Valid options are {none,snappy,zstd:<level>}
// Example: compression=zstd:3;
// Unsupported: Options.Logger, Options.EncryptionKey
func (opt Options) FromSuperFlag(superflag string) Options {
	// currentOptions act as a default value for the options superflag.
	currentOptions := generateSuperFlag(opt)
	currentOptions += "compression=;"

	flags := z.NewSuperFlag(superflag).MergeAndCheckDefault(currentOptions)
	v := reflect.ValueOf(&opt).Elem()
	optionsStruct := v.Type()
	for i := 0; i < v.NumField(); i++ {
		// only iterate over exported fields
		if field := v.Field(i); field.CanInterface() {
			// z.SuperFlag stores keys as lowercase, keep everything case
			// insensitive
			name := strings.ToLower(optionsStruct.Field(i).Name)
			if name == "compression" {
				// We will specially handle this later. Skip it here.
				continue
			}
			kind := v.Field(i).Kind()
			switch kind {
			case reflect.Bool:
				field.SetBool(flags.GetBool(name))
			case reflect.Int, reflect.Int64:
				field.SetInt(flags.GetInt64(name))
			case reflect.Uint32, reflect.Uint64:
				field.SetUint(uint64(flags.GetUint64(name)))
			case reflect.Float64:
				field.SetFloat(flags.GetFloat64(name))
			case reflect.String:
				field.SetString(flags.GetString(name))
			}
		}
	}

	// Only update the options for special flags that were present in the input superflag.
	inputFlag := z.NewSuperFlag(superflag)
	if inputFlag.Has("compression") {
		ctype, clevel, err := parseCompression(flags.GetString("compression"))
		switch err {
		case nil:
			opt.Compression = ctype
			opt.ZSTDCompressionLevel = clevel
		default:
			ctype = options.CompressionType(flags.GetUint32("compression"))
			y.AssertTruef(ctype <= 2, "ERROR: Invalid format or compression type. Got: %s",
				flags.GetString("compression"))
			opt.Compression = ctype
		}
	}

	return opt
}

// The default value of SyncWrites is false.
func (opt Options) WithSyncWrites(val bool) Options {
	opt.SyncWrites = val
	return opt
}

// The default value of NumVersionsToKeep is 1.
func (opt Options) WithNumVersionsToKeep(val int) Options {
	opt.NumVersionsToKeep = val
	return opt
}

// WithNumGoroutines sets the number of goroutines to be used in Stream.
//
// The default value of NumGoroutines is 8.
func (opt Options) WithNumGoroutines(val int) Options {
	opt.NumGoroutines = val
	return opt
}

// Default value is set to true
func (opt Options) WithMetricsEnabled(val bool) Options {
	opt.MetricsEnabled = val
	return opt
}

// WithLogger returns a new Options value with Logger set to the given value.
// The default value of Logger writes to stderr using the log package from the Go standard library.
func (opt Options) SetDefaultLoggerToFile(path string, val loggingLevel) Options {
	opt.Logger = newLogger(path, val)
	return opt
}

// WithLogger returns a new Options value with Logger set to the given value.
// The default value of Logger writes to stderr using the log package from the Go standard library.
func (opt Options) WithLogger(val Logger) Options {
	opt.Logger = val
	return opt
}

// WithLoggingLevel returns a new Options value with logging level of the
// default logger set to the given value.
// LoggingLevel sets the level of logging. It should be one of DEBUG, INFO,
// WARNING or ERROR levels.
//
// The default value of LoggingLevel is INFO.
func (opt Options) WithLoggingLevel(val loggingLevel) Options {
	opt.Logger = defaultLogger(val)
	return opt
}

// WithBaseTableSize returns a new Options value with MaxTableSize set to the given value.
//
// BaseTableSize sets the maximum size in bytes for LSM table or file in the base level.
//
// The default value of BaseTableSize is 2MB.
func (opt Options) WithBaseTableSize(val int64) Options {
	opt.BaseTableSize = val
	return opt
}

// WithLevelSizeMultiplier returns a new Options value with LevelSizeMultiplier set to the given
// value.
//
// LevelSizeMultiplier sets the ratio between the maximum sizes of contiguous levels in the LSM.
// Once a level grows to be larger than this ratio allowed, the compaction process will be
//
//	triggered.
//
// The default value of LevelSizeMultiplier is 10.
func (opt Options) WithLevelSizeMultiplier(val int) Options {
	opt.LevelSizeMultiplier = val
	return opt
}

// WithMaxLevels returns a new Options value with MaxLevels set to the given value.
//
// Maximum number of levels of compaction allowed in the LSM.
//
// The default value of MaxLevels is 7.
func (opt Options) WithMaxLevels(val int) Options {
	opt.MaxLevels = val
	return opt
}

// WithValueThreshold returns a new Options value with ValueThreshold set to the given value.
//
// ValueThreshold sets the threshold used to decide whether a value is stored directly in the LSM
// tree or separately in the log value files.
//
// The default value of ValueThreshold is 1 MB, but LSMOnlyOptions sets it to maxValueThreshold.
func (opt Options) WithValueThreshold(val int64) Options {
	opt.ValueThreshold = val
	return opt
}

// WithNumMemtables returns a new Options value with NumMemtables set to the given value.
//
// NumMemtables sets the maximum number of tables to keep in memory before stalling.
//
// The default value of NumMemtables is 5.
func (opt Options) WithNumMemtables(val int) Options {
	opt.NumMemtables = val
	return opt
}

// WithMemTableSize returns a new Options value with MemTableSize set to the given value.
//
// MemTableSize sets the maximum size in bytes for memtable table.
//
// The default value of MemTableSize is 64MB.
func (opt Options) WithMemTableSize(val int64) Options {
	opt.MemTableSize = val
	return opt
}

// WithBloomFalsePositive returns a new Options value with BloomFalsePositive set
// to the given value.
//
// BloomFalsePositive sets the false positive probability of the bloom filter in any SSTable.
// Before reading a key from table, the bloom filter is checked for key existence.
// BloomFalsePositive might impact read performance of DB. Lower BloomFalsePositive value might
// consume more memory.
//
// The default value of BloomFalsePositive is 0.01.
//
// Setting this to 0 disables the bloom filter completely.
func (opt Options) WithBloomFalsePositive(val float64) Options {
	opt.BloomFalsePositive = val
	return opt
}

// WithBlockSize returns a new Options value with BlockSize set to the given value.
//
// BlockSize sets the size of any block in SSTable. SSTable is divided into multiple blocks
// internally. Each block is compressed using prefix diff encoding.
//
// The default value of BlockSize is 4KB.
func (opt Options) WithBlockSize(val int) Options {
	opt.BlockSize = val
	return opt
}

// WithNumLevelZeroTables sets the maximum number of Level 0 tables before compaction starts.
//
// The default value of NumLevelZeroTables is 5.
func (opt Options) WithNumLevelZeroTables(val int) Options {
	opt.NumLevelZeroTables = val
	return opt
}

// WithNumLevelZeroTablesStall sets the number of Level 0 tables that once reached causes the DB to
// stall until compaction succeeds.
//
// The default value of NumLevelZeroTablesStall is 10.
func (opt Options) WithNumLevelZeroTablesStall(val int) Options {
	opt.NumLevelZeroTablesStall = val
	return opt
}

// WithBaseLevelSize sets the maximum size target for the base level.
//
// The default value is 10MB.
func (opt Options) WithBaseLevelSize(val int64) Options {
	opt.BaseLevelSize = val
	return opt
}

// WithValueLogFileSize sets the maximum size of a single value log file.
//
// The default value of ValueLogFileSize is 1GB.
func (opt Options) WithValueLogFileSize(val int64) Options {
	opt.ValueLogFileSize = val
	return opt
}

// WithValueLogMaxEntries sets the maximum number of entries a value log file
// can hold approximately.  A actual size limit of a value log file is the
// minimum of ValueLogFileSize and ValueLogMaxEntries.
//
// The default value of ValueLogMaxEntries is one million (1000000).
func (opt Options) WithValueLogMaxEntries(val uint32) Options {
	opt.ValueLogMaxEntries = val
	return opt
}

// WithNumCompactors sets the number of compaction workers to run concurrently.  Setting this to
// zero stops compactions, which could eventually cause writes to block forever.
//
// The default value of NumCompactors is 2. One is dedicated just for L0 and L1.
func (opt Options) WithNumCompactors(val int) Options {
	opt.NumCompactors = val
	return opt
}

// WithCompactL0OnClose determines whether Level 0 should be compacted before closing the DB.  This
// ensures that both reads and writes are efficient when the DB is opened later.
//
// The default value of CompactL0OnClose is false.
func (opt Options) WithCompactL0OnClose(val bool) Options {
	opt.CompactL0OnClose = val
	return opt
}

// WithCompression is used to enable or disable compression. When compression is enabled, every
// block will be compressed using the specified algorithm.  This option doesn't affect existing
// tables. Only the newly created tables will be compressed.
//
// The default compression algorithm used is zstd when built with Cgo. Without Cgo, the default is
// snappy. Compression is enabled by default.
func (opt Options) WithCompression(cType options.CompressionType) Options {
	opt.Compression = cType
	return opt
}

// WithVerifyValueChecksum is used to set VerifyValueChecksum. When VerifyValueChecksum is set to
// true, checksum will be verified for every entry read from the value log. If the value is stored
// in SST (value size less than value threshold) then the checksum validation will not be done.
//
// The default value of VerifyValueChecksum is False.
func (opt Options) WithVerifyValueChecksum(val bool) Options {
	opt.VerifyValueChecksum = val
	return opt
}

// WithChecksumVerificationMode returns a new Options value with ChecksumVerificationMode set to
// the given value.
//
// ChecksumVerificationMode indicates when the db should verify checksums for SSTable blocks.
//
// The default value of VerifyValueChecksum is options.NoVerification.
func (opt Options) WithChecksumVerificationMode(cvMode options.ChecksumVerificationMode) Options {
	opt.ChecksumVerificationMode = cvMode
	return opt
}

// WithBlockCacheSize returns a new Options value with BlockCacheSize set to the given value.
//
// This value specifies how much data cache should hold in memory. A small size
// of cache means lower memory consumption and lookups/iterations would take
// longer. It is recommended to use a cache if you're using compression or encryption.
// If compression and encryption both are disabled, adding a cache will lead to
// unnecessary overhead which will affect the read performance. Setting size to
// zero disables the cache altogether.
//
// Default value of BlockCacheSize is zero.
func (opt Options) WithBlockCacheSize(size int64) Options {
	opt.BlockCacheSize = size
	return opt
}

func (opt Options) WithZSTDCompressionLevel(cLevel int) Options {
	opt.ZSTDCompressionLevel = cLevel
	return opt
}

// WithIndexCacheSize returns a new Options value with IndexCacheSize set to
// the given value.
//
// This value specifies how much memory should be used by table indices. These
// indices include the block offsets and the bloomfilters. HakesKV uses bloom
// filters to speed up lookups. Each table has its own bloom
// filter and each bloom filter is approximately of 5 MB.
//
// Zero value for IndexCacheSize means all the indices will be kept in
// memory and the cache is disabled.
//
// The default value of IndexCacheSize is 0 which means all indices are kept in
// memory.
func (opt Options) WithIndexCacheSize(size int64) Options {
	opt.IndexCacheSize = size
	return opt
}

// WithNamespaceOffset returns a new Options value with NamespaceOffset set to the given value. DB
// will expect the namespace in each key at the 8 bytes starting from NamespaceOffset. A negative
// value means that namespace is not stored in the key.
//
// The default value for NamespaceOffset is -1.
func (opt Options) WithNamespaceOffset(offset int) Options {
	opt.NamespaceOffset = offset
	return opt
}

func (opt Options) WithCSSCli(c io.CSSCli) Options {
	if c == nil {
		return opt
	}
	opt.cssSetup = true
	opt.cssCli = c
	return opt
}

func (opt Options) WithDLogHandler(h io.DLogHandler) Options {
	if h == nil {
		return opt
	}
	opt.dlsSetup = true
	opt.dlogHandler = h
	return opt
}

func (opt Options) WithManifestHandler(h ManifestHandler) Options {
	if h == nil {
		return opt
	}
	opt.manifestHandlerSetup = true
	opt.manifestHandler = h
	return opt
}

func (opt Options) WithCompactionScheduler(sched CompactionScheduler) Options {
	if sched == nil {
		return opt
	}
	opt.compscheduler = sched
	return opt
}

func (opt Options) WithCompactionPrefetch(t table.CompactionPrefetchType, sz int) Options {
	opt.compactionPrefetch = table.PrefetchOptions{
		Type:         t,
		PrefetchSize: sz,
	}
	return opt
}

func (opt Options) WithSstCache(sc io.SstCache, policy SstCachePolicy) Options {
	if sc == nil {
		return opt
	}
	opt.sstCache = sc
	opt.sstCachePolicy = policy
	return opt
}

func (opt Options) WithUid(uid string) Options {
	opt.uid = uid
	return opt
}
