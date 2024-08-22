//
// Copyright (C) 2017 Dgraph Labs, Inc. and Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.26.0
// 	protoc        v3.6.1
// source: hakes-store/kv/proto/manifest.proto

package proto

import (
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

type ManifestChange_Operation int32

const (
	ManifestChange_CREATE ManifestChange_Operation = 0
	ManifestChange_DELETE ManifestChange_Operation = 1
	ManifestChange_MOVE   ManifestChange_Operation = 2
)

// Enum value maps for ManifestChange_Operation.
var (
	ManifestChange_Operation_name = map[int32]string{
		0: "CREATE",
		1: "DELETE",
		2: "MOVE",
	}
	ManifestChange_Operation_value = map[string]int32{
		"CREATE": 0,
		"DELETE": 1,
		"MOVE":   2,
	}
)

func (x ManifestChange_Operation) Enum() *ManifestChange_Operation {
	p := new(ManifestChange_Operation)
	*p = x
	return p
}

func (x ManifestChange_Operation) String() string {
	return protoimpl.X.EnumStringOf(x.Descriptor(), protoreflect.EnumNumber(x))
}

func (ManifestChange_Operation) Descriptor() protoreflect.EnumDescriptor {
	return file_hakes_store_kv_proto_manifest_proto_enumTypes[0].Descriptor()
}

func (ManifestChange_Operation) Type() protoreflect.EnumType {
	return &file_hakes_store_kv_proto_manifest_proto_enumTypes[0]
}

func (x ManifestChange_Operation) Number() protoreflect.EnumNumber {
	return protoreflect.EnumNumber(x)
}

// Deprecated: Use ManifestChange_Operation.Descriptor instead.
func (ManifestChange_Operation) EnumDescriptor() ([]byte, []int) {
	return file_hakes_store_kv_proto_manifest_proto_rawDescGZIP(), []int{1, 0}
}

type ManifestChangeSet struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// A set of changes that are applied atomically.
	Changes []*ManifestChange `protobuf:"bytes,1,rep,name=changes,proto3" json:"changes,omitempty"`
}

func (x *ManifestChangeSet) Reset() {
	*x = ManifestChangeSet{}
	if protoimpl.UnsafeEnabled {
		mi := &file_hakes_store_kv_proto_manifest_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ManifestChangeSet) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ManifestChangeSet) ProtoMessage() {}

func (x *ManifestChangeSet) ProtoReflect() protoreflect.Message {
	mi := &file_hakes_store_kv_proto_manifest_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ManifestChangeSet.ProtoReflect.Descriptor instead.
func (*ManifestChangeSet) Descriptor() ([]byte, []int) {
	return file_hakes_store_kv_proto_manifest_proto_rawDescGZIP(), []int{0}
}

func (x *ManifestChangeSet) GetChanges() []*ManifestChange {
	if x != nil {
		return x.Changes
	}
	return nil
}

type ManifestChange struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Id          string                   `protobuf:"bytes,1,opt,name=Id,proto3" json:"Id,omitempty"` // Table ID.
	Op          ManifestChange_Operation `protobuf:"varint,2,opt,name=Op,proto3,enum=kv.ManifestChange_Operation" json:"Op,omitempty"`
	Level       uint32                   `protobuf:"varint,3,opt,name=Level,proto3" json:"Level,omitempty"`             // Only used for CREATE and MOVE.
	Compression uint32                   `protobuf:"varint,4,opt,name=compression,proto3" json:"compression,omitempty"` // Only used for CREATE Op.
}

func (x *ManifestChange) Reset() {
	*x = ManifestChange{}
	if protoimpl.UnsafeEnabled {
		mi := &file_hakes_store_kv_proto_manifest_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *ManifestChange) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*ManifestChange) ProtoMessage() {}

func (x *ManifestChange) ProtoReflect() protoreflect.Message {
	mi := &file_hakes_store_kv_proto_manifest_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use ManifestChange.ProtoReflect.Descriptor instead.
func (*ManifestChange) Descriptor() ([]byte, []int) {
	return file_hakes_store_kv_proto_manifest_proto_rawDescGZIP(), []int{1}
}

func (x *ManifestChange) GetId() string {
	if x != nil {
		return x.Id
	}
	return ""
}

func (x *ManifestChange) GetOp() ManifestChange_Operation {
	if x != nil {
		return x.Op
	}
	return ManifestChange_CREATE
}

func (x *ManifestChange) GetLevel() uint32 {
	if x != nil {
		return x.Level
	}
	return 0
}

func (x *ManifestChange) GetCompression() uint32 {
	if x != nil {
		return x.Compression
	}
	return 0
}

var File_hakes_store_kv_proto_manifest_proto protoreflect.FileDescriptor

var file_hakes_store_kv_proto_manifest_proto_rawDesc = []byte{
	0x0a, 0x23, 0x68, 0x61, 0x6b, 0x65, 0x73, 0x2d, 0x73, 0x74, 0x6f, 0x72, 0x65, 0x2f, 0x6b, 0x76,
	0x2f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x2f, 0x6d, 0x61, 0x6e, 0x69, 0x66, 0x65, 0x73, 0x74, 0x2e,
	0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x02, 0x6b, 0x76, 0x22, 0x41, 0x0a, 0x11, 0x4d, 0x61, 0x6e,
	0x69, 0x66, 0x65, 0x73, 0x74, 0x43, 0x68, 0x61, 0x6e, 0x67, 0x65, 0x53, 0x65, 0x74, 0x12, 0x2c,
	0x0a, 0x07, 0x63, 0x68, 0x61, 0x6e, 0x67, 0x65, 0x73, 0x18, 0x01, 0x20, 0x03, 0x28, 0x0b, 0x32,
	0x12, 0x2e, 0x6b, 0x76, 0x2e, 0x4d, 0x61, 0x6e, 0x69, 0x66, 0x65, 0x73, 0x74, 0x43, 0x68, 0x61,
	0x6e, 0x67, 0x65, 0x52, 0x07, 0x63, 0x68, 0x61, 0x6e, 0x67, 0x65, 0x73, 0x22, 0xb5, 0x01, 0x0a,
	0x0e, 0x4d, 0x61, 0x6e, 0x69, 0x66, 0x65, 0x73, 0x74, 0x43, 0x68, 0x61, 0x6e, 0x67, 0x65, 0x12,
	0x0e, 0x0a, 0x02, 0x49, 0x64, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x02, 0x49, 0x64, 0x12,
	0x2c, 0x0a, 0x02, 0x4f, 0x70, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0e, 0x32, 0x1c, 0x2e, 0x6b, 0x76,
	0x2e, 0x4d, 0x61, 0x6e, 0x69, 0x66, 0x65, 0x73, 0x74, 0x43, 0x68, 0x61, 0x6e, 0x67, 0x65, 0x2e,
	0x4f, 0x70, 0x65, 0x72, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x52, 0x02, 0x4f, 0x70, 0x12, 0x14, 0x0a,
	0x05, 0x4c, 0x65, 0x76, 0x65, 0x6c, 0x18, 0x03, 0x20, 0x01, 0x28, 0x0d, 0x52, 0x05, 0x4c, 0x65,
	0x76, 0x65, 0x6c, 0x12, 0x20, 0x0a, 0x0b, 0x63, 0x6f, 0x6d, 0x70, 0x72, 0x65, 0x73, 0x73, 0x69,
	0x6f, 0x6e, 0x18, 0x04, 0x20, 0x01, 0x28, 0x0d, 0x52, 0x0b, 0x63, 0x6f, 0x6d, 0x70, 0x72, 0x65,
	0x73, 0x73, 0x69, 0x6f, 0x6e, 0x22, 0x2d, 0x0a, 0x09, 0x4f, 0x70, 0x65, 0x72, 0x61, 0x74, 0x69,
	0x6f, 0x6e, 0x12, 0x0a, 0x0a, 0x06, 0x43, 0x52, 0x45, 0x41, 0x54, 0x45, 0x10, 0x00, 0x12, 0x0a,
	0x0a, 0x06, 0x44, 0x45, 0x4c, 0x45, 0x54, 0x45, 0x10, 0x01, 0x12, 0x08, 0x0a, 0x04, 0x4d, 0x4f,
	0x56, 0x45, 0x10, 0x02, 0x42, 0x22, 0x5a, 0x20, 0x68, 0x61, 0x6b, 0x65, 0x73, 0x2d, 0x73, 0x74,
	0x6f, 0x72, 0x65, 0x2f, 0x68, 0x61, 0x6b, 0x65, 0x73, 0x2d, 0x73, 0x74, 0x6f, 0x72, 0x65, 0x2f,
	0x6b, 0x76, 0x2f, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_hakes_store_kv_proto_manifest_proto_rawDescOnce sync.Once
	file_hakes_store_kv_proto_manifest_proto_rawDescData = file_hakes_store_kv_proto_manifest_proto_rawDesc
)

func file_hakes_store_kv_proto_manifest_proto_rawDescGZIP() []byte {
	file_hakes_store_kv_proto_manifest_proto_rawDescOnce.Do(func() {
		file_hakes_store_kv_proto_manifest_proto_rawDescData = protoimpl.X.CompressGZIP(file_hakes_store_kv_proto_manifest_proto_rawDescData)
	})
	return file_hakes_store_kv_proto_manifest_proto_rawDescData
}

var file_hakes_store_kv_proto_manifest_proto_enumTypes = make([]protoimpl.EnumInfo, 1)
var file_hakes_store_kv_proto_manifest_proto_msgTypes = make([]protoimpl.MessageInfo, 2)
var file_hakes_store_kv_proto_manifest_proto_goTypes = []interface{}{
	(ManifestChange_Operation)(0), // 0: kv.ManifestChange.Operation
	(*ManifestChangeSet)(nil),     // 1: kv.ManifestChangeSet
	(*ManifestChange)(nil),        // 2: kv.ManifestChange
}
var file_hakes_store_kv_proto_manifest_proto_depIdxs = []int32{
	2, // 0: kv.ManifestChangeSet.changes:type_name -> kv.ManifestChange
	0, // 1: kv.ManifestChange.Op:type_name -> kv.ManifestChange.Operation
	2, // [2:2] is the sub-list for method output_type
	2, // [2:2] is the sub-list for method input_type
	2, // [2:2] is the sub-list for extension type_name
	2, // [2:2] is the sub-list for extension extendee
	0, // [0:2] is the sub-list for field type_name
}

func init() { file_hakes_store_kv_proto_manifest_proto_init() }
func file_hakes_store_kv_proto_manifest_proto_init() {
	if File_hakes_store_kv_proto_manifest_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_hakes_store_kv_proto_manifest_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ManifestChangeSet); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_hakes_store_kv_proto_manifest_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*ManifestChange); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_hakes_store_kv_proto_manifest_proto_rawDesc,
			NumEnums:      1,
			NumMessages:   2,
			NumExtensions: 0,
			NumServices:   0,
		},
		GoTypes:           file_hakes_store_kv_proto_manifest_proto_goTypes,
		DependencyIndexes: file_hakes_store_kv_proto_manifest_proto_depIdxs,
		EnumInfos:         file_hakes_store_kv_proto_manifest_proto_enumTypes,
		MessageInfos:      file_hakes_store_kv_proto_manifest_proto_msgTypes,
	}.Build()
	File_hakes_store_kv_proto_manifest_proto = out.File
	file_hakes_store_kv_proto_manifest_proto_rawDesc = nil
	file_hakes_store_kv_proto_manifest_proto_goTypes = nil
	file_hakes_store_kv_proto_manifest_proto_depIdxs = nil
}
