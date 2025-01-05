#include "utils/io.h"

#include <cassert>
#include <cstring>

namespace hakes {
/**
 * streamlined functions from index_read.cpp and index_write.cpp
 */

size_t StringIOWriter::operator()(const void* ptr, size_t size, size_t nitems) {
  size_t bytes = size * nitems;
  if (bytes > 0) {
    size_t o = data.size();
    data.resize(o + bytes);
    memcpy(&data[o], ptr, size * nitems);
  }
  return nitems;
}

size_t StringIOReader::operator()(void* ptr, size_t size, size_t nitems) {
  if (rp >= data_len) return 0;
  size_t nremain = (data_len - rp) / size;
  if (nremain < nitems) nitems = nremain;
  if (size * nitems > 0) {
    memcpy(ptr, data + rp, size * nitems);
    rp += size * nitems;
  }
  return nitems;
}

FileIOReader::FileIOReader(const char* fname) {
  name = fname;
  f = fopen(fname, "rb");
  assert(f);
  need_close = true;
}

FileIOReader::~FileIOReader() {
  if (need_close) {
    int ret = fclose(f);
    assert(ret == 0);
  }
}

size_t FileIOReader::operator()(void* ptr, size_t size, size_t nitems) {
  return fread(ptr, size, nitems, f);
}

FileIOWriter::FileIOWriter(const char* fname) {
  name = fname;
  f = fopen(fname, "wb");
  assert(f);
  need_close = true;
}

FileIOWriter::~FileIOWriter() {
  if (need_close) {
    int ret = fclose(f);
    if (ret != 0) {
      // we cannot raise and exception in the destructor
      fprintf(stderr, "file %s close error: %s", name.c_str(), strerror(errno));
    }
  }
}

size_t FileIOWriter::operator()(const void* ptr, size_t size, size_t nitems) {
  return fwrite(ptr, size, nitems, f);
}

}  // namespace hakes
