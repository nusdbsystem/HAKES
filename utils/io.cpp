#include "utils/io.h"

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
}  // namespace hakes
