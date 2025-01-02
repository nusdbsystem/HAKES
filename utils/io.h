#ifndef HAKES_UTILS_IO_H_
#define HAKES_UTILS_IO_H_

#include <string>

namespace hakes {

struct IOReader {
  // name that can be used in error messages
  std::string name;

  // fread. Returns number of items read or 0 in case of EOF.
  virtual size_t operator()(void* ptr, size_t size, size_t nitems) = 0;

  virtual ~IOReader() {}
};

struct IOWriter {
  // name that can be used in error messages
  std::string name;

  // fwrite. Return number of items written
  virtual size_t operator()(const void* ptr, size_t size, size_t nitems) = 0;

  virtual ~IOWriter() noexcept(false) {}
};

struct StringIOReader : IOReader {
  StringIOReader(const char* data, size_t data_len)
      : data(data), data_len(data_len) {}
  const char* data;
  size_t data_len;
  size_t rp = 0;
  size_t operator()(void* ptr, size_t size, size_t nitems) override;
};

struct StringIOWriter : IOWriter {
  std::string data;
  size_t operator()(const void* ptr, size_t size, size_t nitems) override;
};

}  // namespace hakes

#endif  // HAKES_UTILS_IO_H_
