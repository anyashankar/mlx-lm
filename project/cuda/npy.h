// Minimal .npy reader (format versions 1-3, little-endian, C order only).
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

struct Npy {
  std::string descr;           // e.g. "<f4", "<i4", "<u4"
  std::vector<int64_t> shape;
  std::vector<char> data;
  size_t count() const {
    size_t n = 1;
    for (auto s : shape) n *= (size_t)s;
    return n;
  }
  template <class T> const T* as() const { return reinterpret_cast<const T*>(data.data()); }
};

inline Npy load_npy(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + path);
  char magic[6];
  unsigned char ver[2];
  if (std::fread(magic, 1, 6, f) != 6 || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
    std::fclose(f);
    throw std::runtime_error("not a .npy file: " + path);
  }
  if (std::fread(ver, 1, 2, f) != 2) { std::fclose(f); throw std::runtime_error("truncated " + path); }
  uint32_t hlen = 0;
  if (ver[0] == 1) {
    unsigned char b[2];
    if (std::fread(b, 1, 2, f) != 2) { std::fclose(f); throw std::runtime_error("truncated " + path); }
    hlen = b[0] | (b[1] << 8);
  } else {
    unsigned char b[4];
    if (std::fread(b, 1, 4, f) != 4) { std::fclose(f); throw std::runtime_error("truncated " + path); }
    hlen = b[0] | (b[1] << 8) | (b[2] << 16) | ((uint32_t)b[3] << 24);
  }
  std::string h(hlen, ' ');
  if (std::fread(&h[0], 1, hlen, f) != hlen) { std::fclose(f); throw std::runtime_error("truncated " + path); }

  Npy a;
  auto field = [&](const char* key) {
    size_t p = h.find(key);
    if (p == std::string::npos) throw std::runtime_error(std::string("npy header lacks ") + key + ": " + path);
    return p + std::strlen(key);
  };
  size_t p = field("'descr':");
  size_t q1 = h.find('\'', p), q2 = h.find('\'', q1 + 1);
  a.descr = h.substr(q1 + 1, q2 - q1 - 1);
  p = field("'fortran_order':");
  if (h.compare(h.find_first_not_of(' ', p), 4, "True") == 0) {
    std::fclose(f);
    throw std::runtime_error("fortran-order arrays not supported: " + path);
  }
  p = field("'shape':");
  size_t l = h.find('(', p), r = h.find(')', l);
  std::string dims = h.substr(l + 1, r - l - 1);
  size_t i = 0;
  while (i < dims.size()) {
    while (i < dims.size() && (dims[i] == ' ' || dims[i] == ',')) i++;
    if (i >= dims.size()) break;
    size_t j = i;
    while (j < dims.size() && dims[j] >= '0' && dims[j] <= '9') j++;
    a.shape.push_back(std::stoll(dims.substr(i, j - i)));
    i = j;
  }
  size_t item = 0;
  if (a.descr == "<f4" || a.descr == "<i4" || a.descr == "<u4") item = 4;
  else throw std::runtime_error("unsupported dtype " + a.descr + " in " + path);
  a.data.resize(a.count() * item);
  if (std::fread(a.data.data(), 1, a.data.size(), f) != a.data.size()) {
    std::fclose(f);
    throw std::runtime_error("truncated data in " + path);
  }
  std::fclose(f);
  return a;
}

inline bool file_exists(const std::string& p) {
  FILE* f = std::fopen(p.c_str(), "rb");
  if (f) std::fclose(f);
  return f != nullptr;
}
