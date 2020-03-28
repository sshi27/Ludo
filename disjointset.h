#pragma once

#include <vector>
#include <cinttypes>

using namespace std;
/*! \file disjointset.h
 *  Disjoint Set data structure.
 */
/*!
 * \brief Disjoint Set data structure. Helps to test the acyclicity of the graph during construction.
 * */
class DisjointSet {
  vector<uint32_t> mem;
public:
  inline void __set(uint32_t i, uint32_t parent) {
    mem[i] = parent;
  }
  
  uint32_t representative(uint32_t i) {
    if (mem[i] == uint32_t(-1)) {
      return mem[i] = i;
    } else if (mem[i] != i) {
      return mem[i] = representative(mem[i]);
    } else {
      return i;
    }
  }
  
  inline void merge(uint32_t a, uint32_t b) {
    mem[representative(b)] = representative(a);
  }
  
  inline bool sameSet(uint32_t a, uint32_t b) {
    return representative(a) == representative(b);
  }
  
  inline bool isRoot(uint32_t a) const {
    return (mem[a] == a);
  }
  
  //! re-initilize the disjoint sets.
  inline void reset() {
    memset(&mem[0], -1, mem.size() * sizeof(mem[0]));
  }
  
  //! add new keys, so that the total number of elements equal to n.
  inline void resize(size_t n) {
    mem.resize(n, -1);
  }
  
  DisjointSet() {}
  
  explicit DisjointSet(uint32_t capacity) {
    resize(capacity);
  }
};
