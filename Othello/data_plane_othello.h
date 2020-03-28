#pragma once

#include "control_plane_othello.h"

using namespace std;

template<class K, bool l2, uint8_t DL>
class OthelloFilterControlPlane;

/**
 * Describes the data structure *l-Othello*. It classifies keys of *keyType* into *2^L* classes.
 * The array are all stored in an array of uint64_t. There are actually m_a+m_b cells in this array, each of length L.
 * \note Be VERY careful!!!! valueType must be some kind of int with no more than 8 bytes' length
 */
template<class K, class V, uint8_t L = sizeof(V) * 8, uint8_t DL = 0>
class DataPlaneOthello {
  template<class K1, bool l2, uint8_t DL1>
  friend
  class OthelloFilterControlPlane;

public:
  //*******builtin values
  const static int MAX_REHASH = 50; //!< Maximum number of rehash tries before report an error. If this limit is reached, Othello build fails.
  const static int VDL = L + DL;
  static_assert(
    VDL <= 64, "Value is too long. You should consider another solution to avoid space waste. ");
  const static uint64_t VDEMASK = ~(uint64_t(-1) << VDL);   // lower VDL bits are 1, others are 0
  const static uint64_t DEMASK = ~(uint64_t(-1) << DL);   // lower DL bits are 1, others are 0
  const static uint64_t VMASK = ~(uint64_t(-1) << L);   // lower L bits are 1, others are 0
  const static uint64_t VDMASK = (VDEMASK << 1) & VDEMASK; // [1, VDL) bits are 1
  
  //****************************************
  //*************DATA Plane
  //****************************************
  vector<uint64_t> mem{};        // memory space for array A and array B. All elements are stored compactly into consecutive uint64_t
  uint32_t ma = 0;               // number of elements of array A
  uint32_t mb = 0;               // number of elements of array B
  Hasher64<K> hab;          // hash function Ha
  Hasher32<K> hd;
  
  vector<uint8_t> lock = vector<uint8_t>(8192, 0);
  
  inline uint32_t multiply_high_u32(uint32_t x, uint32_t y) const {
    return (uint32_t) (((uint64_t) x * (uint64_t) y) >> 32);
  }
  
  vector<uint8_t> versions;
  
  inline uint64_t fast_map_to_A(uint32_t x) const {
    // Map x (uniform in 2^64) to the range [0, num_buckets_ -1]
    // using Lemire's alternative to modulo reduction:
    // http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    // Instead of x % N, use (x * N) >> 64.
    return multiply_high_u32(x, ma);
  }
  
  inline uint64_t fast_map_to_B(uint32_t x) const {
    return multiply_high_u32(x, mb);
  }
  
  /// \param k
  /// \return ma + the index of k into array B
  inline void getIndices(const K &k, uint32_t &aInd, uint32_t &bInd) const {
    uint64_t hash = hab(k);
    bInd = fast_map_to_B(hash >> 32) + ma;
    aInd = fast_map_to_A(hash);
  }
  
  /// Set the index-th element to be value. if the index > ma, it is the (index - ma)-th element in array B
  /// \param index in array A or array B
  /// \param value
  inline void memSet(uint32_t index, uint64_t value) {
    if (VDL == 0) return;
    
    lock[index & 8191]++;
    
    uint64_t v = uint64_t(value) & VDEMASK;
    
    uint64_t i = (uint64_t) index * VDL;
    uint32_t start = i / 64;
    uint8_t offset = uint8_t(i % 64);
    char left = char(offset + VDL - 64);
    
    uint64_t mask = ~(VDEMASK << offset); // [offset, offset + VDL) should be 0, and others are 1
    
    mem[start] &= mask;
    mem[start] |= v << offset;
    
    if (left > 0) {
      mask = uint64_t(-1) << left;     // lower left bits should be 0, and others are 1
      mem[start + 1] &= mask;
      mem[start + 1] |= v >> (VDL - left);
    }
    
    lock[index & 8191]++;
  }
  
  /// \param index in array A or array B
  /// \return the index-th element. if the index > ma, it is the (index - ma)-th element in array B
  inline uint64_t memGet(uint32_t index) const {
    uint64_t i = (uint64_t) index * VDL;
    uint32_t start = i / 64;
    uint8_t offset = uint8_t(i % 64);
    
    char left = char(offset + VDL - 64);
    left = char(left < 0 ? 0 : left);
    
    uint64_t mask = ~(uint64_t(-1)
      << (VDL - left));     // lower VDL-left bits should be 1, and others are 0
    uint64_t result = (mem[start] >> offset) & mask;
    
    if (left > 0) {
      mask = ~(uint64_t(-1) << left);     // lower left bits should be 1, and others are 0
      result |= (mem[start + 1] & mask) << (VDL - left);
    }
    
    return result;
  }
  
  inline void memValueSet(uint32_t index, uint64_t value) {
    if (L == 0) return;
    
    lock[index & 8191]++;
    COMPILER_BARRIER();
    
    uint64_t v = uint64_t(value) & VMASK;
    
    uint64_t i = (uint64_t) index * VDL + DL;
    uint32_t start = i / 64;
    uint8_t offset = uint8_t(i % 64);
    char left = char(offset + L - 64);
    
    uint64_t mask = ~(VMASK << offset); // [offset, offset + L) should be 0, and others are 1
    
    mem[start] &= mask;
    mem[start] |= v << offset;
    
    if (left > 0) {
      mask = uint64_t(-1) << left;     // lower left bits should be 0, and others are 1
      mem[start + 1] &= mask;
      mem[start + 1] |= v >> (L - left);
    }
    
    COMPILER_BARRIER();
    lock[index & 8191]++;
  }
  
  inline uint64_t memValueGet(uint32_t index) const {
    if (L == 0) return 0;
    
    uint64_t i = (uint64_t) index * VDL + DL;
    uint32_t start = i / 64;
    uint8_t offset = uint8_t(i % 64);
    char left = char(offset + L - 64);
    left = char(left < 0 ? 0 : left);
    
    uint64_t mask = ~(uint64_t(-1)
      << (L - left));     // lower L-left bits should be 1, and others are 0
    uint64_t result = (mem[start] >> offset) & mask;
    
    if (left > 0) {
      mask = ~(uint64_t(-1) << left);     // lower left bits should be 1, and others are 0
      result |= (mem[start + 1] & mask) << (L - left);
    }
    
    return result;
  }
  
  template<bool keepDigest = false>
  inline void fillSingle(uint32_t valueToFill, uint32_t nodeToFill) {
    if (keepDigest) {
      memValueSet(nodeToFill, valueToFill);
    } else {
      memSet(nodeToFill, valueToFill);
    }
  }
  
  inline void setTaken(uint32_t nodeIndex) {
    if (DL) {
      memSet(nodeIndex, memGet(nodeIndex) | 1);
    }
  }
  
  inline void setEmpty(uint32_t nodeIndex) {
    if (DL) {
      memSet(nodeIndex, memGet(nodeIndex) & ~uint64_t(1));
    }
  }
  
  template<bool keepDigest = false>
  /// fix the value and index at single node by xoring x
  /// \param x the xor'ed number
  inline void fixSingle(uint32_t nodeToFix, uint64_t x) {
    if (keepDigest) {
      uint64_t valueToFill = x ^memValueGet(nodeToFix);
      memValueSet(nodeToFix, valueToFill);
    } else {
      uint64_t valueToFill = x ^memGet(nodeToFix);
      memSet(nodeToFix, valueToFill);
    }
  }
  
  /// Fix the values of a connected tree starting at the root node and avoid searching keyId
  /// Assume:
  /// 1. the value of root is not properly set before the function call
  /// 2. the values are in the value array
  /// 3. the root is always from array A
  /// Side effect: all node in this tree is set and if updateToFilled
  inline void fixHalfTreeByConnectedComponent(vector<uint32_t> indices, uint32_t xorTemplate) {
    for (uint32_t index: indices) {
      fixSingle<true>(index, xorTemplate);
    }
  }

public:
  /// \param k
  /// \param v the lookup value for k
  /// \return the lookup is successfully passed the digest match, but it does not mean the key is really a member
  inline bool lookUp(const K &k, V &v) const {
    uint32_t ha, hb;
    getIndices(k, ha, hb);
    
    while (true) {
      uint8_t va1 = lock[ha & 8191], vb1 = lock[hb & 8191];
      COMPILER_BARRIER();
      
      if (va1 % 2 == 1 || vb1 % 2 == 1) continue;
      
      uint64_t aa = memGet(ha);
      uint64_t bb = memGet(hb);
      
      COMPILER_BARRIER();
      uint8_t va2 = lock[ha & 8191], vb2 = lock[hb & 8191];
      
      if (va1 != va2 || vb1 != vb2) continue;
      
      ////printf("%llx   [%x] %x ^ [%x] %x = %x\n", k,ha,aa&LMASK,hb,bb&LMASK,(aa^bb)&LMASK);
      uint64_t vd = aa ^bb;
      
      v = vd >> DL;  // extract correct v
      
      if (DL == 0) return true;      // no filter features
      
      if ((aa & 1) == 0 || (bb & 1) == 0) {
        return false;
      }     // with filter features, then the last bit must be 1
      
      if (DL == 1) return true;  // shortcut for one bit digest
      
      uint32_t digest = uint32_t(vd & DEMASK);
      return (digest | 1) == ((hd(k) & DEMASK) | 1);        // ignore the last bit
    }
  }
  
  inline V lookUp(const K &k) const {
    V result;
    bool success = lookUp(k, result);
    if (success) return result;
    
    throw runtime_error("No matched key! ");
  }

public:
  inline DataPlaneOthello() = default;
  
  template<bool maintainDP, bool maintainDisjointSet, bool randomized>
  explicit
  DataPlaneOthello(const ControlPlaneOthello<K, V, L, DL, maintainDP, maintainDisjointSet, randomized> &cp) {
    fullSync(cp);
    
    #ifndef NDEBUG
    for (int i = 0; i < cp.keyCnt; ++i) {
      auto &k = cp.keys[i];
      V out;
      assert(cp.lookUp(k, out) && (out == lookUp(k)));
    }
    #endif
    
    versions.resize(ma + mb);
  }
  
  template<bool maintainDisjointSet, bool randomized>
  void fullSync(const ControlPlaneOthello<K, V, L, DL, true, maintainDisjointSet, randomized> &cp) {
    this->ma = cp.ma;
    this->mb = cp.mb;
    this->hab = cp.hab;
    this->mem = cp.mem;
    this->hd = cp.hd;
  }
  
  template<bool maintainDisjointSet, bool randomized>
  void
  fullSync(const ControlPlaneOthello<K, V, L, DL, false, maintainDisjointSet, randomized> &cp) {
    (const_cast<ControlPlaneOthello<K, V, L, DL, false, maintainDisjointSet, randomized> &>(cp)).prepareDP();
    
    this->ma = cp.ma;
    this->mb = cp.mb;
    this->hab = cp.hab;
    this->hd = cp.hd;
    this->mem = cp.mem;
  }
  
  virtual uint64_t getMemoryCost() const {
    return mem.size() * sizeof(mem[0]);
  }
};

template<class K, class V, uint8_t L = sizeof(V) * 8, uint8_t DL = 6>
class OthelloWithFilter : public DataPlaneOthello<K, V, L, DL> {
};
