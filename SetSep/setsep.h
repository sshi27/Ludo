#pragma once

#include "../common.h"
#include "bucket_map_to_group.h"

// 1024 hash values per block: 256 buckets, 2 bit each bucket: 512 bits = 64B
// 64 groups per block: 64 * VALUELENGTH * x for hash, 64*y*VALUELENGTH for bitmaps ==> 8VALUELENGTH(x+y) B
// 64*12*VALUELENGTH  bits for 1024 keys
//

//==> 64K blocks ==> 4M for buckets in total, 0.5(x+y)*VALUELENGTH M for seeds+bitmaps ==>
//==> n/1024 blocks ==> 1.5nVALUELENGTH for map,
//
//Just to confirm, for SetSep, using the recommended (16+8) setting,
//memory cost: (0.5+1.5VALUELENGTH)n bits to store n VALUELENGTH-bit values.
//lookUp time cost: compute hash functions : once [Key universe]->[0..n/4], once [0..255]->[0..63], VALUELENGTH times [Key universe]->[0..15]
//memory access per lookUp: 2+VALUELENGTH times. in number of cache lines:
//           When VALUELENGTH = 1 : 2 with probability 31/32, and 3 with probability 1/32.
//           When VALUELENGTH = 2 : 2
//           When VALUELENGTH = 3 : 2 with probability 15/16, and 3 with probability 1/16.
//           When VALUELENGTH = 4 : 2 with probability 61/64, and 3 with probability 5/64.

template<typename K, typename V, uint8_t VL>
class SetSep {
  static_assert(VL <= 64, "The value is too long, please consider other implementation to save memory");
public:
  typedef unsigned __int128 uint128_t;
  
  static constexpr uint KeysPerBlock = 1024;
  static constexpr uint BucketsPerBlock = 256;
  static constexpr uint GroupsPerBlock = 64;
  
  struct Group {
    uint16_t seeds[VL];
    uint8_t bitmaps[VL];
  };
  
  struct Block {
    uint8_t bucketMap[64];      // for 256 buckets, 2 bits for one bucket.  64 * 8 = 256 * 2 = 512
    Group groups[64];
    
    inline void setBucketMap(uint bidx, uint8_t to) {
      bucketMap[bidx / 4] &= ~(3U << ((bidx % 4) * 2U));
      bucketMap[bidx / 4] |= (to & 3U) << ((bidx % 4) * 2U);
    }
    
    inline uint8_t getBucketMap(uint bidx) const {
      return (bucketMap[bidx / 4] >> ((bidx % 4) * 2U)) & 3U;
    }
  };
  
  // DP & CP common states
  unsigned long long int seed = 0x19900111ULL;
  vector<Block> blocks;
  unordered_map<K, V> overflow;
  unordered_map<K, uint32_t> k2i;
  mutex mtx;  // for overflow
  
  // DP & CP common functions
  uint32_t inline getGlobalHash(const K &k0) const {
    uint32_t r = Hasher32<K>(seed)(k0);
    return uint32_t((uint64_t(r) * bucketCount) >> 32);
  }
  
  static inline uint8_t getBit(uint64_t x, uint y) { return (x >> y) & 1U; }
  
  // CP additional states
  uint32_t blockCount;
  uint32_t groupCount;
  uint32_t bucketCount;
  
  vector<vector<uint32_t>> keysInBucket;
  vector<vector<uint32_t>> keysInGroup;
  
  vector<K> keys;  // keep across builds
  vector<V> values;  // keep across builds
  uint32_t minimalKeyCapacity = 0;    // keep across builds
  uint32_t keyCnt = 0;    // keep across builds
  uint32_t keyCntReserve = 0;   // keep across builds
  
  bool compact = true;   // keep across builds
  int threadCnt = 1;
  
  // CP functions
  explicit SetSep(uint32_t keyCapacity = 1, bool compact = true, const vector<K> &keys = vector<K>(),
                  const vector<V> &values = vector<V>())
    : keyCnt(min((uint32_t) min(keys.size(), values.size()), keyCapacity)),
      keys(keys), values(values), minimalKeyCapacity(0), compact(compact), threadCnt(1) {
    // threadCnt(std::thread::hardware_concurrency()) {
    resizeMem(keyCapacity, true);
  }
  
  /// Resize key and value related memory for the Othello to be able to hold keyCnt keys
  /// \param targetCapacity the target capacity
  /// \note Side effect: will change keyCnt, and if hash size is changed, a rebuild is performed
  void resizeMem(uint32_t targetCapacity, bool forceBuild = false) {
    targetCapacity = max(keyCnt, max(targetCapacity, minimalKeyCapacity));
    
    uint32_t nextN, tmp = 1;
    
    while (tmp < targetCapacity) {
      tmp <<= 1U;
    }
    
    if (compact) {
      nextN = targetCapacity;
    } else {
      nextN = tmp;
    }
    
    if ((compact && nextN != keyCntReserve) || (!compact && (nextN > keyCntReserve || nextN < 0.4 * keyCntReserve))) {
      keyCntReserve = max(256U, nextN);
      keys.resize(keyCntReserve);
      values.resize(keyCntReserve);
      
      build();
    } else if (forceBuild) {
      build();
    }
  }
  
  void build() {
    while (!tryBuild());
  }
  
  // lv =
  // 0: fail if group fail
  // 1: fail if block fail
  // 2 or above: if block fail, rebuild
  bool insert(const K &k, const V &v, int lv = 2) {
    if (isMember(k)) {
      updateMapping(k, v);
      return true;
    }
    
    if (keyCnt + 1 > keys.size()) {
      resizeMem(keyCnt + 1);
    }
    
    keys[keyCnt] = k;
    values[keyCnt] = v;
    k2i.insert(make_pair(k, keyCnt));
    
    uint32_t bucketId = getGlobalHash(k);
    keysInBucket[bucketId].push_back(keyCnt);
    
    uint32_t blockId = bucketId / BucketsPerBlock;
    uint32_t bucketIdx = bucketId & (BucketsPerBlock - 1);
    
    uint8_t sel = blocks[blockId].getBucketMap(bucketIdx);
    
    uint8_t groupIdx = map256_64[bucketIdx][sel];
    keysInGroup[blockId * GroupsPerBlock + groupIdx].push_back(keyCnt);
    
    if (VL <= 2) { // probability of the value is already satisfied is >= 0.25
      V out = v + 1;
      if (lookUp(k, out) && out == v) {
        Counter::count("skip insert");
        keyCnt++;
        return true;
      }
    }
    
    bool lv0 = buildGroup(blockId, groupIdx);
    if (lv0) {
      Counter::count("group-lv");
      keyCnt++;
      return true;
    }
    if (lv <= 0) return false;
    
    bool lv1 = buildBlock(blockId);
    if (lv1) {
      Counter::count("block-lv");
      keyCnt++;
      return true;
    }
    if (lv <= 1) return false;
    
    keyCnt++;
    Counter::count("global-lv");
    build();
    
    return true;
  }
  
  inline void remove(const K &key, uint32_t keyId = uint32_t(-1)) {
    uint32_t bucketId = getGlobalHash(key);
    uint32_t blockId = bucketId / BucketsPerBlock;
    uint32_t bucketIdx = bucketId & (BucketsPerBlock - 1);
    
    uint8_t sel = blocks[blockId].getBucketMap(bucketIdx);
    uint8_t groupIdx = map256_64[bucketIdx][sel];
    
    if (keyId == uint32_t(-1)) {
      typename unordered_map<K, uint32_t>::const_iterator it = k2i.find(key);
      if (it != k2i.end()) {
        keyId = it->second;
      }
    }
    k2i.erase(key);
    
    overflow.erase(key);
    
    // now keyId must be a valid value
    // keyId == -1 means the key is alien.
    if (keyId == uint32_t(-1)) return;
    
    keyCnt--;
    
    vector<uint32_t> &kb = keysInBucket[bucketId];
    unsigned long l1 = kb.size() - 1;
    for (unsigned long i = 0; i < l1; ++i) {
      if (kb[i] == keyId) {
        kb[i] = kb[l1];
        break;
      }
    }
    kb.pop_back();
    
    vector<uint32_t> &kg = keysInGroup[blockId * GroupsPerBlock + groupIdx];
    unsigned long l2 = kg.size() - 1;
    for (unsigned long i = 0; i < l2; ++i) {
      if (kg[i] == keyId) {
        kg[i] = kg[l2];
        break;
      }
    }
    kg.pop_back();
    
    if (keyId != keyCnt) {
      const K &lastKey = keys[keyCnt];
      keys[keyId] = lastKey;
      values[keyId] = values[keyCnt];
      
      uint32_t bucketId = getGlobalHash(lastKey);
      uint32_t blockId = bucketId / BucketsPerBlock;
      uint32_t bucketIdx = bucketId & (BucketsPerBlock - 1);
      
      uint8_t sel = blocks[blockId].getBucketMap(bucketIdx);
      uint8_t groupIdx = map256_64[bucketIdx][sel];
      
      vector<uint32_t> &kb = keysInBucket[bucketId];
      for (unsigned long i = 0; i < kb.size(); ++i) {
        if (kb[i] == keyCnt) {
          kb[i] = keyId;
          break;
        }
      }
      
      vector<uint32_t> &kg = keysInGroup[blockId * GroupsPerBlock + groupIdx];
      for (unsigned long i = 0; i < kg.size(); ++i) {
        if (kg[i] == keyCnt) {
          kg[i] = keyId;
          break;
        }
      }
      
      k2i.erase(lastKey);
      k2i.insert(make_pair(lastKey, keyId));
    }
  }
  
  SetSep<K, V, VL> Copy() const {
    SetSep<K, V, VL> another(*this);
    return another;
  }
  
  inline bool isMember(const K &x) const {
    typename unordered_map<K, uint32_t>::const_iterator it = k2i.find(x);
    return it != k2i.end();
  }
  
  inline void updateMapping(const K &k, V val) {
    if (!isMember(k)) return;
    uint32_t kid = lookUpIndex(k);
    values[kid] = val;
    
    uint32_t bucketId = getGlobalHash(k);
    uint32_t blockId = bucketId / BucketsPerBlock;
    uint32_t bucketIdx = bucketId & (BucketsPerBlock - 1);
    
    uint8_t sel = blocks[blockId].getBucketMap(bucketIdx);
    
    uint8_t groupIdx = map256_64[bucketIdx][sel];
    
    if (!buildGroup(blockId, groupIdx)) { if (!buildBlock(blockId)) build(); }
  }
  
  inline bool lookUpViaIndex(const K &key, V &out) const {
    typename unordered_map<K, uint32_t>::const_iterator it = k2i.find(key);
    if (it != k2i.end()) {
      uint32_t keyId = it->second;
      out = values[keyId];
      return true;
    }
    return false;
  }
  
  inline uint32_t lookUpIndex(const K &key) const {
    typename unordered_map<K, uint32_t>::const_iterator it = k2i.find(key);
    if (it != k2i.end()) {
      return it->second;
    }
    return -1;
  }

private:
  bool tryBuild() {
    seed = ((uint64_t(rand()) << 32U) | rand());
    blocks.clear();
    overflow.clear();
    k2i.clear();
    
    blockCount = (keyCntReserve + KeysPerBlock - 1) / KeysPerBlock;
    
    groupCount = blockCount * GroupsPerBlock;
    bucketCount = blockCount * BucketsPerBlock;
    
    keysInBucket.resize(0);
    keysInBucket.resize(bucketCount);
    keysInGroup.resize(0);
    keysInGroup.resize(groupCount);
    
    blocks.resize(0);
    blocks.resize(blockCount);
    
    for (uint32_t i = 0; i < keyCnt; i++) {
      uint32_t bucketID = getGlobalHash(keys[i]);
      keysInBucket[bucketID].push_back(i);
      k2i.insert(make_pair(keys[i], i));
    }
    
    thread threads[threadCnt];
    
    for (int i = 0; i < threadCnt; ++i) {
      threads[i] = std::thread([](SetSep<K, V, VL> *cp, int id) {
        cp->buildInParallel(id);
      }, this, i);
    }
    
    for (int i = 0; i < threadCnt; ++i) {
      threads[i].join();
    }
    
    bool succ = !overflowTooLarge();
    if (!succ) {
      Counter::count("overflow times");
      Counter::count("overflow keys", overflow.size());
    }
    return succ;
  }
  
  bool overflowTooLarge(int more = 0) const {
    return overflow.size() + more > max(20U, keyCnt / 50U);
  }
  
  void buildInParallel(int id) {
    for (uint32_t blockID = 0; blockID < blockCount; blockID++)
      if (blockID % threadCnt == id) {
        //build block #i, with buckets id #i<<8, ..#i<<8 + 255, groups #i<<6 .. #i..64, bucketChoice[i<<6..i<<6+63](one element for 4 buckets)
        buildBlock(blockID);
      }
  }
  
  bool buildBlock(uint32_t blockID) {
    for (int i = 0; i < GroupsPerBlock; ++i) {
      keysInGroup[blockID * GroupsPerBlock + i].clear();
    }
    
    std::vector<std::pair<uint32_t, uint32_t>> sizeToBucketIdx(BucketsPerBlock);  // key count, bucket index in block
    
    for (int i = 0; i < BucketsPerBlock; i++)
      sizeToBucketIdx[i] = std::make_pair(keysInBucket[blockID * BucketsPerBlock + i].size(), i);
    
    struct ordering {
      bool operator()(std::pair<uint32_t, uint32_t> const &a, std::pair<uint32_t, uint32_t> const &b) {
        return a.first > b.first;
      }
    };
    std::sort(sizeToBucketIdx.begin(), sizeToBucketIdx.end(), ordering());
    
    uint32_t groupKeyCnt[GroupsPerBlock];
    memset(groupKeyCnt, 0, sizeof(groupKeyCnt));
    
    for (int i = 0; i < BucketsPerBlock; i++) {
      uint32_t bucketIdx = sizeToBucketIdx[i].second;
      uint32_t start = bucketIdx % 4;
      uint32_t minGroupSize = uint32_t(-1);
      uint8_t selectedId = 255;
      
      for (uint32_t ii = start; ii < start + 4; ii++) {
        uint32_t candidate = map256_64[bucketIdx][ii % 4];
        if (groupKeyCnt[candidate] < minGroupSize) {
          minGroupSize = groupKeyCnt[candidate];
          selectedId = uint8_t(ii % 4);
          if (minGroupSize == 0) break;
        }
      }
      assert(selectedId != 255);
      
      blocks[blockID].setBucketMap(bucketIdx, selectedId);
      
      uint8_t groupIdx = map256_64[bucketIdx][selectedId];
      groupKeyCnt[groupIdx] += sizeToBucketIdx[i].first;
      
      for (uint32_t k : keysInBucket[blockID * BucketsPerBlock + bucketIdx])
        keysInGroup[blockID * GroupsPerBlock + groupIdx].push_back(k);
    }
    
    bool succ = true;
    for (int i = 0; i < GroupsPerBlock && succ; i++) {
      succ &= buildGroup(blockID, i);
    }
    
    return succ;
  }
  
  inline bool buildGroup(uint32_t blockId, uint32_t groupIdx) {
    for (uint8_t i = 0; i < VL; i++) {
      blocks[blockId].groups[groupIdx].seeds[i] = 0;
    }
    
    if (keysInGroup[blockId * GroupsPerBlock + groupIdx].empty()) return true;
    
    vector<K> _keys;
    vector<V> _values;
    
    for (uint32_t kid: keysInGroup[blockId * GroupsPerBlock + groupIdx]) {
      _keys.push_back(keys[kid]);
      _values.push_back(values[kid]);
      overflow.erase(keys[kid]);
    }
    
    bool succ = true;
    for (uint8_t i = 0; i < VL; i++) {
      if (!buildGroupOneBit<V>(blockId, groupIdx, _keys, _values, i)) {
        succ = false;
        break;
      }
    }
    
    if (!succ && !overflowTooLarge(_keys.size())) {
      mtx.lock();
      for (int ii = 0; ii < _keys.size(); ii++) {
        overflow.insert(make_pair(_keys[ii], _values[ii]));
      }
      succ = true;
      
      mtx.unlock();
      
      blocks[blockId].groups[groupIdx].seeds[0] = 65535U;
    }
    
    return succ;
  }
  
  template<class T>
  bool
  buildGroupOneBit(uint32_t blockId, uint32_t groupIdx, const vector<K> &_keys, const vector<T> &_values, uint8_t loc) {
    uint32_t seed = 0;
    vector<uint32_t> h_ini;
    vector<uint32_t> h_acc;
    vector<uint32_t> h;
    
    for (int i = 0; i < _keys.size(); i++) {
      uint64_t hash = Hasher64<K>(0xe2211)(_keys[i]);
      h_ini.push_back(uint32_t(hash));
      h_acc.push_back(uint32_t(hash >> 32));
      
      h.push_back(h_ini[i]);
    }
    
    const uint8_t bitMapLen = 8;
    uint8_t bits[bitMapLen];
    *(uint64_t *) (bits) = 0;
    
    do {
      bool succ = true;
      
      uint8_t assigned[bitMapLen];
      *(uint64_t *) (assigned) = 0;
      
      for (int i = 0; i < _keys.size() && succ; i++) {
        uint32_t t = h[i] >> 29;
        if (!assigned[t]) {
          bits[t] = getBit(_values[i], loc);
          assigned[t] = true;
        } else {
          succ &= (getBit(_values[i], loc) == bits[t]);
        }
      }
      
      if (succ) {
        blocks[blockId].groups[groupIdx].seeds[loc] = seed;
        
        uint bitmap = 0;
        for (int i = bitMapLen - 1; i >= 0; i--) {
          bitmap <<= 1U;
          bitmap += bits[i];
        }
        blocks[blockId].groups[groupIdx].bitmaps[loc] = bitmap;
        return true;
      } else {
        seed++;
        for (int i = 0; i < _keys.size(); i++)
          h[i] += h_acc[i];
      }
    } while (seed != 65535U);
    
    return false;
  }
  
  // DP functions
public:
  inline bool lookUp(const K &key, V &out) const {
    uint32_t bucketId = getGlobalHash(key);
    uint32_t blockId = bucketId / BucketsPerBlock;
    uint32_t bucketIdx = bucketId & (BucketsPerBlock - 1);
    
    uint8_t sel = blocks[blockId].getBucketMap(bucketIdx);
    
    uint8_t groupIdx = map256_64[bucketIdx][sel];
    if (blocks[blockId].groups[groupIdx].seeds[0] != 65535U) {
      out = lookUpInGroup(key, blocks[blockId].groups[groupIdx].seeds, blocks[blockId].groups[groupIdx].bitmaps, VL);
    } else {
      typename unordered_map<K, V>::const_iterator it = overflow.find(key);
      if (it == overflow.end()) { return false; }
      else { out = it->second; }
    }
    return true;
  }
  
  template<class T = V>
  inline T lookUpInGroup(const K &key, const uint16_t *seeds, const uint8_t *bitmaps, uint8_t L) const {
    uint64_t hash = Hasher64<K>(0xe2211)(key);
    uint32_t h1 = uint32_t(hash);
    uint32_t h2 = uint32_t(hash >> 32U);
    
    T ans = 0;
    for (int i = L - 1; i >= 0; i--) {
      uint8_t ret = lookUpOneBit(key, seeds[i], bitmaps[i], h1, h2);
      ans <<= 1;
      ans += ret;
    }
    return ans;
  }
  
  inline uint8_t lookUpOneBit(const K &key, uint16_t seed, uint8_t bitmap, uint32_t h1, uint32_t h2) const {
    uint32_t bitmapIndex = (h1 + h2 * seed) >> 29;
    return getBit(bitmap, bitmapIndex);
  }
};
