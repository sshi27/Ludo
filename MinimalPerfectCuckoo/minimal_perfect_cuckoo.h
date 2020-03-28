/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 ==============================================================================*/

#pragma once

#include "../CuckooPresized/cuckoo_ht.h"
#include "../hash.h"
#include "../common.h"
#include "../Othello/data_plane_othello.h"

// Class for efficiently storing key->value mappings when the size is
// known in advance and the keys are pre-hashed into uint64s.
// Keys should have "good enough" randomness (be spread across the
// entire 64 bit space).
//
// Important:  Clients wishing to use deterministic keys must
// ensure that their keys fall in the range 0 .. (uint64max-1);
// the table uses 2^64-1 as the "not occupied" flag.
//
// Inserted k must be unique, and there are no update
// or delete functions (until some subsequent use of this table
// requires them).
//
// Threads must synchronize their access to a PresizedHeadlessCuckoo.
//
// The cuckoo hash table is 4-way associative (each "bucket" has 4
// "slots" for key/value entries).  Uses breadth-first-search to find
// a good cuckoo path with less data movement (see
// http://www.cs.cmu.edu/~dga/papers/cuckoo-eurosys14.pdf )

static const uint8_t LocatorSeedLength = 5;
static const uint8_t MaxArrangementSeed = (1 << LocatorSeedLength) - 1;

template<class Key, class Value, uint8_t VL, uint8_t DL>
class DataPlaneMinimalPerfectCuckoo;

struct MPC_PathEntry {
  uint32_t bid: 30;
  uint8_t sid : 2;
  uint8_t newSeed;
  uint8_t s0:2, s1:2, s2:2, s3:2;
  vector<uint32_t> locatorCC;
  // locatorCC contains one end of the modified key, while overflowCC contains both ends
};

template<class Key, class Value, uint8_t VL = sizeof(Value) * 8, uint8_t DL = 0>
class ControlPlaneMinimalPerfectCuckoo {
  static_assert(sizeof(Value) * 8 >= VL + DL);
  static const uint64_t ValueMask = (1ULL << VL) - 1;
  static const uint64_t DigestMask = ((1ULL << DL) - 1) << VL;
  static const uint64_t VDMask = (1ULL << (VL + DL)) - 1;

  // The load factor is chosen slightly conservatively for speed and
  // to avoid the need for a table rebuild on insertion failure.
  // 0.94 is achievable, but 0.85 is faster and keeps the code simple
  // at the cost of a small amount of memory.
  // NOTE:  0 < kLoadFactor <= 1.0
  static constexpr double kLoadFactor = 0.95;

  // Cuckoo insert:  The maximum number of entries to scan should be ~400
  // (Source:  Personal communication with Michael Mitzenmacher;  empirical
  // experiments validate.).  After trying 400 candidate locations, declare
  // the table full - it's probably full of unresolvable cycles.  Less than
  // 400 reduces max occupancy;  much more results in very poor performance
  // around the full point.  For (2,4) a max BFS path len of 5 results in ~682
  // nodes to visit, calculated below, and is a good value.
  static constexpr uint8_t kMaxBFSPathLen = 5;

  static const uint8_t kSlotsPerBucket = 4;   // modification to this value leads to undefined behavior

  // Constants for BFS cuckoo path search:
  // The visited list must be maintained for all but the last level of search
  // in order to trace back the path. The BFS search has two roots
  // and each can go to a total depth (including the root) of 5.
  // The queue must be sized for 4 * \sum_{k=0...4}{(3*kSlotsPerBucket)^k}.
  // The visited queue, however, does not need to hold the deepest level,
  // and so it is sized 4 * \sum{k=0...3}{(3*kSlotsPerBucket)^k}
  static constexpr int calMaxQueueSize() {
    int result = 0;
    int term = 4;
    for (int i = 0; i < kMaxBFSPathLen; ++i) {
      result += term;
      term *= ((2 - 1) * kSlotsPerBucket);
    }
    return result;
  }

  static constexpr int calVisitedListSize() {
    int result = 0;
    int term = 4;
    for (int i = 0; i < kMaxBFSPathLen - 1; ++i) {
      result += term;
      term *= ((2 - 1) * kSlotsPerBucket);
    }
    return result;
  }

  static constexpr int kMaxQueueSize = calMaxQueueSize();
  static constexpr int kVisitedListSize = calVisitedListSize();

  unordered_map<Key, Value> fallback;

public:
  // Buckets are organized with key_types clustered for access speed
  // and for compactness while remaining aligned.
  struct Bucket {
    uint8_t seed = 0;
    uint8_t occupiedMask = 0;
    Key keys[kSlotsPerBucket];
    Value values[kSlotsPerBucket];
  };

  struct CuckooPathEntry {
    uint32_t bucket;
    int depth;
    int parent;      // To index in the visited array.
    int parent_slot; // Which slot in our parent did we come from?  -1 == root.
  };

  // CuckooPathQueue is a trivial circular queue for path entries.
  // The caller is responsible for not inserting more than kMaxQueueSize
  // entries.  Each PresizedHeadlessCuckoo has one (heap-allocated) CuckooPathQueue
  // that it reuses across inserts.
  class CuckooPathQueue {
  public:
    CuckooPathQueue()
      : head_(0), tail_(0) {
    }

    void push_back(CuckooPathEntry e) {
      queue_[tail_] = e;
      tail_ = (tail_ + 1) % kMaxQueueSize;
    }

    CuckooPathEntry pop_front() {
      CuckooPathEntry &e = queue_[head_];
      head_ = (head_ + 1) % kMaxQueueSize;
      return e;
    }

    bool empty() const {
      return head_ == tail_;
    }

    bool full() const {
      return ((tail_ + 1) % kMaxQueueSize) == head_;
    }

    void reset() {
      head_ = tail_ = 0;
    }

  private:
    CuckooPathEntry queue_[kMaxQueueSize];
    int head_;
    int tail_;
  };

  FastHasher64<Key> h;   // the hash value is divided in half for two hashes
  uint32_t entryCount = 0;

  ControlPlaneOthello<Key, uint8_t, 1, 0, true> locator;

  // Set upon initialization: num_entries / kLoadFactor / kSlotsPerBucket.
  std::vector<Bucket> buckets_;

  // Utility function to compute (x * y) >> 64, or "multiply high".
  // On x86-64, this is a single instruction, but not all platforms
  // support the __uint128_t type, so we provide a generic
  // implementation as well.
  inline uint32_t multiply_high_u32(uint32_t x, uint32_t y) const {
    return (uint32_t) (((uint64_t) x * (uint64_t) y) >> 32);
  }

  inline void fast_map_to_buckets(uint64_t x, uint32_t *twoBuckets) const {
    // Map x (uniform in 2^64) to the range [0, num_buckets_ -1]
    // using Lemire's alternative to modulo reduction:
    // http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    // Instead of x % N, use (x * N) >> 64.
    twoBuckets[0] = multiply_high_u32(x, buckets_.size());
    twoBuckets[1] = multiply_high_u32(x >> 32, buckets_.size());
  }

  // The key type is fixed as a pre-hashed key for this specialized use.
  explicit ControlPlaneMinimalPerfectCuckoo(uint32_t num_entries = 64) {
    Clear(num_entries);
  }

  // The key type is fixed as a pre-hashed key for this specialized use.
  template<class V, uint8_t vl, uint8_t dl>
  ControlPlaneMinimalPerfectCuckoo(ControlPlaneMinimalPerfectCuckoo<Key, V, vl, dl> another, unordered_map<V, Value> m)
    : locator(another.locator), entryCount(another.entryCount),
      h(another.h), digestH(another.digestH) {
    Bucket empty_bucket;
    buckets_.clear();
    unsigned long bucketCnt = another.buckets_.size();
    buckets_.resize(bucketCnt, empty_bucket);

    for (uint i = 0; i < bucketCnt; ++i) {
      typename ControlPlaneMinimalPerfectCuckoo<Key, V, vl, dl>::Bucket &ob = another.buckets_[i];
      Bucket &b = buckets_[i];

      b.keys = ob.keys;
      b.seed = ob.seed;
      b.occupiedMask = ob.occupiedMask;

      for (char slot = 0; slot < kSlotsPerBucket; slot++) {
        if (b.occupiedMask & (1 << slot)) {
          b.values[slot] = m[b.values[slot]];
        }
      }
    }
  }

  void Clear(uint32_t num_entries) {
    entryCount = 0;
    h.setSeed(rand());

    num_entries /= kLoadFactor;
    uint32_t num_buckets = (num_entries + kSlotsPerBucket - 1) / kSlotsPerBucket;
    // Very small cuckoo tables don't work, because the probability
    // of having same-bucket hashes is large.  We compromise for those
    // uses by having a larger static starting size.
    num_buckets += 32;

    if (num_buckets >= (1 << 30)) {
      throw runtime_error("Current design only support up to 4 billion key set size! ");
    }

    Bucket empty_bucket;
    buckets_.clear();
    buckets_.resize(num_buckets, empty_bucket);

    locator.clear();
    locator.resizeKey(num_entries, true);
  }

  pair<uint32_t, uint32_t> locate(const Key &k) const {
    uint32_t buckets[2];
    fast_map_to_buckets(h(k), buckets);

    for (uint32_t &b : buckets) {
      const Bucket &bucket = buckets_[b];
      for (uint32_t slot = 0; slot < kSlotsPerBucket; slot++) {
        if ((bucket.occupiedMask & (1 << slot)) && (bucket.keys[slot] == k)) {
          return make_pair(b, slot);
        }
      }
    }

    return make_pair((uint32_t) -1, (uint32_t) -1);
  }

  // Returns collided key if some key collides with the key being inserted;
  // returns null if the table is full and the k-v is inserted to the fallback table; returns &k if inserted successfully.
  const Key *insert(const Key &k, Value v, vector<MPC_PathEntry> *const path = 0) {
    v = v & ValueMask;

    // Merged find and duplicate checking.
    uint32_t target_bucket;
    char target_slot = -1;
    entryCount++;

    uint32_t buckets[2];
    fast_map_to_buckets(h(k), buckets);

    for (char i = 0; i < 2 && (target_slot == -1); ++i) {
      uint32_t bi = buckets[i];
      Bucket &bucket = buckets_[bi];

      for (char slot = 0; slot < kSlotsPerBucket; slot++) {
        if (bucket.occupiedMask & (1 << slot)) {
#ifdef FULL_DEBUG
          if (k == bucket.keys[slot]) { // Duplicates are not allowed.
            entryCount--;
            return &bucket.keys[slot];
          }
#endif
        } else if (target_slot == -1) {
          target_bucket = bi;
          target_slot = slot; // do not break, to go through full duplication test

#ifndef FULL_DEBUG
          break;
#endif
        }
      }
    }

    if (target_slot != -1) {
      Counter::count("Cuckoo direct insert");
      putItem(k, v, target_bucket, target_slot, path);
      return &k;
    }

    // No space, perform cuckooInsert
    Counter::count("Cuckoo cuckoo insert");

    if (CuckooInsert(k, v, path)) {
      return &k;
    } else {
      Counter::count_("Cuckoo insert fail, inserted to fallback");
      fallback.insert(make_pair(k, v));

      return nullptr;
    }
  }

  inline bool remove(const Key &k) {
    uint32_t buckets[2];
    fast_map_to_buckets(h(k), buckets);

    for (uint32_t &b : buckets) {
      Bucket &bucket = buckets_[b];

      if (RemoveInBucket(k, bucket)) {
        entryCount--;
        return true;
      }
    }

    return false;
  }

  // slot :2    bucket:30
  inline uint32_t updateMapping(const Key &k, Value val) {
    uint32_t buckets[2];
    fast_map_to_buckets(h(k), buckets);

    for (uint32_t &b : buckets) {
      Bucket &bucket = buckets_[b];

      for (char slot = 0; slot < kSlotsPerBucket; slot++) {
        if (!(bucket.occupiedMask & (1 << slot))) continue;

        if (k == bucket.keys[slot]) {
          bucket.values[slot] = val;
          return (b << 2) + (FastHasher64<Key>(getSeed(b))(k) >> 62);
        }
      }
    }

    return uint32_t(-1);
  }

  // Returns true if found.  Sets *out = value.
  inline bool lookUp(const Key &k, Value &out) const {
    if (!fallback.empty()) {
      auto it = fallback.find(k);
      if (it != fallback.end())
        return it->second;
    }
    uint32_t buckets[2];
    fast_map_to_buckets(h(k), buckets);

    for (uint32_t &b : buckets) {
      const Bucket &bucket = buckets_[b];
      if (FindInBucket(k, bucket, out)) return true;
    }

    return false;
  }

//  /// compose two maps in place
//  void Compose(unordered_map<Value, Value> &migrate) {
//    for (auto &bucket : buckets_) {
//      for (char slot = 0; slot < kSlotsPerBucket; ++slot) {
//        if (bucket.occupiedMask & (1ULL << slot)) {
//          Value &value = bucket.values[slot];
//          auto it = migrate.find(value);
//          if (it != migrate.end()) {
//            Value dst = it->second;
//            if (dst == Value(-1)) bucket.occupiedMask &= ~(1ULL << slot);
//            else bucket.values[slot] = it->second;
//          }
//
////          checkIntegrity();
//        }
//      }
//    }
//  }

  template<class NV>
  ControlPlaneMinimalPerfectCuckoo<Key, NV> Compose(unordered_map<Value, NV> &migrate) {
    ControlPlaneMinimalPerfectCuckoo<Key, NV> other;

    other.entryCount = entryCount;
    other.cpq_.reset();
    other.buckets_.resize(buckets_.size());
    other.h = h;

    for (uint32_t i = 0; i < buckets_.size(); ++i) {
      auto &bsrc = buckets_[i];
      auto &bdst = other.buckets_[i];
      bdst.occupiedMask = bsrc.occupiedMask;

      for (char slot = 0; slot < kSlotsPerBucket; ++slot) {
        if (!(bsrc.occupiedMask & (1 << slot))) continue;

        bdst.keys[slot] = bsrc.keys[slot];

        Value &value = bsrc.values[slot];
        auto it = migrate.find(value);
        if (it != migrate.end()) {
          bdst.values[slot] = it->second;
        } else {
          bdst.occupiedMask &= ~(1 << slot);
          other.entryCount--;
        }
      }
    }

    return other;
  }

  void SelfCompose(unordered_map<Value, Value> &migrate) {
    for (uint32_t i = 0; i < buckets_.size(); ++i) {
      auto &bucket = buckets_[i];

      for (char slot = 0; slot < kSlotsPerBucket; ++slot) {
        if (!(bucket.occupiedMask & (1 << slot))) continue;

        Value &value = bucket.values[slot];
        auto it = migrate.find(value);
        if (it != migrate.end()) {
          bucket.values[slot] = it->second;
        } else {
          bucket.occupiedMask &= ~(1 << slot);
          entryCount--;
        }
      }
    }
  }

  unordered_map<Key, Value, Hasher32<Key>> toMap() const {
    unordered_map<Key, Value, Hasher32<Key>> map;

    for (auto &bucket: buckets_) {  // all buckets
      for (char slot = 0; slot < kSlotsPerBucket; ++slot) {
        if (bucket.occupiedMask & (1 << slot)) {
          map.insert(make_pair(bucket.keys[slot], bucket.values[slot]));
        }
      }
    }

    return map;
  }

  // This function will not take memory on the heap into account because we don't know the memory layout of keys on heap
  uint64_t getMemoryCost() const {
    return buckets_.size() * sizeof(buckets_[0]);
  }

//  void checkIntegrity() {
//    for (auto &bucket: buckets_) {  // all buckets
//      for (char slot = 0; slot < 4; ++slot) {
//        if (bucket.occupiedMask & (1ULL << slot)) {
//          const Key &key = bucket.keys[slot];
//          const uint16_t host = bucket.values[slot];
//          assert(host < 60000);
//        }
//      }
//    }
//  }

  // For the associative cuckoo table, check all of the slots in
  // the bucket to see if the key is present.
  inline bool FindInBucket(const Key &k, const Bucket &bucket, Value &out) const {
    for (char s = 0; s < kSlotsPerBucket; s++) {
      if ((bucket.occupiedMask & (1U << s)) && (bucket.keys[s] == k)) {
        out = bucket.values[s];
        return true;
      }
    }
    return false;
  }

  inline char FindFreeSlot(uint32_t bucket) const {
    return FindFreeSlot(buckets_[bucket]);
  }

  //  returns either -1 or the index of an
  //  available slot (0 <= slot < kSlotsPerBucket)
  inline char FindFreeSlot(const Bucket &bucket) const {
    for (char i = 0; i < kSlotsPerBucket; i++) {
      if (!(bucket.occupiedMask & (1U << i))) {
        return i;
      }
    }
    return -1;
  }

  inline int64_t registerKey(const Key &k, uint32_t currentBucket, bool withDp) {
    uint32_t buckets[2];
    fast_map_to_buckets(h(k), buckets);

    return locator.insert(k, buckets[1] == currentBucket, withDp);
  }

  inline void unregisterKey(const Key &k) {
    locator.remove(k);
  }

  inline int64_t toggleKey(const Key &k) {
    uint8_t result;
    if (locator.lookUp(k, result)) return locator.updateMapping(k, !result);
    throw runtime_error("No such key");
  }

  // For the associative cuckoo table, check all of the slots in
  // the bucket to see if the key is present.
  inline bool RemoveInBucket(const Key &k, Bucket &bucket) {
    for (char i = 0; i < kSlotsPerBucket; i++) {
      if ((bucket.occupiedMask & (1U << i)) && bucket.keys[i] == k) {
        bucket.occupiedMask ^= 1U << i;

        unregisterKey(k);
//        checkIntegrity();
        return true;
      }
    }
    return false;
  }

  inline uint8_t getSeed(uint32_t bucketId) const {
    uint8_t result = buckets_[bucketId].seed;
    return result;
  }

  FastHasher64<Key> digestH = FastHasher64<Key>(rand());

  inline void
  putItem(const Key &k, const Value &v, uint32_t b, uint8_t slot, vector<MPC_PathEntry> *const path = 0) {
    Bucket &bucket = buckets_[b];

    bucket.keys[slot] = k;
    bucket.values[slot] = v + ((digestH(k) << VL) & DigestMask);
    bucket.occupiedMask |= 1U << slot;

    if (path) {
      uint8_t oldSeed = getSeed(b);
      uint8_t toSlot[4];
      uint8_t seed = updateSeed(b, toSlot, slot);
      int64_t result = registerKey(k, b, path != nullptr);

      path->push_back({b, uint8_t(FastHasher64<Key>(seed)(k) >> 62), seed,
                       toSlot[0], toSlot[1], toSlot[2], toSlot[3],
                       result ? locator.getHalfTree(k, result > 0, false) : vector<uint32_t>()});
    }
    // checkIntegrity();
  }

  inline void
  moveItem(uint32_t sBkt, uint8_t sSlot, uint32_t dBkt, uint8_t dSlot, vector<MPC_PathEntry> *const path) {
    Counter::count("Cuckoo copy item");
    Bucket &dst_bucket = buckets_[dBkt];
    Bucket &src_bucket = buckets_[sBkt];

    dst_bucket.keys[dSlot] = src_bucket.keys[sSlot];
    dst_bucket.values[dSlot] = src_bucket.values[sSlot];

    if (path) {
      Key &k = dst_bucket.keys[dSlot];

      uint8_t oldSeed = getSeed(dBkt);
      uint8_t toSlot[4];
      uint8_t seed = updateSeed(dBkt, toSlot, dSlot);
      int64_t result = toggleKey(k);
      assert(result != 0);

      path->push_back({dBkt, uint8_t(FastHasher64<Key>(seed)(k) >> 62), seed,
                       toSlot[0], toSlot[1], toSlot[2], toSlot[3],
                       locator.getHalfTree(k, result > 0, false)});
    }
//    checkIntegrity();
  }

  uint8_t updateSeed(uint32_t bktIdx, uint8_t *dpSlotMove = 0, char slotWithNewKey = -1) {
    Bucket &bucket = buckets_[bktIdx];
    FastHasher64<Key> h;
    bool occupied[4];

    for (uint8_t seed = 0; seed < 255; ++seed) {
      *(uint32_t *) occupied = 0U;
      h.setSeed(seed);

      bool success = true;

      for (char slot = 0; slot < kSlotsPerBucket; ++slot) {
        if (bucket.occupiedMask & (1 << slot)) {
          uint8_t i = uint8_t(h(bucket.keys[slot]) >> 62);
          if (occupied[i]) {
            success = false;
            break;
          } else { occupied[i] = true; }
        }
      }

      if (success) {
        bool withDp = dpSlotMove != nullptr;

        if (withDp) {
          FastHasher64<Key> oldH(getSeed(bktIdx));

          memset(dpSlotMove, -1, 4);
          for (char slot = 0; slot < kSlotsPerBucket; ++slot) {
            if ((bucket.occupiedMask & (1 << slot)) && slot != slotWithNewKey) {
              uint8_t oldSlot = uint8_t(oldH(bucket.keys[slot]) >> 62);
              uint8_t toSlot = uint8_t(h(bucket.keys[slot]) >> 62);

              dpSlotMove[oldSlot] = toSlot;
            }
          }

          *(uint32_t *) occupied = 0U;
          for (char slot = 0; slot < kSlotsPerBucket; ++slot) {
            if (dpSlotMove[slot] != uint8_t(-1)) {
              occupied[dpSlotMove[slot]] = true;
            }
          }

          char firstUnusedNewSlot = -1;
          for (char slot = 0; slot < kSlotsPerBucket; ++slot) {
            if (!occupied[slot]) {
              firstUnusedNewSlot = slot;
              break;
            }
          }

          for (char slot = 0; slot < kSlotsPerBucket; ++slot) {
            if (dpSlotMove[slot] == uint8_t(-1)) {
              assert(firstUnusedNewSlot >= 0);
              dpSlotMove[slot] = uint8_t(firstUnusedNewSlot);
            }
          }
        }

        Counter::countMax("MPC max seed", seed);
        bucket.seed = seed;

        return seed;
      }
    }

    throw runtime_error("Cannot generate a proper hash seed within 255 tries, which is rare");
  }

  void prepareToExport() {
    vector<Key> keys;
    vector<uint8_t> values;
    keys.reserve(entryCount);
    values.reserve(entryCount);

    for (uint32_t bktIdx = 0; bktIdx < buckets_.size(); ++bktIdx) {
      Bucket &bucket = buckets_[bktIdx];
      updateSeed(bktIdx);

      for (char s = 0; s < kSlotsPerBucket; ++s) {
        if (bucket.occupiedMask & (1 << s)) {
          uint32_t buckets[2];
          Key k = bucket.keys[s];
          fast_map_to_buckets(h(k), buckets);

          keys.push_back(k);
          values.push_back(uint8_t(buckets[1] == bktIdx));
        }
      }
    }

    locator.keys = keys;
    locator.values = values;
    locator.keyCnt = keys.size();
    locator.build();
  }

  Bucket getDpBucket(uint32_t index) const {
    const Bucket cpBucket = buckets_[index];

    Bucket result;
    result.occupiedMask = 0;
    result.seed = getSeed(index);
    FastHasher64<Key> h(result.seed);

    for (char cpSlot = 0; cpSlot < 4; ++cpSlot) {
      if (!(cpBucket.occupiedMask & (1ULL << cpSlot))) continue;

      char dpSlot = h(cpBucket.keys[cpSlot]) >> 62;
      result.occupiedMask |= 1 << dpSlot;
      result.values[dpSlot] = cpBucket.values[cpSlot];
    }

    return result;
  }

  // Insert uses the BFS optimization (search before moving) to reduce
  // the number of cache lines dirtied during search.
  /// @return cuckoo path if rememberPath is true. or {1} to indicate success and {} to indicate fail.
  bool CuckooInsert(const Key &k, const Value &v, vector<MPC_PathEntry> *const path) {
    CuckooPathQueue cpq_;
    CuckooPathEntry visited_[kVisitedListSize];

    int visited_end = -1;
    cpq_.reset();

    {
      uint32_t buckets[2];
      fast_map_to_buckets(h(k), buckets);

      for (uint32_t b : buckets) {
        cpq_.push_back({b, 1, -1, -1}); // Note depth starts at 1.
        Counter::count("Cuckoo reach bucket");
      }
    }

    while (!cpq_.empty()) {
      CuckooPathEntry entry = cpq_.pop_front();
      Counter::count("Cuckoo visit bucket");
      char free_slot = FindFreeSlot(entry.bucket);
      if (free_slot != -1) {
        Counter::count("Cuckoo total depth", entry.depth);
        Counter::countMax("Cuckoo max depth", entry.depth);

        // found a free slot in this path. just insert and follow this path
        buckets_[entry.bucket].occupiedMask |= 1U << free_slot;
        while (entry.depth > 1) {
          // "copy" instead of "swap" because one entry is always zero.
          // After, write target key/value over top of last copied entry.
          CuckooPathEntry parent = visited_[entry.parent];

          moveItem(parent.bucket, entry.parent_slot, entry.bucket, free_slot, path);

          free_slot = entry.parent_slot;
          entry = parent;
        }

        putItem(k, v, entry.bucket, free_slot, path);
        return true;
      } else if (entry.depth < kMaxBFSPathLen) {
        visited_[++visited_end] = entry;
        auto parent_index = visited_end;

        // Don't always start with the same slot, to even out the path depth.
        char start_slot = (entry.depth + entry.bucket) % kSlotsPerBucket;
        const Bucket &bucket = buckets_[entry.bucket];

        for (char i = 0; i < kSlotsPerBucket; i++) {
          char slot = (start_slot + i) % kSlotsPerBucket;

          uint32_t buckets[2];
          fast_map_to_buckets(h(bucket.keys[slot]), buckets);

          char j = 0;
          if (buckets[j] == entry.bucket) j++;
          if (buckets[j] == entry.bucket) continue;

          cpq_.push_back({buckets[j], entry.depth + 1, parent_index, slot});
          Counter::count("Cuckoo reach bucket");
        }
      }
    }

    return false;
  }
};

template<class Key, class Value, uint8_t VL = sizeof(Value) * 8, uint8_t DL = 0>
class DataPlaneMinimalPerfectCuckoo {
  static const uint8_t kSlotsPerBucket = 4;   // modification to this value leads to undefined behavior
  static const uint8_t bucketLength = LocatorSeedLength + kSlotsPerBucket * (VL + DL);

  static const uint64_t ValueMask = (1ULL << VL) - 1;
  static const uint64_t DigestMask = ((1ULL << DL) - 1) << VL;
  static const uint64_t VDMask = (1ULL << (VL + DL)) - 1;

public:
  FastHasher64<Key> h;
  uint32_t num_buckets_;

  std::vector<uint64_t> memory;
  FastHasher64<Key> digestH;
  DataPlaneOthello<Key, uint8_t, 1> locator;
  CuckooHashTable<uint32_t, uint8_t> overflow;

  struct Bucket {  // only as parameters and return values for easy access. the storage is compact.
    uint8_t seed;
    Value values[kSlotsPerBucket];

    bool operator==(const Bucket &other) const {
      if (seed != other.seed) return false;

      for (char s = 0; s < kSlotsPerBucket; s++) {
        if ((values[s] & ValueMask) != (other.values[s] & ValueMask)) return false;
      }

      return true;
    }

    bool operator!=(const Bucket &other) const {
      return !(*this == other);
    }
  };

  explicit DataPlaneMinimalPerfectCuckoo(const ControlPlaneMinimalPerfectCuckoo<Key, Value, VL, DL> &cp)
    : num_buckets_(cp.buckets_.size()), h(cp.h), locator(cp.locator), overflow(cp.entryCount * 0.012),
      digestH(cp.digestH) {

    resetMemory();

    for (uint32_t bktIdx = 0; bktIdx < num_buckets_; ++bktIdx) {
      const typename ControlPlaneMinimalPerfectCuckoo<Key, Value, VL>::Bucket &cpBucket = cp.buckets_[bktIdx];
      Bucket dpBucket;
      dpBucket.seed = cpBucket.seed > MaxArrangementSeed ? MaxArrangementSeed : cpBucket.seed;
      memset(dpBucket.values, 0, kSlotsPerBucket * sizeof(Value));

      if (cpBucket.seed >= MaxArrangementSeed) {
        overflow.insert(bktIdx, cpBucket.seed);
      }

      const FastHasher64<Key> locateHash(cpBucket.seed);

      for (char slot = 0; slot < kSlotsPerBucket; ++slot) {
        if (cpBucket.occupiedMask & (1U << slot)) {
          const Key &k = cpBucket.keys[slot];
          dpBucket.values[locateHash(k) >> 62] = cpBucket.values[slot];
        }
      }

      writeBucket(dpBucket, bktIdx);
    }
  }

  template<class V2>
  DataPlaneMinimalPerfectCuckoo(const ControlPlaneMinimalPerfectCuckoo<Key, V2, VL, DL> &cp, unordered_map<V2, Value> m)
    : num_buckets_(cp.buckets_.size()), h(cp.h), locator(cp.locator),
      overflow(cp.entryCount * 0.012), digestH(cp.digestH) {

    resetMemory();

    for (uint32_t bktIdx = 0; bktIdx < num_buckets_; ++bktIdx) {
      const typename ControlPlaneMinimalPerfectCuckoo<Key, V2, VL>::Bucket &cpBucket = cp.buckets_[bktIdx];
      Bucket dpBucket;
      dpBucket.seed = cpBucket.seed >= MaxArrangementSeed ? MaxArrangementSeed : cpBucket.seed;
      memset(dpBucket.values, 0, kSlotsPerBucket * sizeof(Value));

      if (cpBucket.seed >= MaxArrangementSeed) {
        overflow.insert(bktIdx, cpBucket.seed);
      }

      const FastHasher64<Key> locateHash(cpBucket.seed);

      for (char slot = 0; slot < kSlotsPerBucket; ++slot) {
        if (cpBucket.occupiedMask & (1U << slot)) {
          const Key &k = cpBucket.keys[slot];
          dpBucket.values[locateHash(k) >> 62] = m[cpBucket.values[slot]];
        }
      }

      writeBucket(dpBucket, bktIdx);
    }
  }

  inline void resetMemory() {
    memory.resize(((uint64_t) num_buckets_ * bucketLength + 63) / 64);
  }

  template<char length>
  inline void writeMem(uint64_t start, char offset, uint64_t v) {
    assert(v < (1ULL << length));

    // [offset, offset + length) should be 0 and others are 1
    uint64_t mask = ~uint64_t(((1ULL << length) - 1) << offset);
    char overflow = char(offset + length - 64);

    memory[start] &= mask;
    memory[start] |= (uint64_t) v << offset;

    if (overflow > 0) {
      mask = uint64_t(-1) << overflow;     // lower "overflow" bits should be 0, and others are 1
      memory[start + 1] &= mask;
      memory[start + 1] |= v >> (length - overflow);
    }

    assert(v == readMem<length>(start, offset));
  }

  template<char length>
  inline uint64_t readMem(uint64_t start, char offset) const {
    char left = char(offset + length - 64);
    left = char(left < 0 ? 0 : left);

    uint64_t mask = ~(uint64_t(-1)
      << (length - left));   // lower length-left bits should be 1, and others are 0
    uint64_t result = (memory[start] >> offset) & mask;

    if (left > 0) {
      mask = ~(uint64_t(-1) << left);     // lower left bits should be 1, and others are 0
      result |= (memory[start + 1] & mask) << (length - left);
    }

    return result;
  }

  vector<uint8_t> lock = vector<uint8_t>(8192, 0);

  // Only call it during update!
  inline void writeBucket(Bucket &bucket, uint32_t index) {
    uint64_t i1 = (uint64_t) index * bucketLength;
    uint64_t start = i1 / 64;
    char offset = char(i1 % 64);

    assert(bucket.seed <= MaxArrangementSeed);
    writeMem<LocatorSeedLength>(start, offset, bucket.seed);

    for (char i = 0; i < kSlotsPerBucket; ++i) {
      uint64_t offsetFromBeginning = i1 + LocatorSeedLength + i * VL;
      start = offsetFromBeginning / 64;
      offset = char(offsetFromBeginning % 64);

      writeMem<VL>(start, offset, bucket.values[i]);
    }

#ifndef NDEBUG
    Bucket retrieved = readBucket(index);
    retrieved.seed = min(MaxArrangementSeed, retrieved.seed);
    assert(retrieved == bucket);
#endif
  }

  inline Bucket readBucket(uint32_t index) const {
    Bucket bucket;

    uint64_t i1 = (uint64_t) index * bucketLength;
    uint64_t start = i1 / 64;
    char offset = char(i1 % 64);

    bucket.seed = readMem<LocatorSeedLength>(start, offset);
    if (bucket.seed == MaxArrangementSeed) {
      overflow.lookUp(index, bucket.seed);
    }

    for (char i = 0; i < kSlotsPerBucket; ++i) {
      uint64_t offsetFromBeginning = i1 + LocatorSeedLength + i * VL;
      start = offsetFromBeginning / 64;
      offset = char(offsetFromBeginning % 64);

      bucket.values[i] = readMem<VL>(start, offset);
    }

    return bucket;
  }

  inline void writeSlot(uint32_t bid, char sid, Value val) {
    uint64_t offsetFromBeginning = uint64_t(bid) * bucketLength + LocatorSeedLength + sid * VL;
    uint64_t start = offsetFromBeginning / 64;
    char offset = char(offsetFromBeginning % 64);

    writeMem<VL>(start, offset, val);
  }

  inline Value readSlot(uint32_t bid, char sid) {
    uint64_t offsetFromBeginning = uint64_t(bid) * bucketLength + LocatorSeedLength + sid * VL;
    uint64_t start = offsetFromBeginning / 64;
    char offset = char(offsetFromBeginning % 64);

    return readMem<VL>(start, offset);
  }

  unordered_map<Key, Value> fallback;

  // Returns true if found.  Sets *out = value.
  inline bool lookUp(const Key &k, Value &out) const {
    if (!fallback.empty()) {
      auto it = fallback.find(k);
      if (it != fallback.end())
        return it->second;
    }
    uint32_t buckets[2];
    fast_map_to_buckets(h(k), buckets);

    while (true) {
      uint8_t va1 = lock[buckets[0] & 8191], vb1 = lock[buckets[1] & 8191];
      COMPILER_BARRIER();
      if (va1 % 2 == 1 || vb1 % 2 == 1) continue;

      Bucket bucket = readBucket(buckets[locator.lookUp(k)]);

      COMPILER_BARRIER();
      uint8_t va2 = lock[buckets[0] & 8191], vb2 = lock[buckets[1] & 8191];

      if (va1 != va2 || vb1 != vb2) continue;

      uint64_t i = FastHasher64<Key>(bucket.seed)(k) >> 62;
      Value result = bucket.values[i];

      if ((result & DigestMask) == ((digestH(k) << VL) & DigestMask)) {
        out = result & ValueMask;
        return true;
      } else { return false; }
    }
  }

  inline void applyInsert(const vector<MPC_PathEntry> &path, Value value) {
    for (int i = 0; i < path.size(); ++i) {
      MPC_PathEntry entry = path[i];
      Bucket bucket = readBucket(entry.bid);
      bucket.seed = min(entry.newSeed, MaxArrangementSeed);

      uint8_t toSlots[] = {entry.s0, entry.s1, entry.s2, entry.s3};

      Value buffer[4];       // solve the permutation is slow. just copy the 4 elements
      for (char s = 0; s < 4; ++s) {
        buffer[s] = bucket.values[s];
      }

      for (char s = 0; s < 4; ++s) {
        bucket.values[toSlots[s]] = buffer[s];
      }

      if (i + 1 == path.size()) {  // put the new value
        bucket.values[entry.sid] = value;
      } else {  // move key from another bucket and slot to this bucket and slot
        MPC_PathEntry from = path[i + 1];
        uint8_t tmp[4] = {from.s0, from.s1, from.s2, from.s3};
        uint8_t sid;
        for (uint8_t ii = 0; ii < 4; ++ii) {
          if (tmp[ii] == from.sid) {
            sid = ii;
            break;
          }
        }
        bucket.values[entry.sid] = readSlot(from.bid, sid);
      }

      lock[entry.bid & 8191]++;
      COMPILER_BARRIER();

      if (entry.locatorCC.size()) {
        locator.fixHalfTreeByConnectedComponent(entry.locatorCC, 1);
      }

      if (bucket.seed == MaxArrangementSeed) {
        overflow.insert(entry.bid, entry.newSeed, true);
      }
      writeBucket(bucket, entry.bid);

      COMPILER_BARRIER();
      lock[entry.bid & 8191]++;
    }
  }

  inline void applyUpdate(uint32_t bs, Value val) {
    uint32_t bid = bs >> 2;
    uint8_t sid = bs & 3;

    lock[bid & 8191]++;
    COMPILER_BARRIER();

    writeSlot(bid, sid, val & ValueMask);

    COMPILER_BARRIER();
    lock[bid & 8191]++;
  }

  inline uint64_t getMemoryCost() const {
    return memory.size() * 8;
  }

  // Utility function to compute (x * y) >> 64, or "multiply high".
  // On x86-64, this is a single instruction, but not all platforms
  // support the __uint128_t type, so we provide a generic
  // implementation as well.
  inline uint32_t multiply_high_u32(uint32_t x, uint32_t y) const {
    return (uint32_t) (((uint64_t) x * (uint64_t) y) >> 32);
  }

  inline void fast_map_to_buckets(uint64_t x, uint32_t *twoBuckets) const {
    // Map x (uniform in 2^64) to the range [0, num_buckets_ -1]
    // using Lemire's alternative to modulo reduction:
    // http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    // Instead of x % N, use (x * N) >> 64.
    twoBuckets[0] = multiply_high_u32(x, num_buckets_);
    twoBuckets[1] = multiply_high_u32(x >> 32, num_buckets_);
  }
};

