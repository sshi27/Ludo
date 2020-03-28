//
// Created by ssqstone on 2019/6/17.
//

#pragma once

//
// Created by ssqstone on 2018/7/13.
//
#pragma once

#include "../control_plane.h"
#include "../common.h"
#include "cuckoo_map.h"

template<class K, class V, class Match = uint8_t, char digestLength = -1>
class DataPlaneCuckooFiltable {
public:
  static const char DL = digestLength > 0 ? digestLength : sizeof(Match) * 8;
  DataPlaneCuckooMap<K, V, Match, DL, 2, 4> level1;
  ControlPlaneCuckooMap<K, V, Match, false, DL, 2, 4> level2;
  
  template<bool B1, bool B2>
  DataPlaneCuckooFiltable(
    ControlPlaneCuckooMap<K, V, Match, B1, DL, 2, 4> lv1,
    ControlPlaneCuckooMap<K, V, Match, B2, DL, 2, 4> lv2)
    : level1(lv1), level2(lv2) {
  }
  
  template<bool B>
  DataPlaneCuckooFiltable(
    DataPlaneCuckooMap<K, V, Match, DL, 2, 4> lv1,
    ControlPlaneCuckooMap<K, V, Match, B, DL, 2, 4> lv2)
    : level1(lv1), level2(lv2) {
  }
  
  // Returns true if found.  Sets *out = value.
  inline bool lookUp(const K &k, V &out) const {
    assert(level1.lookUp(k, out) ^ level2.lookUp(k, out));
    if (level1.lookUp(k, out)) return true;
    
    Counter::count("DataPlaneCuckooFiltable fallback to lv2");
    return level2.lookUp(k, out);
  }
  
  inline void insertLv2(const K &key, V port, const vector<CuckooMove> &cuckooPath) {
    if (!cuckooPath.empty()) {
      for (const CuckooMove &move: cuckooPath) {
        level2.CopyItem(move.bsrc, move.ssrc, move.bdst, move.sdst);
      }
      const CuckooMove &first = cuckooPath.front();
      level2.buckets_[first.bdst].occupiedMask |= 1U << first.sdst;
    }
    
    level2.InsertInternal(key, port, cuckooPath.back().bsrc, cuckooPath.back().ssrc);
  }
  
  inline void insertLv1(Match digest, V port, const vector<CuckooMove> &cuckooPath) {
    for (const CuckooMove &move: cuckooPath) {
      level1.CopyItem(move.bsrc, move.ssrc, move.bdst, move.sdst);
    }
    
    const CuckooMove &first = cuckooPath.front(), &last = cuckooPath.back();
    level1.buckets_[first.bdst].occupiedMask |= 1U << first.sdst;
    
    level1.InsertAt(last.bsrc, last.ssrc, digest, port);
  }
  
  inline void erase(bool lv1, uint32_t bid, uint8_t sid) {
    if (lv1) {
      level1.buckets_[bid].occupiedMask &= ~(1U << sid);
    } else {
      level2.buckets_[bid].occupiedMask &= ~(1U << sid);
    }
  }
  
  inline void modify(bool lv1, uint32_t bid, uint8_t sid, V newValue) {
    if (lv1) {
      level1.buckets_[bid].values[sid] = newValue;
    } else {
      level2.buckets_[bid].values[sid] = newValue;
    }
  }
  
  uint64_t getMemoryCost() const {
    return level1.getMemoryCost() + level2.getMemoryCost();
  }
};

template<class K, class V, class Match, char DL>
class FullKeyDataPlaneCuckooFiltable {
public:
  ControlPlaneCuckooMap<K, V, Match, false, DL, 2, 4> level1;
  ControlPlaneCuckooMap<K, V, Match, false, DL, 2, 4> level2;
  
  template<bool B1, bool B2>
  explicit FullKeyDataPlaneCuckooFiltable(
    const ControlPlaneCuckooMap<K, V, Match, B1, DL, 2, 4> lv1,
    const ControlPlaneCuckooMap<K, V, Match, B2, DL, 2, 4> lv2)
    : level1(lv1), level2(lv2) {
  }
  
  
  // Returns true if found.  Sets *out = value.
  inline bool lookUp(const K &k, V &out) const {
    return level1.lookUp(k, out) || level2.lookUp(k, out);
  }
  
  /// compose two maps in place
  void Compose(unordered_map<V, V> &migrate) {
    level1.Compose(migrate);
    level2.Compose(migrate);
  }
  
  uint64_t getMemoryCost() const {
    return level1.getMemoryCost() + level2.getMemoryCost();
  }
};

template<class K, class V, class Match = uint8_t, char digestLength = -1>
class ControlPlaneCuckooFiltable {

public:
  static const char DL = digestLength > 0 ? digestLength : sizeof(Match) * 8;
  
  ControlPlaneCuckooMap<K, V, Match, true, DL, 2, 4> *level1;
  ControlPlaneCuckooMap<K, V, Match, false, DL, 2, 4> *level2;
  uint32_t capacity;
  
  explicit ControlPlaneCuckooFiltable(uint32_t capacity)
    : capacity(capacity), level1(new ControlPlaneCuckooMap<K, V, Match, true, DL, 2, 4>(capacity)),
      level2(new ControlPlaneCuckooMap<K, V, Match, false, DL, 2, 4>(capacity / 10)) {
  }
  
  virtual ~ControlPlaneCuckooFiltable() {
    delete level1;
    delete level2;
  }
  
  inline void insert_(const K &k, V value, bool inRebuild = false) {
    const K *result = level1->insert(k, value);
    
    if (result == nullptr) { // level1 full, rebuild
      if (inRebuild) {
        throw runtime_error("rebuild in rebuild");
      }
      rebuild(k, value);
    } else if (result != &k) { // collision
      Counter::count("ControlPlaneCuckooFiltable digest collision");
      
      vector<const K *> collisions = level1->FindAllCollisions(k);
      for (int i = 0; i < 2; ++i) {
        if (collisions[i] == nullptr) continue;
        
        V value;
        if (!level1->lookUp(*collisions[i], value)) continue;
        
        if (level2->insert(*collisions[i], value) != collisions[i]) {  // l2 full, rebuild l2
          rebuildL2(*collisions[i], value);
        }
        level1->remove(*collisions[i], false);
      }
      
      level1->PreventCollision(k);
      
      if (level2->insert(k, value) != &k) {  // l2 full, rebuild l2
        rebuildL2(k, value);
      }
    } else {
      Counter::count("ControlPlaneCuckooFiltable lv1 add");
    }
  }
  
  inline void insert(const K &k, V value) {
    insert_(k, value, false);
  }
  
  inline void remove(const K &k) {
    if (!level1->remove(k)) {
      level2->remove(k);
      level1->deleteCollision(k);
    }
  }
  
  inline bool lookUp(const K &k, V &out) const {
    return level1->lookUp(k, out) || level2->lookUp(k, out);
  }

private:
  void rebuild(const K &k, V value) {
    Counter::count("ControlPlaneCuckooFiltable level1 rebuild");
    Clocker rebuild("Cuckoo level1 rebuild");
    
    unordered_map<K, V, Hasher32<K>> map = level1->toMap();
    unordered_map<K, V, Hasher32<K>> map2 = level2->toMap();
    map.insert(map2.begin(), map2.end());
    
    map.insert(make_pair(k, value));
    
    delete level1;
    delete level2;
    level1 = nullptr;
    level2 = nullptr;
    
    while (!level1) {
      Counter::count("ControlPlaneCuckooFiltable level1 re-rebuild");
      
      level1 = new ControlPlaneCuckooMap<K, V, Match, true, DL, 2, 4>(capacity);
      level2 = new ControlPlaneCuckooMap<K, V, Match, false, DL, 2, 4>(capacity / 10);
      
      for (auto it = map.begin(); it != map.end(); ++it) {
        try {
          insert_(it->first, it->second, true);
        } catch (exception &e) {
          delete level1;
          delete level2;
          level1 = nullptr;
          level2 = nullptr;
          break;
        }
      }
    }
  }
  
  void rebuildL2(const K &k, V value) {
    Counter::count("ControlPlaneCuckooFiltable level2 full, only rebuild lv2");
    unordered_map<K, V, Hasher32<K>> map = level2->toMap();
    map.insert(make_pair(k, value));
    
    delete level2;
    level2 = 0;
    
    while (!level2) {
      level2 = new ControlPlaneCuckooMap<K, V, Match, false, DL, 2, 4>(capacity / 10);
      
      for (auto it = map.begin(); it != map.end(); ++it) {
        const K *result = level2->insert(it->first, it->second);
        
        if (result != &it->first) {
          delete level2;
          level2 = 0;
          break;
        }
      }
    }
  }
};

