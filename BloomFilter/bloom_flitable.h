//
// Created by ssqstone on 2018/7/17.
//

#pragma once

#include "../common.h"
#include "../control_plane.h"
#include "../CuckooPresized/cuckoo_map.h"
#include "bloom_filter.h"

template<class K, class V, class Match = uint8_t>
class ControlPlaneBloomFiltable {
public:
  ControlPlaneCuckooMap<K, V, Match, false, 2, 4> *m;
  uint32_t capacity;
  
  explicit ControlPlaneBloomFiltable(uint32_t capacity)
    : capacity(capacity), m(new ControlPlaneCuckooMap<K, V, Match, false, 2, 4>(capacity)) {
  }
  
  inline void insert(const K &k, V host) {
    const K *result = m->insert(k, host);
    
    if (result != &k) {
      rebuild(k, host);
    } else {
      Counter::count("Bloom-CP cuckoo simple add");
    }
  }
  
  void rebuild(const K &k, V host) {
    Counter::count("Bloom-CP cuckoo full, rebuild");
    unordered_map<K, V, Hasher32<K>> map = m->toMap();
    map.insert(make_pair(k, host));
    
    delete m;
    m = 0;
    
    while (!m) {
      m = new ControlPlaneCuckooMap<K, V, Match, false, 2, 4>(capacity);
      
      for (auto it = map.begin(); it != map.end(); ++it) {
        const K *result = m->insert(it->first, it->second);
        
        if (result != &it->first) {
          delete m;
          m = 0;
          break;
        }
      }
    }
  }
  
  void remove(const K &k) {
    m->remove(k);
  }
  
  inline bool lookUp(const K &k, V &out) const {
    return m->lookUp(k, out);
  }
};

template<class K, class V>
class DataPlaneBloomFiltable {
public:
  DataPlaneBloomFiltable(V numOfValues, const ControlPlaneBloomFiltable<K, V> &cp): numOfValues(numOfValues) {
    unordered_map<V, unordered_set<K, Hasher32<K>>> vToKeys;
    for (V v = 0; v < numOfValues; ++v) {
      vToKeys.insert(make_pair(v, unordered_set<K, Hasher32<K>>()));
    }
    
    for (auto &bucket: cp.m->buckets_) {
      for (int slot = 0; slot < 4; ++slot) {
        if (bucket.occupiedMask & (1ULL << slot)) {
          const K &key = bucket.keys[slot];
          const uint16_t host = bucket.values[slot];
          const V port = host;
          
          vToKeys[port].insert(key);
        }
      }
    }
    
    for (V p = 0; p < numOfValues; ++p) {
      filters.push_back(BloomFilter<K>(uint32_t(vToKeys[p].size())));
      auto &filter = filters.back();
      
      countingFilters.push_back(BloomFilter<K, 4>(uint32_t(vToKeys[p].size())));
      auto &countingFilter = countingFilters.back();
      
      for (const K &k:vToKeys[p]) {
        countingFilter.insert(k);
        filter.insert(k);
      }
    }
  }
  
  vector<BloomFilter<K>> filters;
  vector<BloomFilter<K, 4>> countingFilters;
  uint16_t numOfValues;
  
  inline explicit DataPlaneBloomFiltable(V numOfValues) : numOfValues(numOfValues) {
  }
  
  inline bool lookUp(const K &k, V &out) const {
    static V start = V(-1);
    start = V((start + 1) % numOfValues);
    
    for (V i = 0; i < numOfValues; ++i) {
      V port = V((i + start) % numOfValues);
      const BloomFilter<K> &filter = filters[port];
      
      if (filter.getCapacity() && filter.isMember(k)) {
        out = port;
        return true;
      }
    }
    
    return false;
  }
  
  inline void insert(const K &k, V portIdx) {
    countingFilters[portIdx].insert(k);
    filters[portIdx].insert(k);
  }
  
  inline void erase(const K &k, V portIdx) {
    uint64_t toErase = countingFilters[portIdx].erase(k);
    filters[portIdx].mask(k, toErase);
  }
  
  inline void modify(const K &k, V before, V after) {
    erase(k, before);
    insert(k, after);
  }
  
  inline vector<V> matchAll(const K &k) const {
    vector<V> result;
    
    for (V port = 0; port < numOfValues; ++port) {
      const BloomFilter<K> &filter = filters[port];
      
      if (filter.getCapacity() && filter.isMember(k)) result.push_back(port);
    }
    
    return result;
  };
};
