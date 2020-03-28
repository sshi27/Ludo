//
// Created by ssqstone on 2018/7/13.
//
#pragma once

#include "../control_plane.h"
#include "../common.h"
#include "cuckoo_map.h"

template<class K, class Match = uint8_t, bool l2 = false, char digestLength = -1>
class TwoLevelCuckooRouter {
  typedef typename std::conditional<l2, uint8_t, uint16_t>::type P;
public:
  static const char DL = digestLength > 0 ? digestLength : sizeof(Match) * 8;
  DataPlaneCuckooMap<K, P, Match, DL, 2, 4> level1;
  ControlPlaneCuckooMap<K, P, Match, false, DL, 2, 4> level2;
  
  template<bool B1, bool B2>
  TwoLevelCuckooRouter(
    ControlPlaneCuckooMap<K, P, Match, B1, DL, 2, 4> lv1,
    ControlPlaneCuckooMap<K, P, Match, B2, DL, 2, 4> lv2)
    : level1(lv1), level2(lv2) {
  }
  
  template<bool B>
  TwoLevelCuckooRouter(
    DataPlaneCuckooMap<K, P, Match, DL, 2, 4> lv1,
    ControlPlaneCuckooMap<K, P, Match, B, DL, 2, 4> lv2)
    : level1(lv1), level2(lv2) {
  }
  
  // Returns true if found.  Sets *out = value.
  inline bool lookUp(const K &k, P &out) const {
    assert(level1.lookUp(k, out) ^ level2.lookUp(k, out));
    if (level1.lookUp(k, out)) return true;
    
    Counter::count("Cuckoo fallback to lv2");
    return level2.lookUp(k, out);
  }
  
  inline void insertLv2(const K &key, P port, const vector<CuckooMove> &cuckooPath) {
    if (!cuckooPath.empty()) {
      for (const CuckooMove &move: cuckooPath) {
        level2.CopyItem(move.bsrc, move.ssrc, move.bdst, move.sdst);
      }
      const CuckooMove &first = cuckooPath.front();
      level2.buckets_[first.bdst].occupiedMask |= 1U << first.sdst;
    }
    
    level2.InsertInternal(key, port, cuckooPath.back().bsrc, cuckooPath.back().ssrc);
  }
  
  inline void insertLv1(Match digest, P port, const vector<CuckooMove> &cuckooPath) {
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
  
  inline void modify(bool lv1, uint32_t bid, uint8_t sid, P newValue) {
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

template<class K, class Match, char DL>
class TwoLevelCuckooGateway {
public:
  ControlPlaneCuckooMap<K, uint8_t, Match, false, DL, 2, 4> level1;
  ControlPlaneCuckooMap<K, uint8_t, Match, false, DL, 2, 4> level2;
  
  template<bool B1, bool B2>
  explicit TwoLevelCuckooGateway(
    const ControlPlaneCuckooMap<K, uint8_t, Match, B1, DL, 2, 4> lv1,
    const ControlPlaneCuckooMap<K, uint8_t, Match, B2, DL, 2, 4> lv2)
    : level1(lv1), level2(lv2) {
  }
  
  
  // Returns true if found.  Sets *out = value.
  inline bool lookUp(const K &k, uint8_t &out) const {
    return level1.lookUp(k, out) || level2.lookUp(k, out);
  }
  
  /// compose two maps in place
  void Compose(unordered_map<uint8_t, uint8_t> &migrate) {
    level1.Compose(migrate);
    level2.Compose(migrate);
  }
  
  uint64_t getMemoryCost() const {
    return level1.getMemoryCost() + level2.getMemoryCost();
  }
};

template<class K, bool l2, class Match = uint8_t, char digestLength = -1, bool gatewayTwoLevel = true>
class TwoLevelCuckooControlPlane : public ControlPlane<K, l2> {
  typedef typename std::conditional<l2, uint8_t, uint16_t>::type P;

public:
  static const char DL = digestLength > 0 ? digestLength : sizeof(Match) * 8;
  vector<TwoLevelCuckooGateway<K, Match, DL>> gateways;
  vector<TwoLevelCuckooRouter<K, Match, l2, DL>> routers;
  
  using ControlPlane<K, l2>::insert;
  using ControlPlane<K, l2>::remove;
  using ControlPlane<K, l2>::graph;
  using ControlPlane<K, l2>::capacity;
  
  ControlPlaneCuckooMap<K, uint16_t, Match, true, DL, 2, 4> *level1;
  ControlPlaneCuckooMap<K, uint16_t, Match, false, DL, 2, 4> *level2;
  
  explicit TwoLevelCuckooControlPlane(uint32_t capacity)
    : ControlPlane<K, l2>(capacity),
      level1(new ControlPlaneCuckooMap<K, uint16_t, Match, true, DL, 2, 4>(capacity)),
      level2(new ControlPlaneCuckooMap<K, uint16_t, Match, false, DL, 2, 4>(capacity / 10)) {
  }
  
  void scenario(int topo) override {
    ControlPlane<K, l2>::scenario(topo);
    
    gateways.reserve(ControlPlane<K, l2>::getGatewayIds().size());
    routers.reserve(ControlPlane<K, l2>::getRouterIds().size());
  };
  
  virtual ~TwoLevelCuckooControlPlane() {
    delete level1;
    delete level2;
  }
  
  inline const char *getName() const override {
    return "Cuckoo-CP";
  }
  
  inline void insert_(const K &k, uint16_t host, bool inRebuild = false) {
    const K *result = level1->insert(k, host);
    
    if (result == nullptr) { // level1 full, rebuild
      if (inRebuild) {
        throw runtime_error("rebuild in rebuild");
      }
      rebuild(k, host);
    } else if (result != &k) { // collision
      Counter::count(getName(), "digest collision");
      
      vector<const K *> collisions = level1->FindAllCollisions(k);
      for (int i = 0; i < 2; ++i) {
        if (collisions[i] == nullptr) continue;
        
        uint16_t hostId;
        if (!level1->lookUp(*collisions[i], hostId)) continue;
        
        if (level2->insert(*collisions[i], hostId) != collisions[i]) {  // l2 full, rebuild l2
          rebuildL2(*collisions[i], hostId);
        }
        level1->remove(*collisions[i], false);
      }
      
      level1->PreventCollision(k);
      
      if (level2->insert(k, host) != &k) {  // l2 full, rebuild l2
        rebuildL2(k, host);
      }
    } else {
      Counter::count(getName(), "lv1 add");
    }
  }
  
  inline void insert(const K &k, uint16_t host) override {
    insert_(k, host, false);
  }
  
  inline void remove(const K &k) override {
    if (!level1->remove(k)) {
      level2->remove(k);
      level1->deleteCollision(k);
    }
  }
  
  inline bool lookUp(const K &k, uint16_t &out) const override {
    return level1->lookUp(k, out) || level2->lookUp(k, out);
  }
  
  inline uint8_t lookUpGateway(uint16_t id, const K &key) const override {
    uint8_t port = uint8_t(-1);
    return gateways[id].lookUp(key, port), port;
  }
  
  inline P lookUpRouter(uint16_t id, const K &key) const override {
    P port = static_cast<P>(-1);
    return routers[id].lookUp(key, port), port;
  }
  
  using ControlPlane<K, l2>::getGatewayIds;
  using ControlPlane<K, l2>::getRouterIds;
  using ControlPlane<K, l2>::getHostIds;
  
  template<bool isL2>
  typename std::enable_if<isL2, void>::type constructGateway() {
    auto hostIds = ControlPlane<K, isL2>::getHostIds();
    auto gatewayIds = ControlPlane<K, isL2>::getGatewayIds();
    ostringstream oss;
    oss << "Gateway construction (" << gatewayIds.size() << " gateways)";
    Clocker clocker(oss.str());
    
    ControlPlaneCuckooMap<K, uint8_t, Match, false, DL> lv1Temp;
    ControlPlaneCuckooMap<K, uint8_t, Match, false, DL> lv2Temp;
    
    lv1Temp.entryCount = level1->entryCount;
    lv1Temp.cpq_.reset();
    lv1Temp.num_buckets_ = level1->num_buckets_;
    lv1Temp.buckets_.clear();
    lv1Temp.buckets_.resize(level1->buckets_.size());
    
    for (int i = 0; i < 2 + 1; ++i) {
      lv1Temp.h[i] = level1->h[i];
    }
    
    for (int i = 0; i < level1->buckets_.size(); ++i) {
      auto &bsrc = level1->buckets_[i];
      auto &bdst = lv1Temp.buckets_[i];
      bdst.occupiedMask = bsrc.occupiedMask;
      
      for (int slot = 0; slot < 4; ++slot) {
        if (!(bsrc.occupiedMask & (1 << slot))) continue;
        
        bdst.keys[slot] = bsrc.keys[slot];
      }
    }
    
    lv2Temp.entryCount = level2->entryCount;
    lv2Temp.cpq_.reset();
    lv2Temp.num_buckets_ = level2->num_buckets_;
    lv2Temp.buckets_.clear();
    lv2Temp.buckets_.resize(level2->buckets_.size());
    
    for (int i = 0; i < 2 + 1; ++i) {
      lv2Temp.h[i] = level2->h[i];
    }
    
    for (int i = 0; i < level2->buckets_.size(); ++i) {
      auto &bsrc = level2->buckets_[i];
      auto &bdst = lv2Temp.buckets_[i];
      bdst.occupiedMask = bsrc.occupiedMask;
      
      for (int slot = 0; slot < 4; ++slot) {
        if (!(bsrc.occupiedMask & (1 << slot))) continue;
        
        bdst.keys[slot] = bsrc.keys[slot];
      }
    }
    
    for (int gatewayId: gatewayIds) {
      unordered_map<uint16_t, P> hostToPort;
      
      for (int i = 0; i < hostIds.size(); ++i) {
        int host = hostIds[i];
        vector<Graph<>::ShortestPathCell> path = graph->shortestPathTo[host];
        assert(!path.empty());  // asserting the host is a real host
        uint16_t nextHop = path[gatewayId].nextHop;
        
        P portNum = (P) (std::find(graph->adjacencyList[gatewayId].begin(),
                                   graph->adjacencyList[gatewayId].end(),
                                   Graph<>::AdjacencyMatrixCell({nextHop, 0})) -
                         graph->adjacencyList[gatewayId].begin());
        hostToPort.insert(make_pair(host, portNum));
      }
      
      for (int i = 0; i < level1->buckets_.size(); ++i) {
        auto &bsrc = level1->buckets_[i];
        auto &bdst = lv1Temp.buckets_[i];
        
        for (int slot = 0; slot < 4; ++slot) {
          if (!(bsrc.occupiedMask & (1 << slot))) continue;
          
          bdst.values[slot] = hostToPort[bsrc.values[slot]];
        }
      }
      
      for (int i = 0; i < level2->buckets_.size(); ++i) {
        auto &bsrc = level2->buckets_[i];
        auto &bdst = lv2Temp.buckets_[i];
        
        for (int slot = 0; slot < 4; ++slot) {
          if (!(bsrc.occupiedMask & (1 << slot))) continue;
          
          bdst.values[slot] = hostToPort[bsrc.values[slot]];
        }
      }
      
      TwoLevelCuckooGateway<K, Match, DL> gateway(lv1Temp, lv2Temp);
      
      #ifdef FULL_DEBUG
      unordered_map<K, uint16_t, Hasher32<K>> map;
      for (auto &bucket: level1->buckets_) {  // all buckets
        for (int slot = 0; slot < 4; ++slot) {
          if (bucket.occupiedMask & (1ULL << slot)) {
            const K key = bucket.keys[slot];
            const uint16_t host = bucket.values[slot];
            map.insert(make_pair(key, host));
          }
        }
      }
      
      for (auto &bucket: level2->buckets_) {  // all buckets
        for (int slot = 0; slot < 4; ++slot) {
          if (bucket.occupiedMask & (1ULL << slot)) {
            const K key = bucket.keys[slot];
            const uint16_t host = bucket.values[slot];
            map.insert(make_pair(key, host));
          }
        }
      }
      
      for (auto &pair:map) {
        const K &key = pair.first;
        uint16_t host = pair.second;
        
        auto val = map[key];
        
        uint16_t recordedHost = uint16_t(-1);
        assert(lookUp(key, recordedHost) && recordedHost == host);
        
        P port = uint8_t(-1);
        assert(gateway.lookUp(key, port));
        int gatewayNextHop = graph->adjacencyList[gatewayId][port].to;
        uint16_t nextHop = graph->shortestPathTo[host][gatewayId].nextHop;
        assert(gatewayNextHop == nextHop);
      }
      #endif
      
      if (gateways.empty()) {
        gateways.push_back(gateway);  // just store 1 gateway
      } else {
        asm volatile (""::"g" (gateway): "memory");
      }
    }
  }
  
  template<bool isL2>
  typename std::enable_if<!isL2, void>::type constructGateway() {
    auto hostIds = ControlPlane<K, isL2>::getHostIds();
    auto gatewayIds = ControlPlane<K, isL2>::getGatewayIds();
    ostringstream oss;
    oss << "Gateway construction (" << gatewayIds.size() << " gateways)";
    Clocker clocker(oss.str());
  }
  
  template<bool isL2>
  typename std::enable_if<isL2, void>::type constructRouter() {
    auto hostIds = ControlPlane<K, isL2>::getHostIds();
    auto routerIds = ControlPlane<K, l2>::getRouterIds();
    
    routerIds.resize(10);
    ostringstream oss;
    oss << "Router construction (" << routerIds.size() << " routers)";
    Clocker clocker(oss.str());
    
    ControlPlaneCuckooMap<K, P, Match, false, DL> lv2Temp;
    DataPlaneCuckooMap<K, P, Match, DL> lv1Temp(lv2Temp);
    
    lv1Temp.num_buckets_ = level1->num_buckets_;
    lv1Temp.buckets_.clear();
    lv1Temp.buckets_.resize(level1->buckets_.size());
    
    for (int i = 0; i < 2 + 1; ++i) {
      lv1Temp.h[i] = level1->h[i];
    }
    
    for (int i = 0; i < level1->buckets_.size(); ++i) {
      auto &bsrc = level1->buckets_[i];
      auto &bdst = lv1Temp.buckets_[i];
      bdst.occupiedMask = bsrc.occupiedMask;
      
      for (int slot = 0; slot < 4; ++slot) {
        if (!(bsrc.occupiedMask & (1 << slot))) continue;
        
        bdst.keyDigests[slot] = level1->h[2](bsrc.keys[slot]);
      }
    }
    
    lv2Temp.entryCount = level2->entryCount;
    lv2Temp.cpq_.reset();
    lv2Temp.num_buckets_ = level2->num_buckets_;
    lv2Temp.buckets_.resize(level2->buckets_.size());
    
    for (int i = 0; i < 2 + 1; ++i) {
      lv2Temp.h[i] = level2->h[i];
    }
    
    for (int i = 0; i < level2->buckets_.size(); ++i) {
      auto &bsrc = level2->buckets_[i];
      auto &bdst = lv2Temp.buckets_[i];
      bdst.occupiedMask = bsrc.occupiedMask;
      
      for (int slot = 0; slot < 4; ++slot) {
        if (!(bsrc.occupiedMask & (1 << slot))) continue;
        
        bdst.keys[slot] = bsrc.keys[slot];
      }
    }
    
    for (int routerId: routerIds) {
      unordered_map<uint16_t, P> hostToPort;
      
      for (int i = 0; i < hostIds.size(); ++i) {
        int host = hostIds[i];
        vector<Graph<>::ShortestPathCell> path = graph->shortestPathTo[host];
        assert(!path.empty());  // asserting the host is a real host
        uint16_t nextHop = path[routerId].nextHop;
        
        P portNum = P(std::find(graph->adjacencyList[routerId].begin(),
                                graph->adjacencyList[routerId].end(),
                                Graph<>::AdjacencyMatrixCell({nextHop, 0})) -
                      graph->adjacencyList[routerId].begin());
        assert(portNum != (P) (-1));
        hostToPort.insert(make_pair(host, portNum));
      }
      
      for (int i = 0; i < level1->buckets_.size(); ++i) {
        auto &bsrc = level1->buckets_[i];
        auto &bdst = lv1Temp.buckets_[i];
        
        for (int slot = 0; slot < 4; ++slot) {
          if (!(bsrc.occupiedMask & (1 << slot))) continue;
          
          bdst.values[slot] = hostToPort[bsrc.values[slot]];
        }
      }
      
      for (int i = 0; i < level2->buckets_.size(); ++i) {
        auto &bsrc = level2->buckets_[i];
        auto &bdst = lv2Temp.buckets_[i];
        
        for (int slot = 0; slot < 4; ++slot) {
          if (!(bsrc.occupiedMask & (1 << slot))) continue;
          
          bdst.values[slot] = hostToPort[bsrc.values[slot]];
        }
      }
      
      TwoLevelCuckooRouter<K, Match, l2, DL> router(lv1Temp, lv2Temp);
  
      #ifdef FULL_DEBUG
      uint32_t count = 0;
      unordered_map<K, uint16_t, Hasher32<K>> map;
      for (auto &bucket: level1->buckets_) {  // all buckets
        for (int slot = 0; slot < 4; ++slot) {
          if (bucket.occupiedMask & (1ULL << slot)) {
            const K &key = bucket.keys[slot];
            uint16_t host = bucket.values[slot];
            
            map.insert(make_pair(key, host));
          }
        }
      }
      
      for (auto &bucket: level2->buckets_) {  // all buckets
        for (int slot = 0; slot < 4; ++slot) {
          if (bucket.occupiedMask & (1ULL << slot)) {
            const K &key = bucket.keys[slot];
            uint16_t host = bucket.values[slot];
            map.insert(make_pair(key, host));
          }
        }
      }
      
      for (auto &pair:map) {
        const K &key = pair.first;
        const uint16_t host = pair.second;
        
        P port = P(-1);
        assert(router.lookUp(key, port));
        assert(port != (P) (-1));
        
        int gatewayNextHop = graph->adjacencyList[routerId][port].to;
        uint16_t nextHop = graph->shortestPathTo[host][routerId].nextHop;
        
        assert (gatewayNextHop == nextHop);
      }
      #endif
      
      if (routers.empty()) {
        routers.push_back(router);  // just store 1 router
      } else {
        asm volatile (""::"g" (router): "memory");
      }
    }
  }
  
  template<bool isL2>
  typename std::enable_if<!isL2, void>::type constructRouter() {
    auto routerIds = ControlPlane<K, l2>::getRouterIds();
    
    routerIds.resize(1);
    ostringstream oss;
    oss << "Router construction (" << routerIds.size() << " routers)";
    Clocker clocker(oss.str());
    
    for (int routerId: routerIds) {
      TwoLevelCuckooRouter<K, Match, l2, DL> router(*level1, *level2);
      
      if (routers.empty()) {
        routers.push_back(router);  // just store 1 router
      } else {
        asm volatile (""::"g" (router): "memory");
      }
    }
  }
  
  void construct() override {
    constructGateway<l2>();
    constructRouter<l2>();
  }
  
  uint64_t getGatewayMemoryCost() const override {
    uint64_t result = 0;
    
    for (const TwoLevelCuckooGateway<K, Match, DL> &gateway: gateways) {
      result += gateway.getMemoryCost();
    }
    
    return result;
  }
  
  uint64_t getRouterMemoryCost() const override {
    uint64_t result = 0;
    
    for (const TwoLevelCuckooRouter<K, Match, l2, DL> &router: routers) {
      result += router.getMemoryCost();
    }
    
    return result;
  }
  
  uint64_t getControlPlaneMemoryCost() const override {
    return level1->getMemoryCost() + level2->getMemoryCost();
  }

private:
  void rebuild(const K &k, uint16_t host) {
    Counter::count(getName(), "level1 rebuild");
    Clocker rebuild("Cuckoo level1 rebuild");
    
    unordered_map<K, uint16_t, Hasher32<K>> map = level1->toMap();
    unordered_map<K, uint16_t, Hasher32<K>> map2 = level2->toMap();
    map.insert(map2.begin(), map2.end());
    
    map.insert(make_pair(k, host));
    
    delete level1;
    delete level2;
    level1 = nullptr;
    level2 = nullptr;
    
    uint32_t capacity = ControlPlane<K, l2>::capacity;
    while (!level1) {
      Counter::count(getName(), "level1 re-rebuild");
      
      level1 = new ControlPlaneCuckooMap<K, uint16_t, Match, true, DL, 2, 4>(capacity);
      level2 = new ControlPlaneCuckooMap<K, uint16_t, Match, false, DL, 2, 4>(capacity / 10);
      
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
  
  void rebuildL2(const K &k, uint16_t host) {
    Counter::count(getName(), "level2 full, only rebuild lv2");
    unordered_map<K, uint16_t, Hasher32<K>> map = level2->toMap();
    map.insert(make_pair(k, host));
    
    delete level2;
    level2 = 0;
    
    while (!level2) {
      level2 = new ControlPlaneCuckooMap<K, uint16_t, Match, false, DL, 2, 4>(ControlPlane<K, l2>::capacity / 10);
      
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
