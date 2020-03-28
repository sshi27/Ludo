//
// Created by ssqstone on 2018/7/17.
//

#pragma once

#include "../common.h"
#include "../control_plane.h"
#include "../CuckooPresized/cuckoo_map.h"
#include "bloom_filter.h"

template<class K, bool l2>
class BloomRouter {
  typedef typename std::conditional<l2, uint8_t, uint16_t>::type P;
public:
  vector<BloomFilter<K>> filters;
  vector<BloomFilter<K, 4>> countingFilters;
  uint16_t numOfPorts;
  
  inline explicit BloomRouter(uint16_t numOfPorts) : numOfPorts(numOfPorts) {
  }
  
  inline bool lookUp(const K &k, P &out) const {
    static P start = P(-1);
    start = P((start + 1) % numOfPorts);
    
    for (uint16_t i = 0; i < numOfPorts; ++i) {
      P port = P((i + start) % numOfPorts);
      const BloomFilter<K> &filter = filters[port];
      
      if (filter.getCapacity() && filter.isMember(k)) {
        out = port;
        return true;
      }
    }
    
    return false;
  }
  
  inline void insert(const K &k, P portIdx) {
    countingFilters[portIdx].insert(k);
    filters[portIdx].insert(k);
  }
  
  inline void erase(const K &k, P portIdx) {
    uint64_t toErase = countingFilters[portIdx].erase(k);
    filters[portIdx].mask(k, toErase);
  }
  
  inline void modify(const K &k, P before,
                     P after) {
    erase(k, before);
    insert(k, after);
  }
  
  inline vector<P> matchAll(const K &k) const {
    vector<P> result;
    
    for (P port = 0; port < numOfPorts; ++port) {
      const BloomFilter<K> &filter = filters[port];
      
      if (filter.getCapacity() && filter.isMember(k)) result.push_back(port);
    }
    
    return result;
  };
};

template<class K, bool l2, class Match = uint8_t>
class BloomFilterControlPlane : public ControlPlane<K, l2> {
  typedef typename std::conditional<l2, uint8_t, uint16_t>::type P;
public:
  ControlPlaneCuckooMap<K, uint16_t, Match, false, 2, 4> *keyToHost;
  
  vector<ControlPlaneCuckooMap<K, uint8_t, Match, false, 2, 4>> gateways;
  vector<BloomRouter<K, l2>> routers;
  
  using ControlPlane<K, l2>::insert;
  using ControlPlane<K, l2>::remove;
  using ControlPlane<K, l2>::graph;
  using ControlPlane<K, l2>::capacity;
  
  
  explicit BloomFilterControlPlane(uint32_t capacity)
    : ControlPlane<K, l2>(capacity),
      keyToHost(new ControlPlaneCuckooMap<K, uint16_t, Match, false, 2, 4>(capacity)) {
  }
  
  void scenario(int topo) override {
    ControlPlane<K, l2>::scenario(topo);
    
    gateways.reserve(ControlPlane<K, l2>::getGatewayIds().size());
    routers.reserve(ControlPlane<K, l2>::getRouterIds().size());
  };
  
  virtual ~BloomFilterControlPlane() {
    if (keyToHost)
      delete keyToHost;
  }
  
  inline const char *getName() const override {
    return "Bloom-CP";
  }
  
  inline void insert(const K &k, uint16_t host) override {
    const K *result = keyToHost->insert(k, host);
    
    if (result != &k) {
      rebuild(k, host);
    } else {
      Counter::count(getName(), "cuckoo simple add");
    }
  }
  
  void rebuild(const K &k, uint16_t host) {
    Counter::count(getName(), "cuckoo full, rebuild");
    unordered_map<K, uint16_t, Hasher32<K>> map = keyToHost->toMap();
    map.insert(make_pair(k, host));
    
    delete keyToHost;
    keyToHost = 0;
    
    while (!keyToHost) {
      keyToHost = new ControlPlaneCuckooMap<K, uint16_t, Match, false, 2, 4>(ControlPlane<K, l2>::capacity);
      
      for (auto it = map.begin(); it != map.end(); ++it) {
        const K *result = keyToHost->insert(it->first, it->second);
        
        if (result != &it->first) {
          delete keyToHost;
          keyToHost = 0;
          break;
        }
      }
    }
  }
  
  void remove(const K &k) override {
    keyToHost->remove(k);
  }
  
  bool lookUp(const K &k, uint16_t &out) const {
    return keyToHost->lookUp(k, out);
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
    
    for (int gatewayId: gatewayIds) {
      unordered_map<uint16_t, P> hostToPort;
      
      for (int i = 0; i < hostIds.size(); ++i) {
        int host = hostIds[i];
        vector<Graph<>::ShortestPathCell> path = graph->shortestPathTo[host];
        assert(!path.empty());  // asserting the host is a real host
        
        uint16_t nextHop = path[gatewayId].nextHop;
        
        uint8_t portNum = (uint8_t) (std::find(graph->adjacencyList[gatewayId].begin(),
                                               graph->adjacencyList[gatewayId].end(),
                                               Graph<>::AdjacencyMatrixCell({nextHop, 0})) -
                                     graph->adjacencyList[gatewayId].begin());
        assert(portNum != uint8_t(-1));
        hostToPort.insert(make_pair(host, portNum));
      }
      
      ControlPlaneCuckooMap<K, uint8_t, Match, false, 2, 4> gateway = keyToHost->Compose(hostToPort);
  
      #ifdef FULL_DEBUG
      for (auto &bucket: keyToHost->buckets_) {  // all buckets
        for (int slot = 0; slot < 4; ++slot) {
          if (bucket.occupiedMask & (1ULL << slot)) {
            const K &key = bucket.keys[slot];
            const uint16_t host = bucket.values[slot];
            
            P port = P(-1);
            assert(gateway.lookUp(key, port));
            assert(port != uint8_t(-1));
            
            int gatewayNextHop = graph->adjacencyList[gatewayId][port].to;
            uint16_t nextHop = graph->shortestPathTo[host][gatewayId].nextHop;
            assert(gatewayNextHop == nextHop);
          }
        }
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
    
    routerIds.resize(3);
    ostringstream oss;
    oss << "Router construction (" << routerIds.size() << " routers)";
    Clocker clocker(oss.str());
    
    for (int routerId: routerIds) {
      P maxPort = 0;
      // different routers have different key-port mappings
      unordered_map<uint16_t, P> hostToPort;
      for (int i = 0; i < hostIds.size(); ++i) {
        int host = hostIds[i];
        vector<Graph<>::ShortestPathCell> path = graph->shortestPathTo[host];
        assert(!path.empty());  // asserting the host is a real host
        
        uint16_t nextHop = path[routerId].nextHop;
        
        P portNum = (P) (
          std::find(graph->adjacencyList[routerId].begin(),
                    graph->adjacencyList[routerId].end(),
                    Graph<>::AdjacencyMatrixCell({nextHop, 0})) -
          graph->adjacencyList[routerId].begin());
        assert(portNum != P(-1));
        hostToPort.insert(make_pair(host, portNum));
        
        maxPort = max(maxPort, portNum);
      }
      
      // I can never decide the filter size before completely traversing the two "key to host" and "host to port" mappings
      unordered_map<P, unordered_set<K, Hasher32<K>>> portToKeys;
      for (P p = 0; p <= maxPort; ++p) {
        portToKeys.insert(make_pair(p, unordered_set<K, Hasher32<K>>()));
      }
      
      for (auto &bucket: keyToHost->buckets_) {
        for (int slot = 0; slot < 4; ++slot) {
          if (bucket.occupiedMask & (1ULL << slot)) {
            const K &key = bucket.keys[slot];
            const uint16_t host = bucket.values[slot];
            const P port = hostToPort[host];
            
            portToKeys[port].insert(key);
          }
        }
      }
      
      BloomRouter<K, l2> router(P(maxPort + 1));
      
      for (P p = 0; p <= maxPort; ++p) {
        router.filters.push_back(BloomFilter<K>(uint32_t(portToKeys[p].size())));
        auto &filter = router.filters.back();
        
        router.countingFilters.push_back(BloomFilter<K, 4>(uint32_t(portToKeys[p].size())));
        auto &countingFilter = router.countingFilters.back();
        
        for (const K &k:portToKeys[p]) {
          countingFilter.insert(k);
          filter.insert(k);
        }
      }
  
      #ifdef FULL_DEBUG
      uint32_t count = 0;
      for (auto &bucket: keyToHost->buckets_) {  // all buckets
        for (int slot = 0; slot < 4; ++slot) {
          if (bucket.occupiedMask & (1ULL << slot)) {
            const K &key = bucket.keys[slot];
            const uint16_t host = bucket.values[slot];
            
            P port;
            router.lookUp(key, port);
            
            int gatewayNextHop = graph->adjacencyList[routerId][port].to;
            uint16_t nextHop = graph->shortestPathTo[host][routerId].nextHop;
            
            if (gatewayNextHop != nextHop) {
              Counter::count(getName(), "router lookup collision");
              count++;
              if (count > 0.1 * keyToHost->entryCount) {
                cerr << "*******Too many bloom collisions" << endl;
                auto result = router.matchAll(key);
                for (P port: result) {
                  cerr << "  " << port;
                }
                cerr << endl;
              }
            }
          }
        }
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
    
    // I can never decide the filter size before completely traversing the two "key to host" and "host to port" mappings
    P maxPort = static_cast<P>(graph->nodes.size() - 1);
    for (int routerId: routerIds) {
      unordered_map<P, unordered_set<K, Hasher32<K>>> portToKeys;
      for (P p = 0; p <= maxPort; ++p) {
        portToKeys.insert(make_pair(p, unordered_set<K, Hasher32<K>>()));
      }
      
      for (auto &bucket: keyToHost->buckets_) {
        for (int slot = 0; slot < 4; ++slot) {
          if (bucket.occupiedMask & (1ULL << slot)) {
            const K &key = bucket.keys[slot];
            const uint16_t host = bucket.values[slot];
            const P port = host;
            
            portToKeys[port].insert(key);
          }
        }
      }
      
      BloomRouter<K, l2> router(P(maxPort + 1));
      
      for (P p = 0; p <= maxPort; ++p) {
        router.filters.push_back(BloomFilter<K>(uint32_t(portToKeys[p].size())));
        auto &filter = router.filters.back();
        
        router.countingFilters.push_back(BloomFilter<K, 4>(uint32_t(portToKeys[p].size())));
        auto &countingFilter = router.countingFilters.back();
        
        for (const K &k:portToKeys[p]) {
          countingFilter.insert(k);
          filter.insert(k);
        }
      }
  
      #ifdef FULL_DEBUG
      uint32_t count = 0;
      for (auto &bucket: keyToHost->buckets_) {  // all buckets
        for (int slot = 0; slot < 4; ++slot) {
          if (bucket.occupiedMask & (1ULL << slot)) {
            const K &key = bucket.keys[slot];
            const uint16_t host = bucket.values[slot];
            
            P port;
            router.lookUp(key, port);
            
            int gatewayNextHop = graph->adjacencyList[routerId][port].to;
            uint16_t nextHop = graph->shortestPathTo[host][routerId].nextHop;
            
            if (gatewayNextHop != nextHop) {
              Counter::count(getName(), "router lookup collision");
              count++;
              if (count > 0.1 * keyToHost->entryCount) {
                cerr << "*******Too many bloom collisions" << endl;
                auto result = router.matchAll(key);
                for (P port: result) {
                  cerr << "  " << port;
                }
                cerr << endl;
              }
            }
          }
        }
      }
      #endif
      
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
  
  inline bool lookUp(const K &k, uint16_t &out) const override {
    return keyToHost->lookUp(k, out);
  }
  
  inline uint8_t lookUpGateway(uint16_t id, const K &key) const override {
    uint8_t port = uint8_t(-1);
    return gateways[id].lookUp(key, port), port;
  }
  
  inline P lookUpRouter(uint16_t id, const K &key) const override {
    P port = static_cast<P>(-1);
    return routers[id].lookUp(key, port), port;
  }
  
  uint64_t getGatewayMemoryCost() const override {
    uint64_t result = 0;
    
    for (const ControlPlaneCuckooMap<K, uint8_t, Match, false, 2, 4> &gateway: gateways) {
      result += gateway.getMemoryCost();
    }
    
    return result;
  }
  
  uint64_t getRouterMemoryCost() const override {
    uint64_t result = 0;
    
    for (const BloomRouter<K, l2> &router: routers) {
      for (const BloomFilter<K> &filter: router.filters) {
        result += filter.getMemoryCost();
      }
    }
    
    return result;
  }
  
  uint64_t getControlPlaneMemoryCost() const override {
    return keyToHost->getMemoryCost();
  }
};
