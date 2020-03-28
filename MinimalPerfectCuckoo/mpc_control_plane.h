//
// Created by ssqstone on 2018/7/6.
//
#pragma once

#include "../control_plane.h"
#include "minimal_perfect_cuckoo.h"

template<class K, bool l2, uint8_t DL>
class MPCControlPlane;

template<class K>
class MPCGateWay : public DataPlaneMinimalPerfectCuckoo<K, uint32_t, 32> {
private:
  vector<K> keys{};
  vector<uint8_t> ports{};

public:
  inline bool lookUp(const K &k, uint8_t &out) const {
    uint32_t index = DataPlaneMinimalPerfectCuckoo<K, uint32_t, 32>::lookUp(k);
    if (index >= keys.size() || !(keys[index] == k)) return false;
    
    out = ports[index];
    return true;
  }
  
  inline uint64_t getMemoryCost() const {
    return keys.size() * sizeof(keys[0]) + ports.size() * sizeof(ports[0]) +
           DataPlaneMinimalPerfectCuckoo<K, uint32_t, 32>::getMemoryCost();
  }
};

template<class K, bool l2, uint8_t DL = 0>
class MPCControlPlane : public ControlPlane<K, l2> {
  static_assert(DL == 0 || !l2, "You don't need digests while allowing gateways");
  
  typedef typename std::conditional<l2, uint8_t, uint16_t>::type P;

public:
  // a map with set and iteration support, and exportable to data plane
  ControlPlaneMinimalPerfectCuckoo<K, uint16_t, 16, DL> keyToHost;
  vector<DataPlaneMinimalPerfectCuckoo<K, P, l2 ? 8 : 16, DL>> routers;
  vector<MPCGateWay<K>> gateways;
  
  using ControlPlane<K, l2>::insert;
  using ControlPlane<K, l2>::remove;
  using ControlPlane<K, l2>::graph;
  using ControlPlane<K, l2>::capacity;
  
  explicit MPCControlPlane(uint32_t capacity)
    : ControlPlane<K, l2>(capacity), keyToHost(capacity) {
  }
  
  void scenario(int topo) override {
    ControlPlane<K, l2>::scenario(topo);
    
    gateways.reserve(ControlPlane<K, l2>::getGatewayIds().size());
    routers.reserve(ControlPlane<K, l2>::getRouterIds().size());
  };
  
  inline const char *getName() const override {
    return "MPC-CP";
  }
  
  inline void insert(const K &k, uint16_t host) override {
    keyToHost.insert(make_pair(k, host));
  }
  
  void remove(const K &k) override {
    keyToHost.erase(k);
  }
  
  using ControlPlane<K, l2>::getGatewayIds;
  using ControlPlane<K, l2>::getRouterIds;
  using ControlPlane<K, l2>::getHostIds;
  
  template<bool isL2>
  typename std::enable_if<isL2, void>::type constructGateway() {
    throw runtime_error("should work, but not implemented");
  }
  
  template<bool isL2>
  typename std::enable_if<!isL2, void>::type constructGateway() {
    throw runtime_error("impossible");
  }
  
  template<bool isL2>
  typename std::enable_if<isL2, void>::type constructRouter() {
    auto hostIds = ControlPlane<K, isL2>::getHostIds();
    auto routerIds = ControlPlane<K, l2>::getRouterIds();
    
    routerIds.resize(10);
    ostringstream oss;
    oss << "Router construction (" << routerIds.size() << " routers)";
    Clocker clocker(oss.str());
    
    auto hosts = keyToHost.getValues();
    
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
      
      DataPlaneMinimalPerfectCuckoo<K, P, l2 ? 8 : 16, DL> router(keyToHost, hostToPort);
  
      #ifdef FULL_DEBUG
      for (int i = 0; i < keyToHost.size(); ++i) {
        const K &key = keyToHost.keys[i];
        const uint16_t host = keyToHost.values[i];
        P port;
        assert(router.lookUp(key, port) && port != (P) (-1));
        
        int gatewayNextHop = graph->adjacencyList[routerId][port].to;
        uint16_t nextHop = graph->shortestPathTo[host][routerId].nextHop;
        assert(gatewayNextHop == nextHop);
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
      DataPlaneMinimalPerfectCuckoo<K, P, l2 ? 8 : 16, DL> router(keyToHost);
      
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
    return keyToHost.lookUp(k, out);
  }
  
  inline uint8_t lookUpGateway(uint16_t id, const K &key) const override {
    uint8_t port = uint8_t(-1);
    gateways[id].lookUp(key, port);
    return port;
  }
  
  inline P lookUpRouter(uint16_t id, const K &key) const override {
    P port;
    return routers[id].lookUp(key, port) ? port : static_cast<P>(-1);
  }
  
  uint64_t getGatewayMemoryCost() const override {
    uint64_t result = 0;
    
    for (const MPCGateWay<K> &gateway: gateways) {
      result += gateway.getMemoryCost();
    }
    
    return result;
  }
  
  uint64_t getRouterMemoryCost() const override {
    uint64_t result = 0;
    
    for (const auto &router: routers) {
      result += router.getMemoryCost();
    }
    
    return result;
  }
  
  uint64_t getControlPlaneMemoryCost() const override {
    return keyToHost.getMemoryCost();
  }
};
