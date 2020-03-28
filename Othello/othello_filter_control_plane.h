//
// Created by ssqstone on 2018/7/6.
//
#pragma once

#include "../control_plane.h"
#include "control_plane_othello.h"
#include "data_plane_othello.h"

template<class K, bool l2, uint8_t DL>
class OthelloFilterControlPlane;

template<class K>
class OthelloGateWay : public DataPlaneOthello<K, uint32_t, 32, 0> {
  template<class K1, bool l2, uint8_t DL> friend
  class OthelloFilterControlPlane;

private:
  vector<K> keys{};
  vector<uint8_t> ports{};

public:
  inline bool lookUp(const K &k, uint8_t &out) const {
    uint32_t index = DataPlaneOthello<K, uint32_t, 32, 0>::lookUp(k);
    if (index >= keys.size() || !(keys[index] == k)) return false;
    
    out = ports[index];
    return true;
  }
  
  uint64_t getMemoryCost() const {
    return keys.size() * sizeof(keys[0]) + ports.size() * sizeof(ports[0]);
  }
};

template<class K, bool l2, uint8_t DL = 0>
class OthelloFilterControlPlane : public ControlPlane<K, l2> {
  static_assert(DL == 0 || !l2, "You don't need digests while allowing gateways");
  
  typedef typename std::conditional<l2, uint8_t, uint16_t>::type P;

public:
  // a map with set and iteration support, and exportable to data plane
  ControlPlaneOthello<K, uint16_t, 16, DL, true> keyToHost;
  vector<DataPlaneOthello<K, P, l2 ? 8 : 16, DL>> routers;
  vector<OthelloGateWay<K>> gateways;
  
  using ControlPlane<K, l2>::insert;
  using ControlPlane<K, l2>::remove;
  using ControlPlane<K, l2>::graph;
  using ControlPlane<K, l2>::capacity;
  
  
  explicit OthelloFilterControlPlane(uint32_t capacity)
    : ControlPlane<K, l2>(capacity), keyToHost(capacity) {
  }
  
  void scenario(int topo) override {
    ControlPlane<K, l2>::scenario(topo);
    
    gateways.reserve(ControlPlane<K, l2>::getGatewayIds().size());
    routers.reserve(ControlPlane<K, l2>::getRouterIds().size());
  };
  
  inline const char *getName() const override {
    return "Othello-CP";
  }
  
  inline void insert(const K &k, uint16_t host) override {
    keyToHost.insert(make_pair(k, host));
  }
  
  void remove(const K &k) override {
      keyToHost.remove(k);
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
    
    Clocker templ("template creation");
    vector<uint64_t> mem;
    mem.reserve((keyToHost.ma + keyToHost.mb + 1) / 2);
    
    bool lower = true;
    for (uint32_t element: keyToHost.getIndexMemory()) {
      if (lower) {
        mem.push_back(uint64_t(element));
      } else {
        mem.back() |= uint64_t(element) << 32;
      }
      
      lower = !lower;
    }
    
    auto hosts = keyToHost.getValues();
    templ.stop();
    
    for (int gatewayId: gatewayIds) {
      OthelloGateWay<K> gateway;
      gateway.hab = keyToHost.hab;
      gateway.hd = keyToHost.hd;
      gateway.ma = keyToHost.ma;
      gateway.mb = keyToHost.mb;
      gateway.keys = keyToHost.keys;  // copy keys, and construct port array later
      gateway.mem = mem;  // the lookUp structure gives the index of the key array and value array
      
      // construct port array
      for (int i = 0; i < keyToHost.size(); ++i) {
        int host = hosts[i];    // the destination of the i-th key
        vector<Graph<>::ShortestPathCell> path = graph->shortestPathTo[host];
        assert(!path.empty());  // asserting the host is a real host
        uint16_t nextHop = path[gatewayId].nextHop;
        
        P portNum = (uint8_t) (std::find(graph->adjacencyList[gatewayId].begin(),
                                         graph->adjacencyList[gatewayId].end(),
                                         Graph<>::AdjacencyMatrixCell({nextHop, 0})) -
                               graph->adjacencyList[gatewayId].begin());
        assert(portNum != uint8_t(-1));
        gateway.ports.push_back(portNum);
      }
  
      #ifdef FULL_DEBUG
      for (int i = 0; i < keyToHost.size(); ++i) {
        const K &key = keyToHost.keys[i];
        const uint16_t host = keyToHost.values[i];
        P port;
        gateway.lookUp(key, port);
        
        assert(port != uint8_t(-1));
        
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
    
    ControlPlaneOthello<K, P, l2 ? 8 : 16, DL, true> keyToPort;
    // copy the graph from the control plane
    keyToPort.connectivityForest = keyToHost.connectivityForest;
    keyToPort.indMem = keyToHost.indMem;
    keyToPort.keys = keyToHost.keys;
    keyToPort.keyCnt = keyToHost.keyCnt;
    keyToPort.hab = keyToHost.hab;
    keyToPort.hd = keyToHost.hd;
    keyToPort.head = keyToHost.head;
    keyToPort.nextAtA = keyToHost.nextAtA;
    keyToPort.nextAtB = keyToHost.nextAtB;
    keyToPort.ma = keyToHost.ma;
    keyToPort.mb = keyToHost.mb;
    keyToPort.minimalKeyCapacity = keyToHost.minimalKeyCapacity;
    keyToPort.memResize();
    keyToPort.values.resize(keyToPort.keyCnt);
    
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
      
      // different routers have different key-port mappings
      // construct port array
      for (int i = 0; i < keyToHost.size(); ++i) {
        keyToPort.values[i] = hostToPort[keyToHost.values[i]];
      }
      
      // update the lookUp structure
      keyToPort.fillValue();
      
      DataPlaneOthello<K, P, l2 ? 8 : 16, DL> router(keyToPort);
  
      #ifdef FULL_DEBUG
      for (int i = 0; i < keyToHost.size(); ++i) {
        const K &key = keyToHost.keys[i];
        const uint16_t host = keyToHost.values[i];
        P port;
        assert(keyToPort.lookUp(key, port) && port != (P) (-1));
        
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
      DataPlaneOthello<K, P, l2 ? 8 : 16, DL> router(keyToHost);
      
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
    
    for (const OthelloGateWay<K> &gateway: gateways) {
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
