//
// Created by ssqstone on 2018/7/5.
//
#pragma once

#include <network/graph.h>
#include "common.h"

/// both the number of hosts and the number of ports are uint16_t and cannot be larger than 65530
const int MAX_HOSTS = 65530;

template<class K, bool l2>
class ControlPlane {
  typedef typename std::conditional<l2, uint8_t, uint16_t>::type P;
public:
  uint32_t capacity;
  Graph<> *graph = nullptr;
  static unordered_set<ControlPlane<K, l2> *> instances;
  
  explicit ControlPlane(uint32_t capacity) : capacity(capacity) {
    instances.insert(this);
  }
  
  virtual ~ControlPlane() {
    instances.erase(this);
    if (graph)
      delete graph;
  }
  
  inline virtual void insert(const K &k, uint16_t host) = 0;
  
  inline virtual void remove(const K &k) = 0;
  
  virtual void construct() = 0;
  
  inline virtual bool lookUp(const K &k, uint16_t &out) const = 0;
  
  inline virtual const char *getName() const = 0;
  
  template<bool alien = false>
  inline void simulateRoutingTmpl(const K &k) {
    static uint16_t startPoint = uint16_t(-1);
    startPoint = uint16_t(uint16_t(startPoint + 1) % (l2 ? 3 : capacity));
    double cost = 0;
    
    uint16_t current = startPoint;
    uint16_t nextPort = l2 ? lookUpGateway(current, k) : lookUpRouter(current, k);
    int ttl = 10;
    
    while (nextPort != uint16_t(-1) && ttl > 0) {
      cost += graph->adjacencyList[current][nextPort].cost;
      current = graph->adjacencyList[current][nextPort].to;
      
      nextPort = (!l2 || current >= 3) ? lookUpRouter(uint16_t(current - (l2 ? 3 : 0)), k) : lookUpGateway(current, k);
      --ttl;
    }
    
    if (ttl <= 0) Counter::count(getName(), alien ? "Alien Routing Timeout" : "Routing Timeout");
    
    if (alien) Counter::count(getName(), "False Positives", 10 - ttl);
    
    Counter::count(getName(), alien ? "Alien Routing Cost" : "Routing Cost", cost);
  }
  
  inline virtual void simulateRouting(const K &k) {
    simulateRoutingTmpl<false>(k);
  }
  
  inline virtual void simulateAlienRouting(const K &k) {
    simulateRoutingTmpl<true>(k);
  }
  
  inline vector<uint16_t> getGatewayIds() const {
    if (!l2) return {};
    
    return {0, 1, 2};
  }
  
  vector<uint16_t> getRouterIds() const {
    vector<uint16_t> result;
    
    set<uint16_t> gateways;
    for (uint16_t g: getGatewayIds()) {
      gateways.insert(g);
    }
    
    for (uint16_t i = 0; i < graph->nodes.size(); ++i) {
      if (gateways.find(i) == gateways.end()) {
        result.push_back(i);
      }
    }
    
    return result;
  }
  
  vector<uint16_t> getHostIds() const {
    vector<uint16_t> result;
    
    set<uint16_t> gateways;
    for (uint16_t g: getGatewayIds()) {
      gateways.insert(g);
    }
    
    for (uint16_t i = 0; i < graph->nodes.size(); ++i) {
      if (graph->nodes[i].hostAttached && gateways.find(i) == gateways.end()) {
        result.push_back(i);
      }
    }
    
    return result;
  }
  
  virtual uint64_t getGatewayMemoryCost() const = 0;
  
  virtual uint64_t getRouterMemoryCost() const = 0;
  
  virtual uint64_t getControlPlaneMemoryCost() const = 0;
  
  /// simulate send to a gateway, a router, or a host attached to the router
  virtual inline uint8_t lookUpGateway(uint16_t id, const K &key) const = 0;
  
  virtual inline typename std::conditional<l2, uint8_t, uint16_t>::type
  lookUpRouter(uint16_t id, const K &key) const = 0;
  
  virtual void scenario(int topo) {
    static vector<string> domainTopoFiles{"1221.txt", "1239.txt", "1755.txt", "2914.txt", "3257.txt", "3697.txt",
                                          "4755.txt", "7018.txt"};
    
    if (graph != nullptr) {
      delete graph;
      graph = nullptr;
    }
    
    Clocker clocker("setup topology");
    
    if (topo < 8) {
      string domainFile = "../input/topo/" + ((topo == -1) ? string("test.txt") : domainTopoFiles[topo]);
      ifstream fin(domainFile);
      
      string line;
      std::getline(fin, line);
      
      uint32_t nodeCnt, linkCnt;
      
      istringstream iss(line);
      iss >> nodeCnt >> linkCnt;
      
      graph = new Graph<>(nodeCnt);
      
      for (uint16_t id = 0; id < nodeCnt; ++id) {
        std::getline(fin, line);
        std::istringstream iss(line);
        
        int isRoot;
        iss >> isRoot >> isRoot;
        
        graph->addVertex({id, ((isRoot != 0)), nullptr});
      }
      
      for (int id = 0; id < linkCnt; ++id) {
        std::getline(fin, line);
        std::istringstream iss(line);
        
        uint16_t u, v;
        double w;
        iss >> u >> v >> w;
        
        graph->addEdge({u, v, w});
      }
    } else {
      graph = new CompleteGraph<>(topo);
    }
    clocker.stop();
    
    graph->calculateShortestPaths();
  }
};

template<class K, bool l2>
unordered_set<ControlPlane<K, l2> *> ControlPlane<K, l2>::instances;
