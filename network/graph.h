//
// Created by ssqstone on 2018/7/5.
//
#pragma once

#include "../common.h"

template<class T = void>
class Graph {
public:
  struct Node {
    uint16_t id;
    bool hostAttached;   // if the node is host, then there are no out going edges
    T *data;
  };

  struct Edge {
    uint16_t from, to;
    double cost;

    inline bool operator==(const Edge &another) const {
      return tie(from, to, cost) == tie(another.from, another.to, another.cost);
    }
  };

  struct ShortestPathCell {
    double cost;
    uint16_t nextHop;
  };

  struct AdjacencyMatrixCell {
    uint16_t to;
    double cost;

    inline bool operator==(const AdjacencyMatrixCell &other) const {
      return to == other.to;
    }

    inline bool operator<(const AdjacencyMatrixCell &other) const {
      return to < other.to;
    }
  };

  vector<Node> nodes;
  vector<vector<AdjacencyMatrixCell>> adjacencyList;
  vector<vector<ShortestPathCell>> shortestPathTo;       // [i][j] is the nextHop and cost of j to i

  /// After the construction, the graph is a complete graph
  explicit Graph(size_t capacity) {
    nodes.reserve(capacity);
    shortestPathTo.reserve(capacity);

    for (uint16_t i = 0; i < capacity; ++i) {
      adjacencyList.push_back({{i, 0}});

      vector<ShortestPathCell> v2(capacity);
      for (uint16_t j = 0; j < capacity; ++j) {
        v2[j] = {i == j ? 0.0 : 	3.40282347e+38F, i};
      }
      shortestPathTo.push_back(v2);
    }
  }

  /// must be called in the order of 0, 1, 2, ...
  /// \param n
  void addVertex(Node n) {
    nodes.push_back(n);
  }

  /// multiple edges between same two nodes are not checked
  /// \param e
  virtual void addEdge(Edge e) {
    adjacencyList[e.from].push_back({e.to, e.cost});
    shortestPathTo[e.to][e.from].cost = e.cost;

    adjacencyList[e.to].push_back({e.from, e.cost});
    shortestPathTo[e.from][e.to].cost = e.cost;

#ifdef FULL_DEBUG
    assert(checkIntegrity());
#endif
  }

  /// calculate shortest paths for all hosts
  virtual void calculateShortestPaths() {
    Clocker sort("sort adjacency list");

    for (uint16_t i = 0; i < adjacencyList.size(); ++i) {
      std::sort(adjacencyList[i].begin(), adjacencyList[i].end());
    }
    sort.stop();

    Clocker path("shortest path");
    for (uint16_t k = 0; k < nodes.size(); ++k) {
      for (uint16_t i = 0; i < nodes.size(); ++i) {
        for (uint16_t j = 0; j < nodes.size(); ++j) {
          if (k == i || k == j) continue;
          if (shortestPathTo[j][i].cost > shortestPathTo[k][i].cost + shortestPathTo[j][k].cost) {
            shortestPathTo[j][i] = {shortestPathTo[k][i].cost + shortestPathTo[j][k].cost,
                                    shortestPathTo[k][i].nextHop};
          }
        }
      }
    }
    path.stop();
  }

private:
  bool checkIntegrity() {
    for (uint16_t i = 0; i < adjacencyList.size(); ++i) {
      std::sort(adjacencyList[i].begin(), adjacencyList[i].end());
    }

    for (uint16_t i = 0; i < adjacencyList.size(); ++i) {
      int currTo = -1;
      for (AdjacencyMatrixCell cell: adjacencyList[i]) {
        if (cell.to <= currTo)
          return false;
        currTo = cell.to;
      }
    }

    return true;
  }
};

template<class T = void>
class CompleteGraph : public Graph<T> {
public:
  using Graph<T>::adjacencyList;
  using Graph<T>::nodes;
  using Graph<T>::shortestPathTo;
  using Graph<T>::addVertex;
  using Graph<T>::addEdge;

  explicit CompleteGraph(size_t capacity) : Graph<T>(capacity) {
    for (uint16_t i = 0; i < capacity; ++i) {
      addVertex({i, true});
    }

    for (uint16_t i = 0; i < capacity; ++i) {
      for (uint16_t j = i + 1; j < capacity; ++j) {
        Graph<T>::addEdge({i, j, 1});
      }
    }
  }

  virtual void addEdge(typename Graph<T>::Edge e) {
    throw runtime_error("impossible to add an edge into a complete graph");
  }

  /// calculate shortest paths for all hosts
  virtual void calculateShortestPaths() {
    Clocker sort("sort adjacency list");
    for (uint16_t i = 0; i < adjacencyList.size(); ++i) {
      std::sort(adjacencyList[i].begin(), adjacencyList[i].end());
    }
    sort.stop();

    Clocker path("shortest path");
    for (uint16_t i = 0; i < nodes.size(); ++i) {
      for (uint16_t j = 0; j < nodes.size(); ++j) {
        if (i != j)
          shortestPathTo[j][i] = {1, j};
        else
          shortestPathTo[j][i] = {0, j};
      }
    }
    path.stop();
  }
};
