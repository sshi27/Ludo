//
//  dph.h
//  DPH
//
//  Created by Yijun Zhou on 1/29/16.
//  Copyright © 2016 Yijun Zhou. All rights reserved.
//
#pragma once

#include "inttypes.h"
#include <cstdlib>
#include <cstdio>
#include <ctime>

template<class V, uint8_t KL, uint8_t VL = 8 * sizeof(V)>
class DPH {
public:
  static_assert(KL <= 64);
  static const uint64_t KEY_MASK = uint64_t(-1) >> (64 - KL);
  typedef int Status;
  
  //Data structure to store sub-hashtable
  typedef struct {
    bool *taken;
    uint64_t *keys;
    V *values;
    
    uint64_t sizeThreshold;
    uint64_t size;
    uint64_t capacity;
    uint64_t k;
  } Bucket;
  
  //Data structure to store the hash table
  typedef struct {
    Bucket *subTables;
    uint64_t size;//s(M)
    uint64_t k;
  } HashTable;
  
  //If an operation(push_back, delete, check if condition ** is satisfied) succeeds,
  //it will return SUCCESS; otherwise FAIL.
  static const int SUCCESS = 1;
  static const int FAIL = 0;
  
  //If position contains nothing or element is deleted, position will be set
  //to DELETED, which is 0.
  static const uint64_t DELETED = uint64_t(-1);
  
  //If a lookUp operation finds the element it is looking for, it will return FOUND;
  //otherwise NOT_FOUND.
  static const int FOUND = 1;
  static const int NOT_FOUND = 0;
  
  //If a hash function is injective on the list, it will set the flag to NO_COLLISION;
  //otherwise COLLISION.
  static const int NO_COLLISION = 1;
  static const int COLLISION = 0;
  
  //PRIME is a parameter used in the hash function. support up to 2 million numbers
  static const uint64_t PRIME = 47055833459;
  
  uint64_t count = 0;     // A variable records the number of operations(push_back, Delete).
  uint64_t c = 2;         // A parameter used to resize M
  uint64_t SM = 1;        // SM is a parameter to generate a s(M) function.
  uint64_t M = 0;         // The main hash table is to accommodate up to M elements.
  HashTable table;        // The main hash table
  
  //s(M)function is chosen to be O(n), and this makes the total space used is linear in the
  //number of elements currently stored in the table. s(M) returns the new number of sub-hashtables.
  inline uint64_t s(uint64_t size) {
    return (uint64_t) (SM * size);
  }
  
  //Hash functions generator ( According to the paper DPHULB)
  static inline uint64_t h(uint64_t key, uint64_t k, uint64_t p, uint64_t s) {
    uint64_t result = (((unsigned __int128) k * (KEY_MASK & key)) % p) % s;
    return result;
//    return (Hasher64<uint64_t>(k)(key) * (unsigned __int128) s) >> 64;
  }
  
  //push_back the key into the main hash table, Hash.
  Status insert(uint64_t key, V value) {
    count = count + 1;
    
    if (count > M) {
      RehashAll(key, value);
    } else {
      uint64_t bucket_number = h(key, table.k, PRIME, table.size);
      Bucket &subtable = table.subTables[bucket_number];
      
      uint64_t location = h(key, subtable.k, PRIME, subtable.capacity);
      if (subtable.keys[location] == key || !subtable.taken[location]) {
        subtable.taken[location] = true;
        subtable.keys[location] = key;
        subtable.values[location] = value;
      } else {  //A collision occurs so that the sub-hashtable has to be rehashed
        vector<pair<uint64_t, V>> tmp;
        for (uint64_t i = 0; i < subtable.capacity; i++) {
          if (subtable.taken[i]) {
            tmp.push_back(make_pair(subtable.keys[i], subtable.values[i]));
          }
        }
        //Append key to temp_list
        tmp.push_back(make_pair(key, value));
        
        if (subtable.size < subtable.sizeThreshold) { //size of sub-hashtable sufficient
          //Randomly choose hash function, until it is injective on sub-hashtable, that is no collision
          rebuildSubTable(subtable, tmp);
        } else { //Sub-hashtable is too small
          //Double capacity of the sub-hashtable
          if (subtable.sizeThreshold > 1) {
            subtable.sizeThreshold = 2 * subtable.sizeThreshold;
          } else {
            subtable.sizeThreshold = 2;
          }
          subtable.capacity = 2 * subtable.sizeThreshold * (subtable.sizeThreshold - 1);
          
          //If condition ** is still satisfied, rehash the sub-hashtable
          if (isSatisfied()) {
            subtable.taken = (bool *) (realloc(subtable.taken, subtable.capacity * sizeof(bool)));
            subtable.keys = (uint64_t *) (realloc(subtable.keys,
                                                  subtable.capacity * sizeof(uint64_t)));
            subtable.values = (V *) (realloc(subtable.values, subtable.capacity * sizeof(V)));
            
            rebuildSubTable(subtable, tmp);
          } else {
            //Level-1 hash function is "bad"
            RehashAll(key, value);
          }
        }
      }
      
      subtable.size = subtable.size + 1;
    }
    return SUCCESS;
  }
  
  void rebuildSubTable(Bucket &subtable, const vector<pair<uint64_t, V>> &tmp) {
    while (true) {
      bool collision = false;
      for (uint64_t i = 0; i < subtable.capacity; i++) {
        subtable.taken[i] = false;
      }
      subtable.k = (((uint64_t) rand() << 32) + rand()) % (PRIME - 1) + 1;
      
      for (pair<uint64_t, V> kv: tmp) {
        uint64_t new_location = h(kv.first, subtable.k, PRIME, subtable.capacity);
        if (subtable.taken[new_location]) {
          collision = true;
          break;
        } else {
          subtable.taken[new_location] = true;
          subtable.keys[new_location] = kv.first;
          subtable.values[new_location] = kv.second;
        }
      }
      
      if (!collision) break;
    }
  }
  
  //Rehashall is called by push_back, Delete or Initialize. It rehashes the whole hash table.
  Status RehashAll(uint64_t key, V value) {
    uint64_t k;
    
    vector<pair<uint64_t, V>> tableContentWhole;
    
    //Append all the elements in the hash table to all_list
    for (uint64_t i = 0; i < table.size; i++) {
      Bucket &subtable = table.subTables[i];
      for (uint64_t j = 0; j < subtable.capacity; j++) {
        if (subtable.taken[j]) {
          tableContentWhole.push_back(make_pair(subtable.keys[j], subtable.values[j]));
        }
      }
      free(subtable.keys);
      free(subtable.values);
    }
    
    free(table.subTables);
    table.subTables = NULL;
    
    if (key != DELETED) {
      tableContentWhole.push_back(make_pair(key, value));
    }
    
    count = tableContentWhole.size();
    //Resize the whole hash table
    if (count > 4) {
      M = (1 + c) * count;
    } else {
      M = (1 + c) * 4;
    }
    
    vector<vector<pair<uint64_t, V>>> subTableContents(s(M));
    
    //Randomly choose level-1 hash function until the condition ** is satisfied.
    while (true) {
      table.k = (((uint64_t) rand() << 32) + rand()) % (PRIME - 1) + 1;
      table.size = s(M);
      
      //Form every list for sub-table
      for (uint64_t i = 0; i < tableContentWhole.size(); i++) {
        pair<uint64_t, V> kv = tableContentWhole[i];
        uint64_t list_num = h(kv.first, table.k, PRIME, table.size);
        subTableContents[list_num].push_back(kv);
      }
      
      table.subTables = (Bucket *) realloc(table.subTables, table.size * sizeof(Bucket));
      memset(table.subTables, 0, table.size * sizeof(Bucket));
      
      for (uint64_t i = 0; i < table.size; i++) {
        Bucket &subtable = table.subTables[i];
        
        subtable.size = subTableContents[i].size();
        if (subtable.size > 1) {
          subtable.sizeThreshold = 2 * subtable.size;
          subtable.capacity = 2 * subtable.sizeThreshold * (subtable.sizeThreshold - 1);
        } else {
          subtable.sizeThreshold = 2;
          subtable.capacity = 4;
        }
      }
      
      if (isSatisfied()) {
        break;
      }
      
      for (uint64_t i = 0; i < table.size; i++) {
        subTableContents[i].clear();
      }
    }
    
    //Form every sub-hashtable
    for (uint64_t i = 0; i < table.size; i++) {
      Bucket &subtable = table.subTables[i];
      
      subtable.taken = (bool *) malloc(subtable.capacity * sizeof(bool));
      memset(subtable.taken, 0, subtable.capacity * sizeof(bool));
      subtable.keys = (uint64_t *) malloc(subtable.capacity * sizeof(uint64_t));
      subtable.values = (V *) malloc(subtable.capacity * sizeof(V));
      
      //Randomly choose hash function, until  it is injective on sub-hashtable, that is no collision
      rebuildSubTable(subtable, subTableContents[i]);
    }
    return SUCCESS;
  }
  
  //Delete key from the hash table
  Status remove(uint64_t key) {
    count = count + 1;
    uint64_t bucket_number = h(key, table.k, PRIME, table.size);
    Bucket &subtable = table.subTables[bucket_number];
    
    uint64_t location = h(key, subtable.k, PRIME, subtable.capacity);
    if (subtable.taken[location] && key == subtable.keys[location]) {
      subtable.taken[location] = false;
    } else {
      return FAIL;
    }
    
    if (count > M) {
      //Start a new phase
      RehashAll(0);
    }
    
    return SUCCESS;
  }
  
  //lookUp key in hash table
  inline Status lookUp(uint64_t key, V &out) const {
    uint64_t bucket_number = h(key, table.k, PRIME, table.size);
    Bucket &subtable = table.subTables[bucket_number];
    uint64_t location = h(key, subtable.k, PRIME, subtable.capacity);
    
    if (subtable.taken[location] && key == subtable.keys[location]) {
      out = subtable.values[location];
      return FOUND;
    }
    return NOT_FOUND;
  }
  
  //Initialize the hash table
  Status Initialize() {
    table.k = (((uint64_t) rand() << 32) + rand()) % (PRIME - 1) + 1;
    table.size = 0;
    table.subTables = NULL;
    RehashAll(DELETED, V());
    return SUCCESS;
  }
  
  //Check if condition ** is satisfied
  Status isSatisfied() const {
    uint64_t sum = 0;
    uint64_t threshold = 32 * M * M / table.size + 4 * M;
    
    for (uint64_t i = 0; i < table.size; i++) {
      sum += table.subTables[i].capacity;
      if (sum > threshold) {
        //If sum already exceeds threshold, terminate the loop.
        return FAIL;
      }
    }
    
    return SUCCESS;
  }
  
  virtual ~DPH() {
    //Append all the elements in the hash table to all_list
    for (uint64_t i = 0; i < table.size; i++) {
      Bucket &subtable = table.subTables[i];
      free(subtable.keys);
      free(subtable.values);
    }
    
    free(table.subTables);
    table.subTables = NULL;
  }
  
  DPH() {
    Initialize();
  }
  
  DPH Copy() const {
    DPH another(*this);
    
    another.table.subTables = (Bucket *) realloc(table.subTables, table.size * sizeof(Bucket));
    for (uint i = 0; i < another.table.size; ++i) {
      another.table.subTables[i] = table.subTables[i];
      
      Bucket subtable = another.table.subTables[i];
      subtable.taken = (bool *) malloc(subtable.capacity * sizeof(bool));
      subtable.keys = (uint64_t *) malloc(subtable.capacity * sizeof(uint64_t));
      subtable.values = (V *) malloc(subtable.capacity * sizeof(V));
      
      Bucket &mySubtable = table.subTables[i];
      for (uint ii = 0; ii < subtable.capacity; ++ii) {
        subtable.taken[ii] = mySubtable.taken[ii];
        if (subtable.taken[ii]) {
          subtable.keys[ii] = mySubtable.keys[ii];
          subtable.values[ii] = mySubtable.values[ii];
        }
      }
    }
    
    return another;
  }
  
  inline uint64_t memInBytes() const {
    uint64_t sum = 0;
    
    for (uint64_t i = 0; i < table.size; i++) {
      sum += table.subTables[i].capacity * (KL + VL);
    }
    
    return sum / 8 + sizeof(Bucket) * table.size + sizeof(table);
  }
};
