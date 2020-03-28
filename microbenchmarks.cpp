#include "common.h"
#include "cstdlib"
#include "SetSep/setsep.h"
#include "BloomFilter/bloom_flitable.h"
#include "CuckooPresized/cuckoo_map.h"
#include "CuckooPresized/cuckoo_ht.h"
#include "CuckooPresized/cuckoo_filter_control_plane.h"
#include "CuckooPresized/cuckoo_filtable.h"
#include "Othello/data_plane_othello.h"
#include "MinimalPerfectCuckoo/minimal_perfect_cuckoo.h"
#include "DPH/dph.h"

int version = 12;
int cores = min(20U, std::thread::hardware_concurrency());

template<class K>
struct OthelloChange {
  int8_t type;
  vector<uint32_t> cc;
  uint64_t xorTemplate;
  int marks[2];
};

typedef uint32_t Key;
const uint32_t lookupCnt = 1 << 25;

template<int VL, class Val>
void testSetSep(vector<Key> &keys, vector<Val> &values, uint64_t nn, vector<Key> &zipfianKeys) {
  try {
    Clocker construction("SetSep construction");
    SetSep<Key, Val, VL> cp(nn, true, keys, values);
    Counter::count("overflow", cp.overflow.size());
    cout << "overflow: " << cp.overflow.size() << endl;
    construction.stop();
    
    if (VL == 4 || VL == 20) {
      for (int threadCnt = 1; threadCnt <= cores; ++threadCnt) {
        thread threads[threadCnt];
        uint32_t start[threadCnt];

        for (int i = 0; i < threadCnt; ++i) {
          start[i] = i / threadCnt * zipfianKeys.size();
        }

        for (Distribution distribution: {uniform, exponential}) {
          ostringstream oss;
          oss << "SetSep parallel lookup " << threadCnt << " threads " << lookupCnt << " keys "
              << (distribution == exponential ? "Zipfian" : "uniform");
          Clocker plookup(oss.str());

          for (int i = 0; i < threadCnt; ++i) {
            threads[i] = std::thread(
              [](const SetSep<Key, Val, VL> *dp, uint32_t start, const vector<Key> *zipfianKeys,
                 uint32_t lookupCnt) {
                int stupid = 0;

                int ii = start;
                do {
                  const Key &k = zipfianKeys->at(ii);
                  Val val;
                  dp->lookUp(k, val);
                  stupid += val;

                  if (ii == lookupCnt - 1) ii = -1;
                  ++ii;
                } while (ii != start);
                printf("%d\b", stupid & 7);
              }, &cp, start[i], distribution == exponential ? &zipfianKeys : &keys, lookupCnt);
          }

          for (int i = 0; i < threadCnt; ++i) {
            threads[i].join();
          }
        }
      }
    }

//    if (VL == 20 && nn <= 1048576) {
      uint32_t updateCnt = 5000;
      Clocker c("SetSep apply " + to_string(updateCnt) + " updates");
      uint32_t halfSize = (keys.size() / 2);
      for (int updates = 0; updates < updateCnt; ++updates) {
        if (updates % 3 == 0) { // delete
          cp.remove(keys[rand() % halfSize]);
        } else if (updates % 3 == 1) { // modify
          cp.updateMapping(keys[rand() % halfSize], rand());
        } else {// insert
          cp.insert(rand(), rand());
        }
      }
//    }
  } catch (exception &e) {
    cout << e.what() << endl;
  }
}

template<int VL, class Val>
void testOthello(vector<Key> &keys, vector<Val> &values, uint64_t nn, vector<Key> &zipfianKeys) {
  Clocker construction("Othello CP construction");
  ControlPlaneOthello<Key, Val, VL> cp(nn, true, keys, values);
  construction.stop();
  
  Clocker exp("Othello export");
  DataPlaneOthello<Key, Val, VL> dp(cp);
  exp.stop();
  
  if (VL == 4 || VL == 20) {
    for (int threadCnt = 1; threadCnt <= cores; ++threadCnt) {
      thread threads[threadCnt];
      uint32_t start[threadCnt];
      
      for (int i = 0; i < threadCnt; ++i) {
        start[i] = i / threadCnt * zipfianKeys.size();
      }
      
      for (Distribution distribution: {uniform, exponential}) {
        ostringstream oss;
        oss << "Othello parallel lookup " << threadCnt << " threads " << lookupCnt
            << " keys " << (distribution == exponential ? "Zipfian" : "uniform");
        Clocker plookup(oss.str());
        
        for (int i = 0; i < threadCnt; ++i) {
          threads[i] = std::thread(
            [](const DataPlaneOthello<Key, Val, VL> *dp, uint32_t start, const vector<Key> *zipfianKeys,
               uint32_t lookupCnt) {
              int stupid = 0;
              
              int ii = start;
              do {
                const Key &k = zipfianKeys->at(ii);
                Val val;
                dp->lookUp(k, val);
                stupid += val;
                
                if (ii == lookupCnt - 1) ii = -1;
                ++ii;
              } while (ii != start);
              printf("%d\b", stupid & 7);
            }, &dp, start[i], distribution == exponential ? &zipfianKeys : &keys, lookupCnt);
        }
        
        for (int i = 0; i < threadCnt; ++i) {
          threads[i].join();
        }
      }
    }
  }
  
  if (VL == 20) {
    for (int i = nn * 2 / 3; i < nn; ++i) {
      cp.remove(keys[i]);
    }
    
    DataPlaneOthello<Key, Val, VL> dp(cp);
    
    vector<OthelloChange<Key>> oChanges;
    
    Clocker gen("Othello generate " + to_string(lookupCnt) + " updates");
// prepare many updates. modification : insertion : deletion = 1:1:1
    for (int i = 0; i < lookupCnt; ++i) {
      if (i % 3 == 0) {  // delete
        Key k;
        while (true) {
          k = keys[rand() % keys.size()];
          if (cp.isMember(k)) break;
        }
        cp.remove(k);
        
        uint32_t ha, hb;
        cp.getIndices(k, ha, hb);
        
        OthelloChange<Key> oc;
        oc.type = 'D';
        oc.marks[0] = cp.isEmpty(ha) ? ha : -1;
        oc.marks[1] = cp.isEmpty(hb) ? hb : -1;
        oChanges.push_back(oc);
        Counter::count_("Othello deletion update message length",
                        1 + (oc.marks[0] == -1 ? 0 : 4) + (oc.marks[1] == -1 ? 0 : 4));
        Counter::count_("Othello deletion update message");
      } else if (i % 3 == 1) { // modify
        Key k;
        while (true) {
          k = keys[rand() % keys.size()];
          if (cp.isMember(k)) break;
        }
        Val v = rand();
        
        OthelloChange<Key> oc;
        oc.type = 'M';
        int64_t tmp = cp.updateMapping(k, v);
        uint32_t ha, hb;
        cp.getIndices(k, ha, hb);
        oc.xorTemplate = uint64_t(tmp < 0 ? -tmp : tmp);
        oc.cc = cp.getHalfTree(k, tmp > 0, false);
        oChanges.push_back(oc);
        
        Counter::count_("Othello modification update message length",
                        1 + oc.cc.size() * 4 + (VL + 7) / 8);
        Counter::count_("Othello modification update message");
      } else { // insert
        Key k;
        while (true) {
          k = keys[rand() % keys.size()];
          if (!cp.isMember(k)) break;
        }
        Val v = rand();
        
        OthelloChange<Key> oc;
        oc.type = 'A';
        uint32_t ha, hb;
        cp.getIndices(k, ha, hb);
        
        oc.marks[0] = cp.isEmpty(ha) ? ha : -1;
        oc.marks[1] = cp.isEmpty(hb) ? hb : -1;
        int64_t tmp = cp.insert(k, v);
        
        oc.xorTemplate = uint64_t(tmp < 0 ? -tmp : tmp);
        oc.cc = cp.getHalfTree(k, tmp > 0, false);
        oChanges.push_back(oc);
        
        Counter::count_("Othello addition update message length",
                        1 + oc.cc.size() * 4 + (VL + 7) / 8 +
                        (oc.marks[0] == -1 ? 0 : 4) + (oc.marks[1] == -1 ? 0 : 4));
        Counter::count_("Othello addition update message");
      }
    }
    gen.stop();
    
    {
      DataPlaneOthello<Key, Val, VL> dp(cp);
      
      Clocker c("Othello apply " + to_string(lookupCnt) + " updates");
      
      for (OthelloChange<Key> oc: oChanges)
        if (oc.type == 'D') {
          if (oc.marks[0] >= 0) {
            dp.setEmpty(oc.marks[0]);
          }
          
          if (oc.marks[1] >= 0) {
            dp.setEmpty(oc.marks[1]);
          }
        } else if (oc.type == 'M') {
          dp.fixHalfTreeByConnectedComponent(oc.cc, oc.xorTemplate);
        } else {
          dp.fixHalfTreeByConnectedComponent(oc.cc, oc.xorTemplate);
          
          if (oc.marks[0] >= 0) {
            dp.setTaken(oc.marks[0]);
          }
          
          if (oc.marks[1] >= 0) {
            dp.setTaken(oc.marks[1]);
          }
        }
    }
    
    for (Distribution distribution: {uniform, exponential}) {
      for (int updateps = 25; updateps <= 1600; updateps *= 4) {
        for (int threadCnt = 1; threadCnt <= cores; ++threadCnt) {
          thread threads[threadCnt];
          uint32_t start[threadCnt];
          bool running[threadCnt];
          
          for (int i = 0; i < threadCnt; ++i) {
            start[i] = i / threadCnt * zipfianKeys.size();
            running[i] = true;
          }
          
          ostringstream oss;
          oss << "Othello parallel lookup " << threadCnt << " threads " << lookupCnt
              << " keys " << (distribution == exponential ? "Zipfian" : "uniform") << ", under " << updateps
              << " updates per second";
          Clocker plookup(oss.str());
          
          timeval startTime;
          gettimeofday(&startTime, nullptr);
          uint updates = 0;
          
          for (int i = 0; i < threadCnt; ++i) {
            threads[i] = std::thread([](const DataPlaneOthello<Key, Val, VL> *dp,
                                        uint32_t start, const vector<Key> *zipfianKeys, uint32_t lookupCnt,
                                        bool *running) {
              int stupid = 0;
              int ii = start;
              do {
                const Key &k = zipfianKeys->at(ii);
                Val val;
                dp->lookUp(k, val);
                stupid += val;
                
                if (ii == lookupCnt - 1) ii = -1;
                ++ii;
              } while (ii != start);
              *running = false;
              printf("%d\b", stupid & 7);
            }, &dp, start[i], &zipfianKeys, lookupCnt, &running[i]);
          }
          
          uint32_t failCnt = 0;
          while (true) {
            bool active = false;
            for (int i = 0; i < threadCnt; ++i) {
              active |= running[i];
            }
            if (!active || updates >= oChanges.size()) break;
            
            OthelloChange<Key> oc = oChanges[updates];
            if (oc.type == 'D') {
              if (oc.marks[0] >= 0) {
                dp.setEmpty(oc.marks[0]);
              }
              
              if (oc.marks[1] >= 0) {
                dp.setEmpty(oc.marks[1]);
              }
            } else if (oc.type == 'M') {
              dp.fixHalfTreeByConnectedComponent(oc.cc, oc.xorTemplate);
            } else {
              dp.fixHalfTreeByConnectedComponent(oc.cc, oc.xorTemplate);
              
              if (oc.marks[0] >= 0) {
                dp.setTaken(oc.marks[0]);
              }
              
              if (oc.marks[1] >= 0) {
                dp.setTaken(oc.marks[1]);
              }
            }
            
            updates++;
            timeval now;
            gettimeofday(&now, nullptr);
            
            uint64_t diff = diff_us(now, startTime);
            uint64_t shouldDiff = double(updates) / updateps * 1E6;
            if (shouldDiff > diff) {
              this_thread::sleep_for(std::chrono::microseconds(shouldDiff - diff));
            } else {
              if (++failCnt >= 100) {
                cout << ("cannot meet time requirement") << endl;
                break;
              }
            }
          }
          
          for (int i = 0; i < threadCnt; ++i) {
            threads[i].join();
          }
          
          if (failCnt >= 100) break;
        }
      }
    }
  }
}

template<int VL, class Val>
void testLudo(vector<Key> &keys, vector<Val> &values, uint64_t nn, vector<Key> &zipfianKeys) {
  Clocker construction("MPC construction");
  
  Clocker cpBuild("CP build");
  ControlPlaneMinimalPerfectCuckoo<Key, Val, VL> cp(nn);
  for (int i = 0; i < nn; ++i) {
    cp.insert(keys[i], values[i]);
  }
  cpBuild.stop();
  
  Clocker cpPrepare("CP prepare for DP");
  cp.prepareToExport();
  cpPrepare.stop();
  construction.stop();
  
  Clocker exp("MPC export");
  DataPlaneMinimalPerfectCuckoo<Key, Val, VL> dp(cp);
  exp.stop();
  
  if (VL == 4 || VL == 20) {
    for (int threadCnt = 1; threadCnt <= cores; ++threadCnt) {
      thread threads[threadCnt];
      uint32_t start[threadCnt];
      
      for (int i = 0; i < threadCnt; ++i) {
        start[i] = i / threadCnt * zipfianKeys.size();
      }
      
      for (Distribution distribution: {uniform, exponential}) {
        ostringstream oss;
        oss << "MPC parallel lookup " << threadCnt << " threads " << lookupCnt << " keys "
            << (distribution == exponential ? "Zipfian" : "uniform");
        Clocker plookup(oss.str());
        
        for (int i = 0; i < threadCnt; ++i) {
          threads[i] = std::thread([](const DataPlaneMinimalPerfectCuckoo<Key, Val, VL> *dp, uint32_t start,
                                      const vector<Key> *zipfianKeys, uint32_t lookupCnt) {
            int stupid = 0;
            
            int ii = start;
            do {
              const Key &k = zipfianKeys->at(ii);
              Val val;
              dp->lookUp(k, val);
              stupid += val;
              
              if (ii == lookupCnt - 1) ii = -1;
              ++ii;
            } while (ii != start);
            printf("%d\b", stupid & 7);
          }, &dp, start[i], distribution == exponential ? &zipfianKeys : &keys, lookupCnt);
        }
        
        for (int i = 0; i < threadCnt; ++i) {
          threads[i].join();
        }
      }
    }
  }
  
  if (VL == 20) {
    for (int i = nn * 2 / 3; i < nn; ++i) {
      cp.remove(keys[i]);
    }
    
    DataPlaneMinimalPerfectCuckoo<Key, Val, VL> dp(cp);
    
    vector<pair<vector<MPC_PathEntry>, Val>> insertPaths;
    vector<pair<uint32_t, Val>> modifications;
    
    Clocker gen("MPC generate " + to_string(lookupCnt) + " updates");
    // prepare many updates. modification : insertion : deletion = 1:1:1
    for (int i = 0; i < lookupCnt; ++i) {
      if (i % 3 == 0) {  // delete
        Key k;
        while (true) {
          k = keys[rand() % keys.size()];
          Val tmp;
          if (cp.lookUp(k, tmp)) {
            break;
          }
        }
        cp.remove(k);
        
        Counter::count_("MPC deletion update message length", 0);
        Counter::count_("MPC deletion update message");
      } else if (i % 3 == 1) { // modify
        Key k;
        while (true) {
          k = keys[rand() % keys.size()];
          
          Val tmp;
          if (cp.lookUp(k, tmp)) {
            break;
          }
        }
        Val v = rand();
        
        cp.updateMapping(k, v);
        pair<uint32_t, uint32_t> tmp = cp.locate(k);
        modifications.emplace_back((tmp.first << 2) + tmp.second, v);
        
        Counter::count_("MPC modification update message length", 1 + 4 + (VL + 7) / 8);
        Counter::count_("MPC modification update message");
      } else { // insert
        Key k;
        while (true) {
          k = keys[rand() % keys.size()];
          
          Val tmp;
          if (!cp.lookUp(k, tmp)) {
            break;
          }
        }
        Val v = rand();
        vector<MPC_PathEntry> path;
        cp.insert(k, v, &path);
        insertPaths.emplace_back(path, v);
        
        uint32_t s = 0;
        for (auto e: path) {
          s += e.locatorCC.size() * 4;
        }
        
        Counter::count_("MPC addition update message length", 1 + (VL + 7) / 8 + path.size() * (4 + 1 + 1) + s);
        Counter::count_("MPC addition update message");
      }
    }
    gen.stop();
    
    {
      DataPlaneMinimalPerfectCuckoo<Key, Val, VL> dp(cp);
      
      Clocker c("MPC apply " + to_string(lookupCnt) + " updates");
      
      for (int updates = 0; updates < lookupCnt; ++updates) {
        if (updates % 3 == 0) { // delete
          // empty
        } else if (updates % 3 == 1) { // modify
          pair<uint32_t, Val> tmp = modifications.at(updates / 3);
          dp.applyUpdate(tmp.first, tmp.second);
        } else {// insert
          pair<vector<MPC_PathEntry>, Val> tmp = insertPaths.at(updates / 3);
          dp.applyInsert(tmp.first, tmp.second);
        }
      }
    }
    
    for (Distribution distribution: {uniform, exponential})
      for (int threadCnt = 1; threadCnt <= cores; ++threadCnt) {
        for (int updateps = 25; updateps <= 1600; updateps *= 4) {
          thread threads[threadCnt];
          uint32_t start[threadCnt];
          bool running[threadCnt];
          
          for (int i = 0; i < threadCnt; ++i) {
            start[i] = i / threadCnt * zipfianKeys.size();
            running[i] = true;
          }
          
          ostringstream oss;
          oss << "MPC parallel lookup " << threadCnt << " threads " << lookupCnt
              << " keys " << (distribution == exponential ? "Zipfian" : "uniform") << ", under " << updateps
              << " updates per second";
          Clocker plookup(oss.str());
          
          timeval startTime;
          gettimeofday(&startTime, nullptr);
          uint updates = 0;
          
          for (int i = 0; i < threadCnt; ++i) {
            threads[i] = std::thread([](const DataPlaneMinimalPerfectCuckoo<Key, Val, VL> *dp,
                                        uint32_t start, const vector<Key> *zipfianKeys, uint32_t lookupCnt,
                                        bool *running) {
              int stupid = 0;
              int ii = start;
              do {
                const Key &k = zipfianKeys->at(ii);
                Val val;
                dp->lookUp(k, val);
                stupid += val;
                
                if (ii == lookupCnt - 1) ii = -1;
                ++ii;
              } while (ii != start);
              *running = false;
              printf("%d\b", stupid & 7);
            }, &dp, start[i], &zipfianKeys, lookupCnt, running + i);
          }
          
          uint32_t failCnt = 0;
          while (true) {
            bool active = false;
            for (int i = 0; i < threadCnt; ++i) {
              active |= running[i];
            }
            if (!active || modifications.empty() || insertPaths.empty()) break;
            
            if (updates % 3 == 0) { // delete
              // empty
            } else if (updates % 3 == 1) { // modify
              pair<uint32_t, Val> tmp = modifications.at(updates / 3);
              dp.applyUpdate(tmp.first, tmp.second);
            } else {// insert
              pair<vector<MPC_PathEntry>, Val> tmp = insertPaths.at(updates / 3);
              dp.applyInsert(tmp.first, tmp.second);
            }
            
            updates++;
            timeval now;
            gettimeofday(&now, nullptr);
            
            uint64_t diff = diff_us(now, startTime);
            uint64_t shouldDiff = double(updates) / updateps * 1E6;
            if (shouldDiff > diff) {
              this_thread::sleep_for(std::chrono::microseconds(shouldDiff - diff));
            } else {
              if (++failCnt >= 100) {
                cout << ("cannot meet time requirement") << endl;
                break;
              }
            }
          }
          
          for (int i = 0; i < threadCnt; ++i) {
            threads[i].join();
          }
          
          if (failCnt >= 100) break;
        }
      }
  }
}

template<int VL, class Val>
void testDPH(vector<Key> &keys, vector<Val> &values, uint64_t nn, vector<Key> &zipfianKeys) {
  Clocker construction("DPH construction");
  DPH<Val, sizeof(Key) * 8> cp;
  for (int i = 0; i < nn; ++i) {
    cp.insert(keys[i], values[i]);
  }
  Counter::count_("DPH memory", cp.memInBytes());
  construction.stop();
  
  if (VL == 4 || VL == 20) {
    for (int threadCnt = 1; threadCnt <= cores; ++threadCnt) {
      thread threads[threadCnt];
      uint32_t start[threadCnt];
      
      for (int i = 0; i < threadCnt; ++i) {
        start[i] = i / threadCnt * zipfianKeys.size();
      }
      
      for (Distribution distribution: {uniform, exponential}) {
        ostringstream oss;
        oss << "DPH parallel lookup " << threadCnt << " threads " << lookupCnt << " keys "
            << (distribution == exponential ? "Zipfian" : "uniform");
        Clocker plookup(oss.str());
        
        for (int i = 0; i < threadCnt; ++i) {
          threads[i] = std::thread([](const DPH<Val, sizeof(Key) * 8> *cp,
                                      uint32_t start, const vector<Key> *zipfianKeys, uint32_t lookupCnt) {
            int stupid = 0;
            int ii = start;
            do {
              const Key &k = zipfianKeys->at(ii);
              Val val;
              cp->lookUp(k, val);
              stupid += val;
              
              if (ii == lookupCnt - 1) ii = -1;
              ++ii;
            } while (ii != start);
            printf("%d\b", stupid & 7);
          }, &cp, start[i], distribution == exponential ? &zipfianKeys : &keys, lookupCnt);
        }
        
        for (int i = 0; i < threadCnt; ++i) {
          threads[i].join();
        }
      }
    }
  }
}

template<int VL, class Val>
void testBloom(vector<Key> &keys, vector<Val> &values, uint64_t nn, vector<Key> &zipfianKeys) {
  Clocker construction("Bloom construction");
  ControlPlaneBloomFiltable<Key, Val> cp(nn);
  for (int i = 0; i < nn; ++i) {
    cp.insert(keys[i], values[i]);
  }
  construction.stop();
  
  Clocker exp("Bloom export");
  DataPlaneBloomFiltable<Key, Val> dp(16, cp);
  exp.stop();
  
  if (VL == 4 || VL == 20) {
    for (int threadCnt = 1; threadCnt <= cores; ++threadCnt) {
      thread threads[threadCnt];
      uint32_t start[threadCnt];
      
      for (int i = 0; i < threadCnt; ++i) {
        start[i] = i / threadCnt * zipfianKeys.size();
      }
      
      for (Distribution distribution: {uniform, exponential}) {
        ostringstream oss;
        oss << "Bloom parallel lookup " << threadCnt << " threads " << lookupCnt
            << " keys " << (distribution == exponential ? "Zipfian" : "uniform");
        Clocker plookup(oss.str());
        
        for (int i = 0; i < threadCnt; ++i) {
          threads[i] = std::thread(
            [](const DataPlaneBloomFiltable<Key, Val> *dp, uint32_t start, const vector<Key> *zipfianKeys,
               uint32_t lookupCnt) {
              int stupid = 0;
              
              int ii = start;
              do {
                const Key &k = zipfianKeys->at(ii);
                Val val;
                dp->lookUp(k, val);
                stupid += val;
                
                if (ii == lookupCnt - 1) ii = -1;
                ++ii;
              } while (ii != start);
              printf("%d\b", stupid & 7);
            }, &dp, start[i], distribution == exponential ? &zipfianKeys : &keys, lookupCnt);
        }
        
        for (int i = 0; i < threadCnt; ++i) {
          threads[i].join();
        }
      }
    }
  }
}

template<int VL, class Val>
void testPKCuckoo(vector<Key> &keys, vector<Val> &values, uint64_t nn, vector<Key> &zipfianKeys) {
  Clocker construction("Partial key Cuckoo construction");
  Clocker cpBuild("CP build");
  ControlPlaneCuckooFiltable<Key, Val, uint8_t, 0> cp(nn);
  for (int i = 0; i < nn; ++i) {
    cp.insert(keys[i], values[i]);
  }
  cpBuild.stop();
  construction.stop();
  
  Clocker exp("Partial key Cuckoo export");
  DataPlaneCuckooFiltable<Key, Val, uint8_t, 0> dp(*cp.level1, *cp.level2);
  exp.stop();
  
  if (VL == 4 || VL == 20) {
    for (int threadCnt = 1; threadCnt <= cores; ++threadCnt) {
      thread threads[threadCnt];
      uint32_t start[threadCnt];
      
      for (int i = 0; i < threadCnt; ++i) {
        start[i] = i / threadCnt * zipfianKeys.size();
      }
      
      for (Distribution distribution: {uniform, exponential}) {
        ostringstream oss;
        oss << "Partial key Cuckoo parallel lookup " << threadCnt << " threads "
            << lookupCnt << " keys "
            << (distribution == exponential ? "Zipfian" : "uniform");
        Clocker plookup(oss.str());
        
        for (int i = 0; i < threadCnt; ++i) {
          threads[i] = std::thread([](const DataPlaneCuckooFiltable<Key, Val, uint8_t, 0> *dp, uint32_t start,
                                      const vector<Key> *zipfianKeys, uint32_t lookupCnt) {
            int stupid = 0;
            
            int ii = start;
            do {
              const Key &k = zipfianKeys->at(ii);
              Val val;
              dp->lookUp(k, val);
              stupid += val;
              
              if (ii == lookupCnt - 1) ii = -1;
              ++ii;
            } while (ii != start);
            printf("%d\b", stupid & 7);
          }, &dp, start[i], distribution == exponential ? &zipfianKeys : &keys, lookupCnt);
        }
        
        for (int i = 0; i < threadCnt; ++i) {
          threads[i].join();
        }
      }
    }
  }
}

template<int VL, class Val>
void testCuckoo(vector<Key> &keys, vector<Val> &values, uint64_t nn, vector<Key> &zipfianKeys) {
  Clocker construction("Cuckoo construction");
  Clocker cpBuild("CP build");
  ControlPlaneCuckooMap<Key, Val, uint8_t> cp(nn);
  for (int i = 0; i < nn; ++i) {
    cp.insert(keys[i], values[i]);
  }
  cpBuild.stop();
  construction.stop();
  
  if (VL == 4 || VL == 20) {
    for (int threadCnt = 1; threadCnt <= cores; ++threadCnt) {
      thread threads[threadCnt];
      uint32_t start[threadCnt];
      
      for (int i = 0; i < threadCnt; ++i) {
        start[i] = i / threadCnt * zipfianKeys.size();
      }
      
      for (Distribution distribution: {uniform, exponential}) {
        ostringstream oss;
        oss << "Cuckoo parallel lookup " << threadCnt << " threads "
            << lookupCnt << " keys "
            << (distribution == exponential ? "Zipfian" : "uniform");
        Clocker plookup(oss.str());
        
        for (int i = 0; i < threadCnt; ++i) {
          threads[i] = std::thread(
            [](const ControlPlaneCuckooMap<Key, Val, uint8_t> *dp, uint32_t start, const vector<Key> *zipfianKeys,
               uint32_t lookupCnt) {
              int stupid = 0;
              
              int ii = start;
              do {
                const Key &k = zipfianKeys->at(ii);
                Val val;
                dp->lookUp(k, val);
                stupid += val;
                
                if (ii == lookupCnt - 1) ii = -1;
                ++ii;
              } while (ii != start);
              printf("%d\b", stupid & 7);
            }, &cp, start[i], distribution == exponential ? &zipfianKeys : &keys, lookupCnt);
        }
        
        for (int i = 0; i < threadCnt; ++i) {
          threads[i].join();
        }
      }
    }
    
    if (VL == 20 && nn < 1048576) {
      Clocker c("Cuckoo apply " + to_string(lookupCnt) + " updates");
      uint32_t halfSize = (keys.size() / 2);
      for (int updates = 0; updates < lookupCnt; ++updates) {
        if (updates % 3 == 0) { // delete
          cp.remove(keys[rand() % halfSize]);
        } else if (updates % 3 == 1) { // modify
          cp.updateMapping(keys[rand() % halfSize], rand());
        } else {// insert
          cp.insert(rand(), rand());
        }
      }
    }
  }
}

template<int VL, class Val>
void test() {
  for (int repeat = 0; repeat < 10; ++repeat)
    for (uint64_t nn = 131072; nn <= (1U << 30); nn *= 2)
      try {
        ostringstream oss;
        
        oss << "value length " << VL << ", key set size " << nn << ", repeat#" << repeat
            << ", version#" << version;
        
        string logName = "../dist/logs/" + oss.str() + ".log";
        
        ifstream testLog(logName);
        string lastLine, tmp;
        
        while (getline(testLog, tmp)) {
          if (tmp.size() && tmp[0] == '|') lastLine = tmp;
        }
        testLog.close();
        
        if (lastLine.size() >= 3 && lastLine[2] == '-') continue;
        
        TeeOstream tos(logName);
        Clocker clocker(oss.str(), &tos);
        
        LFSRGen<Key> keyGen(0x1234567801234567ULL, max((uint64_t) lookupCnt, nn), 0);
        LFSRGen<Val> valueGen(0x1234567887654321ULL, nn, 0);
        
        vector<Key> keys(max((uint64_t) lookupCnt, nn));
        vector<Val> values(nn);
        
        for (uint64_t i = 0; i < nn; i++) {
          keyGen.gen(&keys[i]);
          Val v;
          valueGen.gen(&v);
          values[i] = v & (uint(-1) >> (32 - VL));
        }
        
        for (uint64_t i = nn; i < max((uint64_t) lookupCnt, nn); i++) {
          keys[i] = keys[i % nn];
        }
        
        vector<Key> zipfianKeys;
        zipfianKeys.reserve(lookupCnt);
        
        for (int i = 0; i < lookupCnt; ++i) {
          InputBase::distribution = exponential;
          InputBase::bound = nn;
          
          uint seed = Hasher32<string>()(logName);
          InputBase::setSeed(seed);
          
          uint32_t idx = InputBase::rand();
          zipfianKeys.push_back(keys[idx]);
        }
  
//        if (nn < 268435456)
          testSetSep<VL, Val>(keys, values, nn, zipfianKeys);
  
//        testCuckoo<VL, Val>(keys, values, nn, zipfianKeys);

//        testOthello<VL, Val>(keys, values, nn, zipfianKeys);
//
//        try {
//          testLudo<VL, Val>(keys, values, nn, zipfianKeys);
//        } catch (exception &e) {
//          cout << e.what() << endl;
//          break;
//        }

//        if (nn < 4194304) {
//          testDPH<VL, Val>(keys, values, nn, zipfianKeys);
//        }

//        if (VL == 4) {
//          testBloom<VL, Val>(keys, values, nn, zipfianKeys);
//        }

//        testPKCuckoo<VL, Val>(keys, values, nn, zipfianKeys);
        
        return;
      } catch (exception &e) {
        cerr << e.what() << endl;
      }
}

int main(int argc, char **argv) {
  commonInit();
  
  if (argc == 1) {
    for (int i = 0; i < 100; ++i) test<4, uint8_t>();
    for (int i = 0; i < 100; ++i) test<8, uint8_t>();
    for (int i = 0; i < 100; ++i) test<12, uint16_t>();
    for (int i = 0; i < 100; ++i) test<16, uint16_t>();
    for (int i = 0; i < 100; ++i) test<20, uint32_t>();
//    test<32>();
  }
  
  switch (atoi(argv[1])) {
    case 4:
      for (int i = 0; i < 100; ++i) test<4, uint8_t>();
      break;
    case 8:
      for (int i = 0; i < 100; ++i) test<8, uint8_t>();
      break;
    case 12:
      for (int i = 0; i < 100; ++i) test<12, uint16_t>();
      break;
    case 16:
      for (int i = 0; i < 100; ++i) test<16, uint16_t>();
      break;
    case 20:
      for (int i = 0; i < 100; ++i) test<20, uint32_t>();
      break;
//    case 16:
//      test<16>();
//    case 32:
//      test<32>();
  }
  
  return 0;
}
