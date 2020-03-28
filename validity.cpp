#include <input/input_types.h>
#include <gperftools/profiler.h>
#include <Othello/control_plane_othello.h>
#include "MinimalPerfectCuckoo/minimal_perfect_cuckoo.h"
#include "CuckooPresized/cuckoo_map.h"
#include "DPH/dph.h"
#include "SetSep/setsep.h"

uint64_t NK_MAX = 20000000;
int version = 0;
Hasher64<string> h(0x19900111L);

template<class K, class Sample>
void testSetSep(uint64_t Nk, Distribution distribution) {
  for (int repeatCnt = 0; repeatCnt < 1; ++repeatCnt) {
    ostringstream oss;
    
    const char *typeName = typeid(Sample) == typeid(Tuple5) ? "5-tuple" :
                           typeid(Sample) == typeid(IPv4) ? "IPv4" :
                           typeid(Sample) == typeid(IPv6) ? "IPv6" :
                           typeid(Sample) == typeid(MAC) ? "MAC" :
                           typeid(Sample) == typeid(ID) ? "ID" :
                           typeid(Sample) == typeid(URL) ? "URL"
                                                         : typeid(K).name();
    
    oss << Nk << " " << (distribution ? "uniform" : "Zipfian") << " "
        << typeName << " repeat#" << repeatCnt << " version#" << version << " SetSep";
    string logName = "../dist/logs/" + oss.str() + ".log";
    
    ifstream testLog(logName);
    string lastLine, tmp;
    
    while (getline(testLog, tmp)) {
      if (tmp.size() && tmp[0] == '|') lastLine = tmp;
    }
    testLog.close();
    
    if (lastLine.size() >= 3 && lastLine[2] == '-') continue;
    
    InputBase::distribution = distribution;
    InputBase::bound = Nk;
    
    uint seed = h(logName);
    InputBase::setSeed(seed);
    
    TeeOstream tos(logName);
    Clocker clocker(oss.str(), &tos);
    
    vector<K> keys(Nk);
    vector<uint16_t> values(Nk);
    uint64_t mask = (1 << 9) - 1;
    
    for (uint64_t i = 0; i < Nk; i++) {
      keys[i] = Sample::enumerate(i);
      values[i] = i & mask;
    }
    
    {
      Clocker add("SetSep build");
      SetSep<K, uint16_t, 9> c(Nk, true, keys, values);
      Counter::count("overflow", c.overflow.size());
      for (int i = 0; i < Nk; ++i) {
        K k = Sample::enumerate(i);
        uint16_t out;
        if (!(c.lookUp(k, out) && (out & mask) == (i & mask))) {
          c.lookUp(k, out);
          Counter::count("error 0");
        }
      }
      
      for (uint64_t i = 0; i < Nk; ++i) {
        K k = Sample::enumerate(i);
        uint16_t out1, out2;
        if (!(c.lookUp(k, out1) && c.lookUpViaIndex(k, out2) && (out1 & mask) == (i & mask) &&
              (out1 & mask) == (out2 & mask))) {
          c.lookUp(k, out1);
          c.lookUpViaIndex(k, out2);
          Counter::count("error 1");
        }
      }
      
      add.stop();
    }
    
    {
      SetSep<K, uint16_t, 9> c(Nk, true);
      
      uint16_t mask = (1 << 9) - 1;
      
      Clocker add("SetSep add");
      for (uint64_t i = 0; c.keyCnt < Nk; ++i) {
        const K k = Sample::enumerate(i);
        if (!c.insert(k, i & mask)) {
          c.insert(k, i & mask);
          Counter::count("error 2");
        }
        
        uint16_t out;
        if (!(c.lookUp(k, out) && (out & mask) == (i & mask))) {
          c.lookUp(k, out);
          Counter::count("error 3");
        }
      }
      
      for (uint64_t i = 0; i < Nk; ++i) {
        K k = Sample::enumerate(i);
        uint16_t out;
        if (!(c.lookUp(k, out) && (out & mask) == (i & mask))) {
          c.lookUp(k, out);
          Counter::count("error 4");
        }
      }
      
      for (uint64_t i = 0; i < Nk; ++i) {
        K k = Sample::enumerate(i);
        uint16_t out1, out2;
        if (!(c.lookUp(k, out1) && c.lookUpViaIndex(k, out2) && (out1 & mask) == (i & mask) &&
              (out1 & mask) == (out2 & mask))) {
          c.lookUp(k, out1);
          c.lookUpViaIndex(k, out2);
          Counter::count("error 5");
        }
      }
      
      Counter::count("overflow", c.overflow.size());
      
      for (uint64_t i = 0; i < Nk; i += 2) {
        const K k = Sample::enumerate(i);
        c.remove(k);
        if (c.isMember(k)) {
          c.remove(k);
          Counter::count("error 6a");
        }
      }
      
      if (c.keyCnt != Nk / 2) {
        Counter::count("error 6");
      }
      
      for (uint64_t i = 1; i < Nk; i += 2) {
        K k = Sample::enumerate(i);
        uint16_t out1, out2;
        if (!(c.lookUp(k, out1) && c.lookUpViaIndex(k, out2) && (out1 & mask) == (i & mask) &&
              (out1 & mask) == (out2 & mask))) {
          c.lookUp(k, out1);
          c.lookUpViaIndex(k, out2);
          Counter::count("error 7");
        }
      }
      
      for (uint64_t i = 0; i < Nk; i += 2) {
        const K k = Sample::enumerate(i);
        if (!c.insert(k, i & mask)) {
          c.insert(k, i & mask);
          Counter::count("error 8");
        }
        
        for (uint64_t ii = 0; ii < Nk; ii++) {
          if (ii > i && ii % 2 == 0) continue;
          
          K k = Sample::enumerate(ii);
          uint16_t out1, out2;
          if (!(c.lookUp(k, out1) && c.lookUpViaIndex(k, out2) && (out1 & mask) == (ii & mask) &&
                (out1 & mask) == (out2 & mask))) {
            c.lookUp(k, out1);
            c.lookUpViaIndex(k, out2);
            Counter::count("error 9");
          }
        }
      }
      
      for (uint64_t i = 0; i < Nk; i++) {
        K k = Sample::enumerate(i);
        uint16_t out1, out2;
        if (!(c.lookUp(k, out1) && c.lookUpViaIndex(k, out2) && (out1 & mask) == (i & mask) &&
              (out1 & mask) == (out2 & mask))) {
          c.lookUp(k, out1);
          c.lookUpViaIndex(k, out2);
          Counter::count("error 9");
        }
      }
      
      for (uint64_t i = 0; i < Nk; ++i) {
        const K k = Sample::enumerate(i);
        c.updateMapping(k, mask - i & mask);
        
        uint16_t out;
        if (!(c.lookUp(k, out) && (out & mask) == (mask - i & mask))) {
          c.lookUp(k, out);
          Counter::count("error 10");
        }
      }
      
      for (uint64_t i = 0; i < Nk; ++i) {
        const K k = Sample::enumerate(i);
        uint16_t out;
        if (!(c.lookUp(k, out) && (out & mask) == (mask - i & mask))) {
          c.lookUp(k, out);
          Counter::count("error 11");
        }
      }
      
      add.stop();
    }

//    {
//      ControlPlaneMinimalPerfectCuckoo<K, uint16_t, 9> c(Nk);
//
//      uint16_t mask = (1 << 9) - 1;
//
//      Clocker add("MPC add");
//      for (uint64_t i = 0; c.entryCount < Nk; ++i) {
//        const K k = Sample::enumerate(i);
//        if (!c.insert(k, i & mask)) {
//          c.insert(k, i & mask);
//          Counter::count("error 2");
//        }
//
//        uint16_t out;
//        if (!(c.lookUp(k, out) && (out & mask) == (i & mask))) {
//          c.lookUp(k, out);
//          Counter::count("error 3");
//        }
//      }
//
//      for (uint64_t i = 0; i < Nk; ++i) {
//        K k = Sample::enumerate(i);
//        uint16_t out;
//        if (!(c.lookUp(k, out) && (out & mask) == (i & mask))) {
//          c.lookUp(k, out);
//          Counter::count("error 4");
//        }
//      }
//
//      Clocker exp("MPC export");
//      c.prepareToExport();
//      DataPlaneMinimalPerfectCuckoo<K, uint16_t, 9> dp(c);
//      Counter::count("MPC overflow rate", (double) dp.overflow.entryCount / Nk);
//      exp.stop();
//
//      for (int ii = 0; ii < c.locator.ma + c.locator.mb; ++ii) {
//        assert(c.locator.memValueGet(ii) == dp.locator.memValueGet(ii));
//      }
//      for (int ii = 0; ii < c.buckets_.size(); ++ii) {
//        auto &b = c.buckets_[ii];
//        if (b.seed >= 31) {
//          uint8_t out = 0;
//          assert(b.seed == (dp.overflow.lookUp(ii, out), out));
//        }
//      }
//
//      for (int bid = 0; bid < c.buckets_.size(); ++bid) {
//        auto cpBucket = c.getDpBucket(bid);
//        auto dpBucket = dp.readBucket(bid);
//
//        for (int s = 0; s < 4; ++s) {
//          if (cpBucket.occupiedMask & (1 << s)) {
//            assert(cpBucket.values[s] == dpBucket.values[s]);
//          }
//        }
//      }
//
//      for (int i = 0; i < Nk; ++i) {
//        K k = Sample::enumerate(i);
//        uint16_t out;
//        if (!dp.lookUp(k, out) || (out & mask) != (i & mask)) {
//          uint16_t cpOut;
//          c.lookUp(k, cpOut);
//          dp.lookUp(k, out);
//          cout << "1 " << out << ", " << cpOut << endl;
//        }
//      }
//
//      unordered_set<int> existingKid;
//      for (int i = 0; i < Nk; ++i) existingKid.insert(i);
//
//      for (int i = 0; i < Nk / 10; ++i) {
//        uint32_t kid = rand() % Nk;
//        K k = Sample::enumerate(kid);
//        uint32_t bs = c.updateMapping(k, -kid & mask);
//        dp.applyUpdate(bs, -kid & mask);
//      }
//
//      for (int ii = 0; ii < c.locator.ma + c.locator.mb; ++ii) {
//        assert(c.locator.memValueGet(ii) == dp.locator.memValueGet(ii));
//      }
//
//      for (int ii = 0; ii < c.buckets_.size(); ++ii) {
//        auto &b = c.buckets_[ii];
//        if (b.seed >= 31) {
//          uint8_t out = 0;
//          assert(b.seed == (dp.overflow.lookUp(ii, out), out));
//        }
//      }
//
//      for (int i : existingKid) {
//        K k = Sample::enumerate(i);
//        uint16_t cpOut;
//        c.lookUp(k, cpOut);
//
//        uint16_t out;
//        if (!dp.lookUp(k, out) || (out & mask) != (cpOut & mask)) {
//          c.lookUp(k, cpOut);
//          dp.lookUp(k, out);
//          cout << "2 " << out << ", " << cpOut << endl;
//        }
//      }
//
//      for (int i = 0; i < Nk / 3; ++i) {
//        uint32_t kid = rand() % Nk;
//        K k = Sample::enumerate(kid);
//          c.remove(k);
//        existingKid.erase(kid);
//      }
//
//      for (int ii = 0; ii < c.locator.ma + c.locator.mb; ++ii) {
//        assert(c.locator.memValueGet(ii) == dp.locator.memValueGet(ii));
//      }
//
//      for (int ii = 0; ii < c.buckets_.size(); ++ii) {
//        auto &b = c.buckets_[ii];
//        if (b.seed >= 31) {
//          uint8_t out = 0;
//          assert(b.seed == (dp.overflow.lookUp(ii, out), out));
//        }
//      }
//
//      for (int i : existingKid) {
//        K k = Sample::enumerate(i);
//        uint16_t cpOut;
//        c.lookUp(k, cpOut);
//
//        uint16_t out;
//        if (!dp.lookUp(k, out) || (out & mask) != (cpOut & mask)) {
//          c.lookUp(k, cpOut);
//          dp.lookUp(k, out);
//          cout << "3 " << out << ", " << cpOut << endl;
//        }
//      }
//      for (int bid = 0; bid < c.buckets_.size(); ++bid) {
//        auto cpBucket = c.getDpBucket(bid);
//        auto dpBucket = dp.readBucket(bid);
//
//        for (int s = 0; s < 4; ++s) {
//          if (cpBucket.occupiedMask & (1 << s)) {
//            assert(cpBucket.values[s] == dpBucket.values[s]);
//          }
//        }
//      }
//
//      for (int i = 0; i < Nk / 5;) {
//        uint32_t kid = rand();
//        if (existingKid.find(kid) != existingKid.end()) continue;
//
//        try {
//          K k = Sample::enumerate(kid);
//          vector<MPC_PathEntry> path;
//          if (c.insert(k, -kid & mask, &path) == &k) {
//            dp.applyInsert(path, -kid & mask);
//            existingKid.insert(kid);
//
//            for (int ii = 0; ii < c.locator.ma + c.locator.mb; ++ii) {
//              assert(c.locator.memValueGet(ii) == dp.locator.memValueGet(ii));
//            }
//
//            for (int ii = 0; ii < c.buckets_.size(); ++ii) {
//              auto &b = c.buckets_[ii];
//              if (b.seed >= 31) {
//                uint8_t out = 0;
//                assert(b.seed == (dp.overflow.lookUp(ii, out), out));
//              }
//            }
//
//            for (int bid = 0; bid < c.buckets_.size(); ++bid) {
//              auto cpBucket = c.getDpBucket(bid);
//              auto dpBucket = dp.readBucket(bid);
//
//              for (int s = 0; s < 4; ++s) {
//                if (cpBucket.occupiedMask & (1 << s)) {
//                  assert(cpBucket.values[s] == dpBucket.values[s]);
//                }
//              }
//            }
//
//            for (int i : existingKid) {
//              K k = Sample::enumerate(i);
//              uint16_t cpOut;
//              c.lookUp(k, cpOut);
//
//              uint16_t out;
//              if (!dp.lookUp(k, out) || (out & mask) != (cpOut & mask)) {
//                c.lookUp(k, cpOut);
//                dp.lookUp(k, out);
//                cout << "4 " << out << ", " << cpOut << endl;
//              }
//            }
//            i++;
//          }
//        } catch (exception &e) {
//          cerr << e.what() << endl;
//        }
//      }
//
//      for (int ii = 0; ii < c.locator.ma + c.locator.mb; ++ii) {
//        assert(c.locator.memValueGet(ii) == dp.locator.memValueGet(ii));
//      }
//
//      for (int i : existingKid) {
//        K k = Sample::enumerate(i);
//        uint16_t cpOut;
//        c.lookUp(k, cpOut);
//
//        uint16_t out;
//        if (!dp.lookUp(k, out) || (out & mask) != (cpOut & mask)) {
//          c.lookUp(k, cpOut);
//          dp.lookUp(k, out);
//          cout << "5 " << out << ", " << cpOut << endl;
//        }
//      }
//    }
  }
}

int main(int argc, char **argv) {
  commonInit();
  NK_MAX = (argc >= 3) ? atol(argv[2]) : NK_MAX;
  
  for (uint64_t Nk = (argc >= 2) ? atol(argv[1]) : 1024ULL; Nk < NK_MAX; Nk <<= 1) {
    for (Distribution distribution:{exponential}) {
      testSetSep<string, ID>(Nk, distribution);
//      testSetSep<string, ID>(Nk, distribution);
    }
  }
}
