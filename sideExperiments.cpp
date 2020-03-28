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

template<class K>
struct OthelloChange {
  int8_t type;
  vector<uint32_t> cc;
  uint64_t xorTemplate;
  int marks[2];
};

template<int VL, class Val>
void seedLength() {
  typedef uint32_t Key;
  
  for (int repeat = 0; repeat < 10; ++repeat)
    for (uint64_t nn = 1048576; nn <= 16 * 1048576; nn *= 2)
      try {
        LFSRGen<Key> keyGen(0x1234567801234567ULL, max((uint64_t) 1E8, nn), 0);
        LFSRGen<Val> valueGen(0x1234567887654321ULL, nn, 0);
        
        vector<Key> keys(nn);
        vector<Val> values(nn);
        
        for (uint64_t i = 0; i < nn; i++) {
          keyGen.gen(&keys[i]);
          Val v;
          valueGen.gen(&v);
          values[i] = v & (uint(-1) >> (32 - VL));
        }
        
        ControlPlaneMinimalPerfectCuckoo<Key, Val, VL> cp(nn);
        for (int i = 0; i < nn * 2/ 3; ++i) {
          vector<MPC_PathEntry> path;
          cp.insert(keys[i], values[i], &path);
        }

        cp.prepareToExport();

        vector<uint> seedLengths;  // 需要多长的seed, 才不会overflow
        for (ControlPlaneMinimalPerfectCuckoo<uint32_t, uint8_t, 4, 0>::Bucket b: cp.buckets_) {
          seedLengths.push_back((uint8_t) ceil(log2(b.seed + 2)));
        }

        std::sort(seedLengths.begin(), seedLengths.end());

        unsigned long e = seedLengths.size() - 1;
        cout << nn << ": ";
        for (int i = 0; i <= 1000; ++i) {
          cout << seedLengths[e * i / 1000] << " ";
        }
        cout << endl;

        vector<uint> pathLengths;
        vector<uint> perEntryCCLengths;
        vector<uint> singleCCLengths;
        Clocker gen("MPC generate 1E6 updates");
        // prepare many updates. modification : insertion : deletion = 1:1:1
        for (int i = 0; i < 1E5; ++i) {
          if ((i & 1) == 0) {  // delete
            Key k;
            while (true) {
              k = keys[rand() % keys.size()];
              Val tmp;
              if (cp.lookUp(k, tmp)) {
                break;
              }
            }
            cp.remove(k);
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
            pathLengths.push_back(path.size());

            uint32_t s = 0;
            for (auto e: path) {
              s += e.locatorCC.size() * 4;
              singleCCLengths.push_back(e.locatorCC.size());
            }

            perEntryCCLengths.push_back(s);
          }
        }

        {
          cout << "Path length samples (1001 points): " << endl;
          std::sort(pathLengths.begin(), pathLengths.end());
          unsigned long e = pathLengths.size() - 1;
          for (int i = 0; i <= 1000; ++i) {
            cout << pathLengths[e * i / 1000] << " ";
          }
          cout << endl;
        }

        {
          cout << "CC size samples (1001 points): " << endl;
          std::sort(singleCCLengths.begin(), singleCCLengths.end());
          unsigned long e = singleCCLengths.size() - 1;
          for (int i = 0; i <= 1000; ++i) {
            cout << singleCCLengths[e * i / 1000] << " ";
          }
          cout << endl;
        }

        {
          cout << "Entry CC size samples (1001 points): " << endl;
          std::sort(perEntryCCLengths.begin(), perEntryCCLengths.end());
          unsigned long e = perEntryCCLengths.size() - 1;
          for (int i = 0; i <= 1000; ++i) {
            cout << perEntryCCLengths[e * i / 1000] << " ";
          }
          cout << endl;
        }
        gen.stop();
      } catch (exception &e) {
        cerr << e.what() << endl;
      }
}

template<int VL, class Val>
void pathLength() {
  typedef uint32_t Key;
  
  for (int repeat = 0; repeat < 10; ++repeat)
    for (uint64_t nn = 1048576; nn <= 16 * 1048576; nn *= 2)
      try {
        LFSRGen<Key> keyGen(0x1234567801234567ULL, max((uint64_t) 1E8, nn), 0);
        LFSRGen<Val> valueGen(0x1234567887654321ULL, nn, 0);
        
        vector<Key> keys(nn);
        vector<Val> values(nn);
        
        for (uint64_t i = 0; i < nn; i++) {
          keyGen.gen(&keys[i]);
          Val v;
          valueGen.gen(&v);
          values[i] = v & (uint(-1) >> (32 - VL));
        }
        
        vector<uint> pathLengths;
        
        ControlPlaneCuckooMap<Key, Val, uint8_t, false> cp(nn);
        
        for (int i = 0; i < nn; ++i) {
          vector<CuckooMove> path;
          cp.template insert<true>(keys[i], values[i], &path);
          pathLengths.push_back(path.size());
        }
        
        {
          cout << nn << ": ";
          std::sort(pathLengths.begin(), pathLengths.end());
          unsigned long e = pathLengths.size() - 1;
          for (int i = 0; i <= 1000; ++i) {
            cout << pathLengths[e * i / 1000] << " ";
          }
          cout << endl;
        }

      } catch (exception &e) {
        cerr << e.what() << endl;
      }
}

int main() {
  commonInit();
  
//  seedLength<4, uint8_t>();
  pathLength<4, uint8_t>();
  
  return 0;
}
