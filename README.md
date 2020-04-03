# Ludo hashing

## Overview

**Ludo hashing** is a new key-value lookup design which costs the least space (3.76 + 1.05l bits per key-value item for l-bit values) among known compact lookup solutions including the recently proposed partial-key Cuckoo and Bloomier perfect hashing. In addition to its space efficiency, Ludo Hashing works well with most practical systems by supporting fast lookups, fast updates, and concurrent writing/reading. We implement Ludo Hashing and evaluate it with both micro-benchmark and two network systems deployed in CloudLab. The results show that in practice Ludo Hashing saves 40% to 80%+ memory cost compared to existing dynamic solutions. It costs only a few GB memory for 1 billion key-value items and achieves high lookup throughput: over 65 million queries per second on a single node with multiple threads.

Our paper will appear in ACM SIGMETRICS 2020.  

**Paper Link:** https://users.soe.ucsc.edu/~qian/papers/LudoHashing.pdf

## Repository Structure

- MinimalPerfectCuckoo/     Ludo hashing
- BloomFilter/              Bloom filter
- CuckooPresized/           Cuckoo hashing
- Othello/                  Othello hashing
- SetSep/                   SetSep


## Run demo

```sh
# 0. setup 
sudo apt-get install google-perftools libgoogle-perftools-dev cmake build-essential pkgconf

# 1. build
mkdir release
cd release
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -G "CodeBlocks - Unix Makefiles" ..
make microbenchmarks -j4

# 2. run tests
./microbenchmarks
```

## API

Here is a small example of VF's APIs.

```c++
// Ludo hashing maintenance structure, where keys are strings, 
// and values are 29-bit integers. 
ControlPlaneMinimalPerfectCuckoo<std::string, uint32_t, 29> cp(nn);  

// Insert a key-value item into Ludo
cp.insert(k, v);

// Get the maintenance structure prepared (compute a proper seed for each bucket)
cp.prepareToExport();
// Export from the maintenance structure to one lookup structure
DataPlaneMinimalPerfectCuckoo<std::string, uint32_t, 29> dp(cp);

// look $k$ up in the lookup structure
uint32_t val = dp.lookUp(k);
```

For more details, check out `microbenchmarks.cpp`. It contains methods on parallel lookup and dynamic updates.

## Authors

- Shouqian Shi(sshi27@ucsc.edu)
- Chen Qian(cqian12@ucsc.edu)
