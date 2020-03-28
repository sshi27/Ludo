#!/usr/bin/env bash
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -G "CodeBlocks - Unix Makefiles" ..
make metrics -j4

for run in {1..1000}
do
sudo ./metrics
done