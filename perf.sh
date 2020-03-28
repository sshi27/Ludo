#!/usr/bin/env bash
mkdir cmake-build-relwithdebinfo
cd cmake-build-relwithdebinfo
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -G "CodeBlocks - Unix Makefiles" ..
make -j8

rm -f ./$1.prof
env CPUPROFILE=$1.prof ./$1 $2 $3 &&
google-pprof --web ./$1 ./$1.prof

