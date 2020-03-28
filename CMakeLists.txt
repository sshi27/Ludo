cmake_minimum_required(VERSION 3.5)
project(Metrics)

# sudo apt-mark hold *grub*
# sudo apt-get install google-perftools libgoogle-perftools-dev cmake build-essential gdb libssl-dev pkgconf tmux clang liblua5.3-dev
# cd ~/MinimalPerfectCuckoo ; mkdir release ; cd release ; cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -G "CodeBlocks - Unix Makefiles" .. ; make -j8

link_directories(/usr/local/lib)
#link_directories(cmph)
set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mavx -maes")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -mavx -maes")

IF ((CMAKE_BUILD_TYPE MATCHES Debug) OR (CMAKE_BUILD_TYPE MATCHES RelWithDebInfo))
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPROFILE ")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DPROFILE ")
    
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb ")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ggdb ")
ENDIF ((CMAKE_BUILD_TYPE MATCHES Debug) OR (CMAKE_BUILD_TYPE MATCHES RelWithDebInfo))

IF ((CMAKE_BUILD_TYPE MATCHES Debug))
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_DEBUG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFULL_DEBUG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-no-as-needed -ggdb -O0")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DFULL_DEBUG")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,-no-as-needed -ggdb -O0")
ENDIF ((CMAKE_BUILD_TYPE MATCHES Debug))

IF ((CMAKE_BUILD_TYPE MATCHES Release) OR (CMAKE_BUILD_TYPE MATCHES RelWithDebInfo))
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -DNDEBUG")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast -DNDEBUG")
ENDIF ((CMAKE_BUILD_TYPE MATCHES Release) OR (CMAKE_BUILD_TYPE MATCHES RelWithDebInfo))

#FIND_PACKAGE(Boost 1.65 COMPONENTS REQUIRED)

include_directories(
        #        ${Boost_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set(COMMON_SOURCE_FILES
        common.cpp
        farmhash/farmhash.cc
        utils/ClientSock.cpp
        utils/hashutil.cc
        input/input_types.cpp
        )

set(HEADER_FILES
        common.h
        hash.h
        control_plane.h
        farmhash/farmhash.h
        utils/hashutil.h
        DPH/dph.h
        lfsr64.h
        disjointset.h
        BloomFilter/bloom_filter.h
        BloomFilter/bloom_flitable.h
        BloomFilter/bloom_fliter_control_plane.h
        Othello/data_plane_othello.h
        Othello/control_plane_othello.h
        Othello/othello_filter_control_plane.h
        CuckooPresized/cuckoo_map.h
        CuckooPresized/cuckoo_ht.h
        CuckooPresized/cuckoo_filter_control_plane.h
        CuckooPresized/cuckoo_filtable.h
        MinimalPerfectCuckoo/minimal_perfect_cuckoo.h
        MinimalPerfectCuckoo/mpc_control_plane.h
        SetSep/setsep.h
        utils/ClientSock.h
        utils/json.hpp
        utils/debugbreak.h
        input/input_types.h
        )

add_executable(validity
        ${HEADER_FILES}
        ${COMMON_SOURCE_FILES}
        validity.cpp)

add_executable(microbenchmarks
        ${HEADER_FILES}
        ${COMMON_SOURCE_FILES}
        microbenchmarks.cpp)

add_executable(sideExperiments
        ${HEADER_FILES}
        ${COMMON_SOURCE_FILES}
        sideExperiments.cpp)

IF (APPLE)
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_WIN32_THREADS_INIT 0)
    set(CMAKE_USE_PTHREADS_INIT 1)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
ELSE ()
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(validity Threads::Threads)
    target_link_libraries(microbenchmarks Threads::Threads)
    target_link_libraries(sideExperiments Threads::Threads)
ENDIF ()


set(Gperftools_DIR "${CMAKE_CURRENT_LIST_DIR}/cmake/")
find_package(Gperftools REQUIRED)
target_link_libraries(validity ${GPERFTOOLS_PROFILER})
target_link_libraries(microbenchmarks ${GPERFTOOLS_PROFILER})
target_link_libraries(sideExperiments ${GPERFTOOLS_PROFILER})

find_package(PkgConfig REQUIRED)
pkg_search_module(OPENSSL REQUIRED openssl)
target_link_libraries(validity ${OPENSSL_LIBRARIES})
target_link_libraries(microbenchmarks ${OPENSSL_LIBRARIES})
target_link_libraries(sideExperiments ${OPENSSL_LIBRARIES})

#TARGET_LINK_LIBRARIES(dynamic_benchmarks LINK_PUBLIC ${Boost_LIBRARIES})
#TARGET_LINK_LIBRARIES(validity LINK_PUBLIC ${Boost_LIBRARIES})
#TARGET_LINK_LIBRARIES(microbenchmarks LINK_PUBLIC ${Boost_LIBRARIES})

#TARGET_LINK_LIBRARIES(validity cmph)
#TARGET_LINK_LIBRARIES(microbenchmarks cmph)
