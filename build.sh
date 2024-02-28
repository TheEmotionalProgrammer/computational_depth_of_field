#!/bin/bash

find . -name CMakeCache.txt -type f -exec rm {} +
cd ken-burns
mkdir -p build
cd build
cmake ..
make

cd ../../

cd poisson_and_bilateral
mkdir -p build
cd build
cmake ..
make
cd ../../
