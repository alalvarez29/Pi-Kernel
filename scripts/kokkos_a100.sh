#!/bin/bash

echo "Building the Pi Kernel benchmark"

cd ..

cmake -Bbuild -H. -DMODEL=kokkos -DKOKKOS_IN_TREE=/home/cc/kokkos -DKokkos_ENABLE_CUDA=ON -DCMAKE_CXX_COMPILER=g++ \
-DKokkos_CUDA_DIR=/usr/local/cuda-12.3 -DKokkos_ARCH_AMPERE80=ON

cmake --build build

echo ""
echo "-------------------------------------"
echo "Executing the Pi Kernel benchmark"
echo "Default Parameters"
echo "Number of steps: 2^36"
echo "Number of repetitions: 100"
echo "Precision: 16"
echo "--------------------------------------"
echo ""

for i in {0..9..1}
do
        ./build/kokkos-pikernel
done
echo "Kokkos Script Pi Kernel executed"

rm -rf build

exit
