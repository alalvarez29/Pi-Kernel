#!/bin/bash

echo "Building the Pi Kernel benchmark"

cd ..

cmake -Bbuild -H. -DMODEL=cuda -DCMAKE_CXX_COMPILER=g++ \
-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.3/bin/nvcc -DCUDA_ARCH=sm_70

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
        ./build/cuda-pikernel
done
echo "CUDA Script Pi Kernel executed"

rm -rf build

exit
