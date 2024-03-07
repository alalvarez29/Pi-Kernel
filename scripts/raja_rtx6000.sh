#!/bin/bash

echo "Building the Pi Kernel benchmark"

cd ..

cmake -Bbuild -H. -DMODEL=raja -DCMAKE_CXX_COMPILER=g++ -DRAJA_IN_TREE=/home/cc/raja -DENABLE_CUDA=ON -DRAJA_ENABLE_CUDA=ON \
-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.2 -DCMAKE_CUDA_COMPILER/usr/local/cuda-12.2/bin/nvcc -DCUDA_ARCH=sm_75 -DTARGET=NVIDIA -DRAJA_ENABLE_VECTORIZATION=OFF

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
        ./build/raja-pikernel
done
echo "RAJA Script Pi Kernel executed"

rm -rf build

exit
