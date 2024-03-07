#!/bin/bash

echo "Building the Pi Kernel benchmark"

cd ..

cmake -Bbuild -H. -DMODEL=omp -DCMAKE_CXX_COMPILER=g++ -DOFFLOAD=NVIDIA:sm_75 \

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
        ./build/omp-pikernel
done
echo "OpenMP Script Pi Kernel executed"

rm -rf build

exit
