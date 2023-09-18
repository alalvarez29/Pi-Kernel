# Pi-Kernel benchmark

This benchmark implements an algorithm to calculate the approximation of the pi number from the Riemann sum for exploring the portability performance of different parallel programming models. 

## Contents

- [Implementation](#implementation)
    - [Riemann sum](#riemann-sum)
    - [Parameters](#Parameters)
- [Programming models](#programming-models)
- [Building](#building)
- [References](#References)

## Implementation

### Riemann sum

This kernel is based on the Riemann sum to get an estimate of the pi number, the following equations describes the mathematical implementation of the aproximation and its discrete form.

<p align="center">
    <img src="imgs/1.jpeg?raw=true" alt="eq"/>
    <img src="imgs/2.jpeg?raw=true" alt="eq"/>
</p>

### Parameters

The parameters used in this benchmark are the following:

| Parameter | Value | Description |
| --- | --- | --- |
| N | 2^30 = 32768∗32768 | Number of subintervals |
| dx | 1 / N | Size of each subinterval |
| nrepeat | 50 | Number of repetitions |
| prec | 16 | Digit precision for the result |

## Programming models

Pi-Kernel is supported by the following parallel programming models:

- Kokkos
- RAJA
- OpenMP
- SYCL
- CUDA

## Building 

The building of the implementation of this benchmark requires the following steps:

1. Download the source code
```
$ git clone https://github.com/uqbarcitizen/Pi-Kernel.git

```

## References