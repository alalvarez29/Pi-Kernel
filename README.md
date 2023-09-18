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

<p align="center">
    | Parameter | Value | Description |
    | --- | --- | --- |
    | N | 2^30 = 32768∗32768 | Number of subintervals |
    | dx | 1 / N | Size of each subinterval |
</p>


## Programming models


## Building 


## References