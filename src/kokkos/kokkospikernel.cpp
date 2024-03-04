#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

#include <Kokkos_Vector.hpp>

int main (int argc, char* argv[])
{
    std::cout << "Approximate pi using a Riemann sum..." << std::endl;
        std::cout << std::endl;

    //PI constant
    const double PI = 3.1415926535897932;
    //N: number of subintervals (2^33 by default)
    const int64_t N = pow(2LL, 36LL);
    //dx: size of each subinterval
    const double dx = 1.0 / N;
    //nrepeat: number of repetitions
    const int nrepeat = 100;

    //Set the precision for printing pi
    int prec = 16;

    Kokkos::initialize (argc, argv);
    {
    #ifdef KOKKOS_ENABLE_CUDA
    #define MemSpace Kokkos::CudaSpace
    #endif

    #ifndef MemSpace
    #define MemSpace Kokkos::HostSpace
    #endif

    using ExecSpace = MemSpace::execution_space;
    using range_policy = Kokkos::RangePolicy<ExecSpace>;

    //Kokkos CUDA implementation
    #if defined(KOKKOS_ENABLE_CUDA)

    std::cout << "Running Kokkos CUDA pi approximation..." << std::endl;

    double cu_totalTime;

    auto cu_t1 = std::chrono::high_resolution_clock::now();

    for(int repeat = 0; repeat < nrepeat; repeat++)
    {
        double cuda_pi = 0.0;

        Kokkos::parallel_reduce(Kokkos::RangePolicy <Kokkos::IndexType<int64_t>> (0,N),                                                                                          KOKKOS_LAMBDA(const int64_t i, double& cu_pi_val)
        {
            double x = (i + 0.5) * dx;

            cu_pi_val += dx / (1.0 + x * x);

        }, cuda_pi);
        Kokkos::fence();

        double cu_pi_r = cuda_pi * 4.0;

        if(repeat == (nrepeat -1))
        {
            std::cout << "\tpi = " << std::setprecision(prec) << cu_pi_r << std::endl;
            std::cout << "\terror = " << std::setprecision(prec) << fabs(cu_pi_r - PI) << std::endl;
        }
    }

    auto cu_t2 = std::chrono::high_resolution_clock::now();
    cu_totalTime = std::chrono::duration_cast<std::chrono::duration<double> >(cu_t2 - cu_t1).count();

        std::cout << "Time elapsed to get the result: " << cu_totalTime << " seconds" << std::endl;
        std::cout << std::endl;

    #endif

    }
    Kokkos::finalize();

    return 0;
}