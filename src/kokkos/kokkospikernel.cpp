#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <chrono>

#include <Kokkos_Core.hpp>

int main (int argc, char* argv[])
{
    std::cout << "Approximate pi using a Riemann sum..." << std::endl;
	std::cout << std::endl;

	//N: number of subintervals (2^30 by default)
	const int N = 32768 * 32768;
	//dx: size of each subinterval
	const double dx = 1.0 / double(N);

	//Set the precision for printing pi
	int prec = 16;

    Kokkos::initialize (argc, argv);
    {
    #ifdef KOKKOS_ENABLE_CUDA
    #define MemSpace Kokkos::CudaSpace
    #endif
    #ifdef KOKKOS_ENABLE_OPENMPTARGET
    #define MemSpace Kokkos::OpenMPTargetSpace
    #endif

    #ifndef MemSpace
    #define MemSpace Kokkos::HostSpace
    #endif

    using ExecSpace = MemSpace::execution_space;
    using range_policy = Kokkos::RangePolicy<ExecSpace>;

    double pi = 0.0;

    double totalTime;

    auto t1 = std::chrono::high_resolution_clock::now();

    Kokkos::parallel_reduce(N, KOKKOS_LAMBDA(const int i, double& pi_val)
    {
        double x = (double(i) + 0.5) * dx;
		pi_val += dx / (1.0 + x * x);
    }, pi);
  
    double pi_r = pi * 4.0;

    auto t2 = std::chrono::high_resolution_clock::now();

    totalTime = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

	std::cout << "\tpi = " << std::setprecision(prec) << pi_r << std::endl;

	std::cout << "Time elapsed to get the result: " << totalTime << " seconds" << std::endl;
	std::cout << std::endl;

    }
    Kokkos::finalize();

    return 0;
}