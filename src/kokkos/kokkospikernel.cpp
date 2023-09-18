#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

#include <Kokkos_Core.hpp>

int main (int argc, char* argv[])
{
    std::cout << "Approximate pi using a Riemann sum..." << std::endl;
	std::cout << std::endl;

    //PI constant
    const double PI = 3.1415926535897932;
	//N: number of subintervals (2^30 by default)
	const int N = 32768 * 32768;
	//dx: size of each subinterval
	const double dx = 1.0 / double(N);
    //nrepeat: number of repetitions
    int nrepeat = 50;

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

    std::cout << "Running Kokkos sequential pi approximation..." << std::endl;
 
    double seq_totalTime;

    auto seq_t1 = std::chrono::high_resolution_clock::now();

    for(int repeat = 0; repeat < nrepeat; repeat++)
    {
        double seq_pi = 0.0;

        for (int i = 0; i < N; ++i) 
        {
            double x = (double(i) + 0.5) * dx;
            seq_pi += dx / (1.0 + x * x);
        }

        seq_pi *= 4.0;

        if(repeat == (nrepeat - 1))
        {
            std::cout << "\tpi = " << std::setprecision(prec) << seq_pi << std::endl;
            std::cout << "\terror = " << std::setprecision(prec) << fabs(seq_pi - PI) << std::endl;    
        }
    }

    auto seq_t2 = std::chrono::high_resolution_clock::now();

    seq_totalTime = std::chrono::duration_cast<std::chrono::duration<double> >(seq_t2 - seq_t1).count();

	std::cout << "Time elapsed to get the result: " << seq_totalTime << " seconds" << std::endl;
	std::cout << std::endl;
    
    #if defined(KOKKOS_ENABLE_OPENMP)

	std::cout << "Running Kokkos OpenMP pi approximation..." << std::endl;
	
    double omp_totalTime;

    auto omp_t1 = std::chrono::high_resolution_clock::now();

    for(int repeat = 0; repeat < nrepeat; repeat++)
    {
        double omp_pi = 0.0;

        Kokkos::parallel_reduce(N, KOKKOS_LAMBDA(const int i, double& omp_pi_val)
        {
            double x = (double(i) + 0.5) * dx;
		    omp_pi_val += dx / (1.0 + x * x);
        }, omp_pi);
  
        double omp_pi_r = omp_pi * 4.0;

        if(repeat == (nrepeat - 1))
        {
            std::cout << "\tpi = " << std::setprecision(prec) << omp_pi_r << std::endl;
            std::cout << "\terror = " << std::setprecision(prec) << fabs(omp_pi_r - PI) << std::endl;
        }      
    }

    auto omp_t2 = std::chrono::high_resolution_clock::now();

    omp_totalTime = std::chrono::duration_cast<std::chrono::duration<double> >(omp_t2 - omp_t1).count();

	std::cout << "Time elapsed to get the result: " << omp_totalTime << " seconds" << std::endl;
	std::cout << std::endl;

    #endif

    #if defined(KOKKOS_ENABLE_CUDA)

    std::cout << "Running Kokkos CUDA pi approximation..." << std::endl;

    double cu_totalTime;

    auto cu_t1 = std::chrono::high_resolution_clock::now();


    for(int repeat = 0; repeat < nrepeat; repeat++)
    {
        double cuda_pi = 0.0;

        Kokkos::parallel_reduce(N, KOKKOS_LAMBDA(const int i, double& cu_pi_val)
        {
            double x = (double(i) + 0.5) * dx;
		    cu_pi_val += dx / (1.0 + x * x);
        }, cuda_pi);
  
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