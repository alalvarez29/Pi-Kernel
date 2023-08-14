#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <chrono>

#include <Kokkos_Core.hpp>

int main (int argc, char* argv[])
{
    std::cout << "Approximate pi using a Riemann sum..." << std::endl;
	std::cout << std::endl;

	//number of subintervals (2^30 by default)
	const int STEP_NUM = 32768 * 32768;
	//size of each subinterval
	const float STEP_LENGTH = 1.0 / float(STEP_NUM);

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

    std::cout << "Running sequential pi approximation..." << std::endl;
 
    float seq_pi = 0.0;

    float seq_totalTime;

    auto seq_t1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < STEP_NUM; ++i) 
    {
        float x = (float(i) + 0.5) * STEP_LENGTH;
        seq_pi += STEP_LENGTH / (1.0 + x * x);
    }

    seq_pi *= 4.0;

    auto seq_t2 = std::chrono::high_resolution_clock::now();

    seq_totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(seq_t2 - seq_t1).count();

	std::cout << "\tpi = " << std::setprecision(prec) << seq_pi << std::endl;

	std::cout << "Time elapsed to get the result: " << seq_totalTime / 1000 << " seconds" << std::endl;
	std::cout << std::endl;


#if defined(KOKKOS_ENABLE_OPENMP)

	std::cout << "Running Kokkos OpenMP pi approximation..." << std::endl;
	
    float omp_pi = 0.0;

    float omp_totalTime;

    auto omp_t1 = std::chrono::high_resolution_clock::now();

    Kokkos::parallel_reduce(STEP_NUM, KOKKOS_LAMBDA(const int i, float& omp_pi_val)
    {
        float x = (float(i) + 0.5) * STEP_LENGTH;
		omp_pi_val += STEP_LENGTH / (1.0 + x * x);
    }, omp_pi);
  
    float omp_pi_r = omp_pi * 4.0;

    auto omp_t2 = std::chrono::high_resolution_clock::now();

    omp_totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(omp_t2 - omp_t1).count();

	std::cout << "\tpi = " << std::setprecision(prec) << omp_pi_r << std::endl;

	std::cout << "Time elapsed to get the result: " << omp_totalTime / 1000 << " seconds" << std::endl;
	std::cout << std::endl;

	#endif

#if defined(KOKKOS_ENABLE_CUDA)

    std::cout << "Running Kokkos CUDA pi approximation..." << std::endl;

    float cuda_pi = 0.0;

    float cu_totalTime;

    auto cu_t1 = std::chrono::high_resolution_clock::now();

    Kokkos::parallel_reduce(STEP_NUM, KOKKOS_LAMBDA(const int i, float& cu_pi_val)
    {
        float x = (float(i) + 0.5) * STEP_LENGTH;
		cu_pi_val += STEP_LENGTH / (1.0 + x * x);
    }, cuda_pi);
  
    float cu_pi_r = cuda_pi * 4.0;

    auto cu_t2 = std::chrono::high_resolution_clock::now();

    cu_totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(cu_t2 - cu_t1).count();

	std::cout << "\tpi = " << std::setprecision(prec) << cu_pi_r << std::endl;

	std::cout << "Time elapsed to get the result: " << cu_totalTime / 1000 << " seconds" << std::endl;
	std::cout << std::endl;

#endif

    }
    Kokkos::finalize();

    return 0;
}