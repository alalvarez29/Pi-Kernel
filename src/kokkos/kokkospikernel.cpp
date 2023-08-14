#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <chrono>

#include <Kokkos_Core.hpp>

const float PI = 3.1415926535897932;
const long STEP_NUM = 32768 * 32768;
const float STEP_LENGTH = 1.0 / STEP_NUM;

int main (int argc, char* argv[])
{
    printf("Approximate pi using a Riemann sum...\n");

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

    printf("Running Kokkos sequential pi approximation...\n");
 
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

    seq_totalTime = std::chrono::duration_cast<std::chrono::duration<float> >(seq_t2 - seq_t1).count();

    printf("PI = %.16lf with error %.16lf\nTime elapsed : %f seconds.\n\n", pi, fabs(seq_pi - PI), seq_totalTime);
    assert(fabs(pi - PI) <= 0.001);

#if defined(KOKKOS_ENABLE_OPENMP)

    printf("Running Kokkos OpenMP pi approximation...\n");
	
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

    omp_totalTime = std::chrono::duration_cast<std::chrono::duration<float> >(omp_t2 - omp_t1).count();

	printf("PI = %.16lf with error %.16lf\nTime elapsed : %f seconds.\n\n", pi, fabs(omp_pi_r - PI), omp_totalTime);
    assert(fabs(omp_pi_r - PI) <= 0.001);

	#endif

#if defined(KOKKOS_ENABLE_CUDA)

    printf("Running Kokkos CUDA pi approximation...\n");

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

    cu_totalTime = std::chrono::duration_cast<std::chrono::duration<float> >(cu_t2 - cu_t1).count();

    printf("PI = %.16lf with error %.16lf\nTime elapsed : %f seconds.\n\n", pi, fabs(cu_pi_r - PI), cu_totalTime);
    assert(fabs(cu_pi_r - PI) <= 0.001);

#endif

    }
    Kokkos::finalize();

    return 0;
}