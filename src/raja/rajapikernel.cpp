#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <chrono>

#include "RAJA/RAJA.hpp"

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{
	std::cout << "Approximate pi using a Riemann sum..." << std::endl;

	const int N = 32768 * 32768;
	const double dx = 1.0 / double(N);

	int prec = 16;

	#if defined(RAJA_ENABLE_OPENMP)

		std::cout << "Running RAJA OpenMP pi approximation..." << std::endl;

		using EXEC_POL2 = RAJA::omp_parallel_for_exec;
		using REDUCE_POL2 = RAJA::omp_reduce;

		RAJA::ReduceSum <REDUCE_POL2, double> omp_pi(0.0);		
		
		double omp_totalTime;

		auto omp_t1 = std::chrono::high_resolution_clock::now();

		RAJA::forall <EXEC_POL2>(RAJA::RangeSegment(0, N), [=](int i)
		{
			double x = (double(i) + 0.5) * dx;
			omp_pi += dx / (1.0 + x * x);
		});

		double omp_pi_val = omp_pi.get() * 4.0;

		auto omp_t2 = std::chrono::high_resolution_clock::now();

		omp_totalTime = std::chrono::duration_cast<std::chrono::duration<double> >(omp_t2 - omp_t1).count();

		std::cout << "pi = " << std::setprecision(prec) << omp_pi_val << std::endl;

		std::cout << "Time elapsep to get the result: " << omp_totalTime << " seconds" << std::endl;

	#endif

	#if defined(RAJA_ENABLE_CUDA)

		std::cout << "Running RAJA CUDA pi approximation..." << std::endl;

		using EXEC_POL3 = RAJA::cuda_exec<CUDA_BLOCK_SIZE>;
		using REDUCE_POL3 = RAJA::cuda_reduce;

		RAJA::ReduceSum <REDUCE_POL3, double> cuda_pi(0.0);
		
		double cu_totalTime;

		auto cu_t1 = std::chrono::high_resolution_clock::now();

		RAJA::forall <EXEC_POL3>(RAJA::RangeSegment(0, N), [=] RAJA_DEVICE (int i)
		{
			double x = (double(i) + 0.5) * dx;
			cuda_pi += dx / (1.0 + x * x);
		});
		double cuda_pi_val = cuda_pi.get() * 4.0;

		auto cu_t2 = std::chrono::high_resolution_clock::now();

		cu_totalTime = std::chrono::duration_cast<std::chrono::duration<double> >(cu_t2 - cu_t1).count();

		std::cout << "pi = " << std::setprecision(prec) << cuda_pi_val << std::endl;

                std::cout << "Time elapsep to get the result: " << cu_totalTime << " seconds" << std::endl;

	#endif

	return 0;
}
