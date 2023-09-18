#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

#include "RAJA/RAJA.hpp"


//CUDA_BLOCK_SIZE specifies the number of threads in a CUDA thread block 
//256 by default
#if defined(RAJA_ENABLE_CUDA)
const int CUDA_BLOCK_SIZE = 256;
#endif

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
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

	std::cout << "Running RAJA sequential pi approximation..." << std::endl;

	using EXEC_POL1 = RAJA::seq_exec;
	using REDUCE_POL1 = RAJA::seq_reduce;

	double seq_totalTime;

	auto seq_t1 = std::chrono::high_resolution_clock::now();

	for(int repeat = 0; repeat < nrepeat; repeat++)
	{
		RAJA::ReduceSum <REDUCE_POL1, double> seq_pi(0.0);

		RAJA::forall <EXEC_POL1>(RAJA::RangeSegment(0, N), [=](int i)
		{
			double x = (double(i) + 0.5) * dx;
			seq_pi += dx / (1.0 + x * x);
		});

		double seq_pi_val = seq_pi.get() * 4.0;

		if(repeat == (nrepeat - 1))
		{
			std::cout << "\tpi = " << std::setprecision(prec) << seq_pi_val << std::endl;
			std::cout << "\terror = " << std::setprecision(prec) << fabs(seq_pi_val - PI) << std::endl;
		}
	}

	auto seq_t2 = std::chrono::high_resolution_clock::now();

	seq_totalTime = std::chrono::duration_cast<std::chrono::duration<double> >(seq_t2 - seq_t1).count();

	std::cout << "Time elapsed to get the result: " << seq_totalTime << " seconds" << std::endl;
	std::cout << std::endl;

	//RAJA OpenMP implementation
	#if defined(RAJA_ENABLE_OPENMP)

		std::cout << "Running RAJA OpenMP pi approximation..." << std::endl;

		using EXEC_POL2 = RAJA::omp_parallel_for_exec;
		using REDUCE_POL2 = RAJA::omp_reduce;

		double omp_totalTime;

		auto omp_t1 = std::chrono::high_resolution_clock::now();

		for(int repeat = 0; repeat < nrepeat; repeat++)
		{
			RAJA::ReduceSum <REDUCE_POL2, double> omp_pi(0.0);

			RAJA::forall <EXEC_POL2>(RAJA::RangeSegment(0, N), [=](int i)
			{
				double x = (double(i) + 0.5) * dx;
				omp_pi += dx / (1.0 + x * x);
			});

			double omp_pi_val = omp_pi.get() * 4.0;

			if(repeat == (nrepeat -1))
			{
				std::cout << "\tpi = " << std::setprecision(prec) << omp_pi_val << std::endl;
				std::cout << "\terror = " << std::setprecision(prec) << fabs(omp_pi_val - PI) << std::endl;
			}
		}

		auto omp_t2 = std::chrono::high_resolution_clock::now();

		omp_totalTime = std::chrono::duration_cast<std::chrono::duration<double> >(omp_t2 - omp_t1).count();

		std::cout << "Time elapsed to get the result: " << omp_totalTime << " seconds" << std::endl;
		std::cout << std::endl;

	#endif

	//RAJA CUDA implementation
	#if defined(RAJA_ENABLE_CUDA)

		std::cout << "Running RAJA CUDA pi approximation..." << std::endl;

		using EXEC_POL3 = RAJA::cuda_exec<CUDA_BLOCK_SIZE>;
		using REDUCE_POL3 = RAJA::cuda_reduce;

		
		double cu_totalTime;

		auto cu_t1 = std::chrono::high_resolution_clock::now();

		for(int repeat = 0; repeat < nrepeat; repeat++)
		{
			RAJA::ReduceSum <REDUCE_POL3, double> cuda_pi(0.0);

			RAJA::forall <EXEC_POL3>(RAJA::RangeSegment(0, N), [=] RAJA_DEVICE (int i)
			{
				double x = (double(i) + 0.5) * dx;
				cuda_pi += dx / (1.0 + x * x);
			});
			double cuda_pi_val = cuda_pi.get() * 4.0;

			if(repeat == (nrepeat - 1))
			{
				std::cout << "\tpi = " << std::setprecision(prec) << cuda_pi_val << std::endl;
				std::cout << "\terror = " << std::setprecision(prec) << fabs(cuda_pi_val - PI) << std::endl;
			}
		}

		auto cu_t2 = std::chrono::high_resolution_clock::now();

		cu_totalTime = std::chrono::duration_cast<std::chrono::duration<double> >(cu_t2 - cu_t1).count();

        std::cout << "Time elapsed to get the result: " << cu_totalTime << " seconds" << std::endl;
		std::cout << std::endl;

	#endif

	return 0;
}
