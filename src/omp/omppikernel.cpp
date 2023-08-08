#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <chrono>

#include <omp.h>

int main(int argc, char* argv[])
{
    std::cout << "Approximate pi using a Riemann sum..." << std::endl;
    std::cout << std::endl;

    const int N = 32768 * 32768;

    const double dx = 1.0 / double(N);

    int prec = 16;

    double sum = 0.0;

    double pi, x, totalTime;

    auto t1 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for reduction(+:sum) private(x)
    for(int i = 0; i < N; i++)
    {
        x = (i + 0.5) * dx;
        sum += 1.0 / (1.0 + x * x);
    }
    pi = dx * sum * 4;

    auto t2 = std::chrono::high_resolution_clock::now();

    totalTime = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

	std::cout << "\tpi = " << std::setprecision(prec) << pi << std::endl;

	std::cout << "Time elapsed to get the result: " << totalTime << " seconds" << std::endl;
	std::cout << std::endl;
}