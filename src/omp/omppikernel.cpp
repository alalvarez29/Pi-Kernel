#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>

#include <omp.h>

int main(int argc, char* argv[])
{
    std::cout << "Approximate pi using a Riemann sum..." << std::endl;
    std::cout << std::endl;

    const double PI = 3.1415926535897932;
    const long N = pow(2,36);
    const double dx = 1.0 / double(N);
    const int nrepeat = 100;
    const int prec = 16;

    double totalTime, x;

    auto t1 = std::chrono::high_resolution_clock::now();

    for(int repeat = 0; repeat < nrepeat; repeat++)
    {
        double omp_pi = 0.0;

        #pragma omp parallel for reduction(+:omp_pi) private(x)
        for(int i = 0; i < N; i++)
        {
            x = (double(i) + 0.5) * dx;
            omp_pi += dx / (1.0 + x * x);
        }

        omp_pi *= 4.0;

        if(repeat == (nrepeat - 1))
        {
            std::cout << "\tpi = " << std::setprecision(prec) << omp_pi << std::endl;
            std::cout << "\terror = " << std::setprecision(prec) << fabs(omp_pi - PI) << std::endl;    
        }         
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    totalTime = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

	std::cout << "Time elapsed to get the result: " << totalTime << " seconds" << std::endl;
	std::cout << std::endl;
}
