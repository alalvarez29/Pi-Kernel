#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <array>
#include <CL/sycl.hpp>
using namespace cl::sycl;

#define NTRD 512

int main(int argc, char* argv[])
{
    std::cout << "Approximate pi using a Riemann sum..." << std::endl;
    std::cout << std::endl;

    const double PI = 3.1415926535897932;
    const int N = 32768 * 32768;
    const double dx = 1.0 / double(N);

    int nrepeat = 50;
    int prec = 16;

    double totalTime, x;

    std::array<double, NTRD> sum;
    for(int i = 0; i < NTRD; i++) sum[i] = 0.0;
    {
        queue q(default_selector_v);
        std::cout << "Running on: " << q.get_device().get_info<info::device::name>() << std::endl;
        range<1> sizeBuf{NTRD};
        buffer<double, 1> sumBuf(sum.data(), sizeBuf);
        q.submit([&](handler &h)
        {
            auto sumAccessor = sumBuf.get_access<access::mode::read_write>(h);
            h.parallel_for(sizeBuf, [=](id<1> tid)
            {
                for(int i=tid; i < N; i+=NTRD)
                {
                    double x = (double(i) + 0.5) * dx;
                    sumAccessor[tid] += 4.0 / (1.0 + x * x);
                }
            });
        });
    }

    double pi = 0.0;
    for(int i = 0; i < NTRD; i++)
    {
        pi += sum[i];
    }
    pi *= dx;

    auto t1 = std::chrono::high_resolution_clock::now();

    auto t2 = std::chrono::high_resolution_clock::now();

    totalTime = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

	std::cout << "Time elapsed to get the result: " << totalTime << " seconds" << std::endl;
	std::cout << std::endl;

    std::cout << "\tpi = " << std::setprecision(prec) << pi << std::endl;
    std::cout << "\terror = " << std::setprecision(prec) << fabs(pi - PI) << std::endl;
}