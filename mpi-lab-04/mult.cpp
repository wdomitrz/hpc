#include <chrono>
#include <iostream>  // HPC can into CPP!
#include <random>

inline void mult_mm(double* A, double* B, double* C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i * n + j] += A[i * n + k] * B[k * n + j];
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "invocation: " << argv[0] << " matrix_size " << std::endl;
        return 1;
    }
    const long long n{std::stoi(std::string{argv[1]})};
    std::mt19937_64 rnd;
    std::uniform_real_distribution<double> doubleDist{0, 1};

    double* A = new double[n * n];
    double* B = new double[n * n];
    double* C = new double[n * n];

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i * n + j] = doubleDist(rnd);
            B[i * n + j] = doubleDist(rnd);
            C[i * n + j] = 0;
        }
    }

    auto startTime = std::chrono::steady_clock::now();
    mult_mm(A, B, C, n);
    auto finishTime = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed{finishTime - startTime};
    std::cout << "My multiplication elapsed time: " << elapsed.count() << "[s]"
              << std::endl;
}
