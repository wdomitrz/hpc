#include <omp.h>

#include <iomanip>
#include <iostream>

// #define STEPS 1000
 // #define THREADS \
    16  // you can also use the OMP_NUM_THREADS environmental variable

double power(double x, long n) {
    if (n == 0) {
        return 1;
    }

    return x * power(x, n - 1);
}

double calcPi(long n) {
    if (n < 0) {
        return 0;
    }

    return 1.0 / power(16, n) *
               (4.0 / (8 * n + 1.0) - 2.0 / (8 * n + 4.0) -
                1.0 / (8 * n + 5.0) - 1.0 / (8 * n + 6.0)) +
           calcPi(n - 1);
}

double powerParallelReduction(double x, long n) {
    // PUT IMPLEMENTATION HERE
    double res = 1;
#pragma omp parallel for reduction(* : res)
    for (long i = 0; i < n; i++) {
        res *= x;
    }
    return res;
}

double powerParallelCritical(double x, long n) {
    // PUT IMPLEMENTATION HERE
    double res = 1;
#pragma omp parallel for
    for (long i = 0; i < n; i++) {
#pragma omp critical
        { res *= x; }
    }
    return res;
}

double calcPiParallelReduction(long n) {
    // PUT IMPLEMENTATION HERE
    double res = 0;
#pragma omp parallel for reduction(+ : res)
    for (long i = 0; i <= n; i++) {
        double current_factor = 1.0 / powerParallelReduction(16, i) *
                                (4.0 / (8 * i + 1.0) - 2.0 / (8 * i + 4.0) -
                                 1.0 / (8 * i + 5.0) - 1.0 / (8 * i + 6.0));
        res += current_factor;
    }
    return res;
}

double calcPiParallelCritical(long n) {
    // PUT IMPLEMENTATION HEREdouble res = 0;
    double res = 0;
#pragma omp parallel for
    for (long i = 0; i <= n; i++) {
        double current_factor = 1.0 / powerParallelCritical(16, i) *
                                (4.0 / (8 * i + 1.0) - 2.0 / (8 * i + 4.0) -
                                 1.0 / (8 * i + 5.0) - 1.0 / (8 * i + 6.0));
#pragma omp critical
        { res += current_factor; }
    }
    return res;
}

int main(int argc, char *argv[]) {
#ifdef SEQUENTIAL
    std::cout << std::setprecision(10) << calcPi(STEPS) << std::endl;
#endif
#ifdef REDUCTION
    std::cout << std::setprecision(10) << calcPiParallelReduction(STEPS)
              << std::endl;
#endif
#ifdef CRITICAL
    std::cout << std::setprecision(10) << calcPiParallelCritical(STEPS)
              << std::endl;
#endif
    return 0;
}
