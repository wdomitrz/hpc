// Illustrates parallel for in TBB.
//
// HPC course, MIM UW
// Krzysztof Rzadca, LGPL

#include <math.h>

#include <iostream>

#include "tbb/tbb.h"

bool is_prime(long num) {
    long sqrt_num = (long)sqrt(num);
    for (long div = 2; div <= sqrt_num; ++div) {
        if ((num % div) == 0) return false;
    }
    return true;
}

int main() {
    const long limit = 5'000'000;
    tbb::tick_count seq_start_time = tbb::tick_count::now();
    for (long i = 1; i < limit; ++i) {
        is_prime(i);
    }
    tbb::tick_count seq_end_time = tbb::tick_count::now();
    double seq_time = (seq_end_time - seq_start_time).seconds();
    std::cout << "seq time for " << limit << " " << seq_time << "[s]"
              << std::endl;

#ifdef NO_ATOMIC
    long primes_counter = 0;
#else
    std::atomic<long> primes_counter(0);
#endif
    std::atomic<long long> ran_len_sq(0), ran_numb(0);
    tbb::tick_count par_start_time = tbb::tick_count::now();
    tbb::parallel_for(                       // execute a parallel for
        tbb::blocked_range<long>(1, limit),  // pon a range from 1 to limit
        [&primes_counter, &ran_len_sq,
         &ran_numb](const tbb::blocked_range<long>&
                        r) {  // inside a loop, for a partial range r,
            long long len = r.end() - r.begin();
            ran_len_sq += len * len;
            ran_numb++;
            long local_counter = 0;
            // run this lambda
            for (long i = r.begin(); i != r.end(); ++i) {
                if (is_prime(i)) local_counter++;
            }
            primes_counter += local_counter;
        });
    tbb::tick_count par_end_time = tbb::tick_count::now();
    double par_time = (par_end_time - par_start_time).seconds();
    std::cout << "par time for " << limit << " " << par_time << "[s]"
              << std::endl;
    std::cout << "primes count:" << primes_counter << "\n";

    std::cout << "numer of ranges: " << ran_numb << "\n"
              << "mean len: " << ((double)limit) / ran_numb << "\n"
              << "std len: "
              << sqrt(((double)ran_len_sq) / ran_numb -
                      (((double)limit) / ran_numb) *
                          (((double)limit) / ran_numb))
              << "\n";
}
