/*
 * A template for the 2016 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki
 * Further modifications by Krzysztof Rzadca 2018
 */
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define BENCHMARKS 10

void catch_error(int res) {
    if (res != 0) {
        printf("Error: %d\n", res);
        exit(res);
    }
}

double benchmark(size_t size) {
    int number_of_processes;
    int myRank;
    MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);
    if (number_of_processes != 2) {
        printf("Wrong number of processes, expected %d, got %d\n", 2,
               number_of_processes);
        exit(1);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    double startTime;
    double endTime;
    double executionTime;
    char msg[size];
    memset(msg, 123, size);

    startTime = MPI_Wtime();
    if (myRank == 0) {
        catch_error(MPI_Send(&msg, size, MPI_CHAR, !myRank, 0, MPI_COMM_WORLD));
        catch_error(MPI_Recv(&msg, size, MPI_CHAR, !myRank, MPI_ANY_TAG,
                             MPI_COMM_WORLD, NULL));
    } else {
        catch_error(MPI_Recv(&msg, size, MPI_CHAR, !myRank, MPI_ANY_TAG,
                             MPI_COMM_WORLD, NULL));
        catch_error(MPI_Send(&msg, size, MPI_CHAR, !myRank, 0, MPI_COMM_WORLD));
    }

    endTime = MPI_Wtime();

    executionTime = endTime - startTime;
    return executionTime;
}

int main(int argc, char* argv[]) {
    struct timespec spec;
    const int REPEAT = 30;

    MPI_Init(&argc, &argv);

    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    for (size_t size = 1; size <= 1000 * 1000; size *= 10) {
        for (int j = 0; j < REPEAT; j++) {
            double res = benchmark(size);
            if (myRank == 0) printf("%d %d %lf\n", j, size, res);
        }
    }

    MPI_Finalize(); /* mark that we've finished communicating */

    return 0;
}
