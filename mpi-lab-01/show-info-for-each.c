/*
 * A template for the 2016 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki
 * Further modifications by Krzysztof Rzadca 2018
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
    struct timespec spec;

    MPI_Init(&argc, &argv); /* intialize the library with parameters caught by
                               the runtime */
    int numProcesses, myRank;

    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    printf("num_processes: %d, my_rank: %d\n", numProcesses, myRank);

    MPI_Finalize(); /* mark that we've finished communicating */

    return 0;
}
