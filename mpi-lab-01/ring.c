/*
 * A template for the 2016 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki
 * Further modifications by Krzysztof Rzadca 2018
 */
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

void catch_error(int res) {
    if (res != 0) {
        printf("Error: %d\n", res);
        exit(res);
    }
}

int main(int argc, char* argv[]) {
    struct timespec spec;
    int myRank;

    MPI_Init(&argc, &argv); /* intialize the library with parameters caught by
                               the runtime */

    int number_of_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    uint64_t res;
    if (myRank == 0) {
        res = 1;
    } else {
        catch_error(MPI_Recv(&res, 1, MPI_UINT64_T, myRank - 1, MPI_ANY_TAG,
                             MPI_COMM_WORLD, NULL));
        res *= myRank;
    }
    catch_error(MPI_Send(&res, 1, MPI_UINT64_T,
                         (myRank + 1) % number_of_processes, 0,
                         MPI_COMM_WORLD));

    if (myRank == 0) {
        catch_error(MPI_Recv(&res, 1, MPI_UINT64_T, number_of_processes - 1,
                             MPI_ANY_TAG, MPI_COMM_WORLD, NULL));
        printf("Result: %d\n", res);
    }
    MPI_Finalize(); /* mark that we've finished communicating */

    return 0;
}
