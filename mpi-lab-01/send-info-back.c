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
    int myRank;

    MPI_Init(&argc, &argv); /* intialize the library with parameters caught by
                               the runtime */

    clock_gettime(CLOCK_REALTIME, &spec);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    srand(spec.tv_nsec);  // use nsec to have a different value across different
                          // processes
    unsigned value = rand() % 11;
    printf("Local %d: %d\n", myRank, value);

    if (myRank != 0) {
        int msg[2] = {myRank, value};
        int res = MPI_Send(msg, 2, MPI_INT, 0, 0, MPI_COMM_WORLD);
        if (res != 0) {
            printf("Error send %d: %d\n", myRank, res);
        }
    } else {
        int number_of_processes;
        MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);

        MPI_Status statuses[number_of_processes];
        int msg[2 * number_of_processes];
        msg[0] = myRank;
        msg[0 + 1] = value;

        for (int i = 1; i < number_of_processes; i++) {
            int res = MPI_Recv(&msg[2 * i], 2, MPI_INT, MPI_ANY_SOURCE,
                               MPI_ANY_TAG, MPI_COMM_WORLD, &statuses[i]);
            if (res != 0) {
                printf("Error recv %d: %d\n", myRank, res);
            }
            printf("Received from %d (or %d): %d\n", statuses[i].MPI_SOURCE,
                   msg[2 * i], msg[2 * i + 1]);
        }
    }

    MPI_Finalize(); /* mark that we've finished communicating */

    return 0;
}
