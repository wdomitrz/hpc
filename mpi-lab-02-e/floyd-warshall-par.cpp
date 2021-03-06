/*
 * A template for the 2019 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki.
 * Refactoring 2019, Łukasz Rączkowski
 */

#include <mpi.h>

#include <cassert>
#include <iostream>
#include <string>

#include "graph-base.h"
#include "graph-utils.h"

static void runFloydWarshallParallel(Graph* graph, int numProcesses,
                                     int myRank) {
    assert(numProcesses <= graph->numVertices);

    /* FIXME: implement */
    const int m = graph->numVertices;
    const int n = getMyFirstRow(myRank + 1) - getMyFirstRow(myRank);
    int my_next_node_id = 0;

    int considered_row_data[graph->numVertices], *considered_row;
    for (int k_node = 0; k_node < numProcesses;
         k_node++) {  // Iterate over vertices
        for (int k = getMyFirstRow(k_node); k < getMyFirstRow(k_node + 1);
             k++) {  // And their nodes

            if (myRank == k_node) {
                considered_row = graph->data[my_next_node_id];
                my_next_node_id++;
            } else {
                considered_row = considered_row_data;
            }
            MPI_Bcast(considered_row, graph->numVertices, MPI_INT, k_node,
                      MPI_COMM_WORLD);

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    int pathSum = graph->data[i][k] + considered_row[j];

                    if (graph->data[i][j] > pathSum) {
                        graph->data[i][j] = pathSum;
                    }
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int numVertices = 0;
    int numProcesses = 0;
    int myRank = 0;
    int showResults = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

#ifdef USE_RANDOM_GRAPH
#ifdef USE_RANDOM_SEED
    srand(USE_RANDOM_SEED);
#endif
#endif

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]).compare("--show-results") == 0) {
            showResults = 1;
        } else {
            numVertices = std::stoi(argv[i]);
        }
    }

    if (numVertices <= 0) {
        std::cerr << "Usage: " << argv[0] << "  [--show-results] <num_vertices>"
                  << std::endl;
        MPI_Finalize();
        return 1;
    }

    if (numProcesses > numVertices) {
        numProcesses = numVertices;

        if (myRank >= numProcesses) {
            MPI_Finalize();
            return 0;
        }
    }

    std::cerr << "Running the Floyd-Warshall algorithm for a graph with "
              << numVertices << " vertices." << std::endl;

    auto graph = createAndDistributeGraph(numVertices, numProcesses, myRank);
    if (graph == nullptr) {
        std::cerr << "Error distributing the graph for the algorithm."
                  << std::endl;
        MPI_Finalize();
        return 2;
    }

    if (showResults) {
        collectAndPrintGraph(graph, numProcesses, myRank);
    }

    double startTime = MPI_Wtime();

    runFloydWarshallParallel(graph, numProcesses, myRank);

    double endTime = MPI_Wtime();

    std::cerr << "The time required for the Floyd-Warshall algorithm on a "
              << numVertices << "-node graph with " << numProcesses
              << " process(es): " << endTime - startTime << std::endl;

    if (showResults) {
        collectAndPrintGraph(graph, numProcesses, myRank);
    }

    destroyGraph(graph, numProcesses, myRank);

    MPI_Finalize();

    return 0;
}
