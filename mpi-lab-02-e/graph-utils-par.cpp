/*
 * A template for the 2019 MPI lab at the University of Warsaw.
 * Copyright (C) 2016, Konrad Iwanicki.
 * Refactoring 2019, Łukasz Rączkowski
 */

#include <mpi.h>

#include <cassert>
#include <cstring>

#include "graph-base.h"
#include "graph-utils.h"

void catch_error(int res) {
    if (res != 0) {
        printf("Error: %d\n", res);
        exit(res);
    }
}

int getFirstGraphRowOfProcess(int numVertices, int numProcesses, int myRank) {
    return myRank * numVertices / numProcesses;
}

Graph* createAndDistributeGraph(int numVertices, int numProcesses, int myRank) {
    assert(numProcesses >= 1 && myRank >= 0 && myRank < numProcesses);

    auto graph = allocateGraphPart(
        numVertices,
        getFirstGraphRowOfProcess(numVertices, numProcesses, myRank),
        getFirstGraphRowOfProcess(numVertices, numProcesses, myRank + 1));

    if (graph == nullptr) {
        return nullptr;
    }

    assert(graph->numVertices > 0 && graph->numVertices == numVertices);
    assert(graph->firstRowIdxIncl >= 0 &&
           graph->lastRowIdxExcl <= graph->numVertices);

    /* FIXME: implement */
    if (myRank == 0) {
        int nextRow[graph->numVertices];
        int myNextRow = 0;
        for (int j = 0; j < numProcesses; j++) {
            for (int i = getMyFirstRow(j); i < getMyFirstRow(j + 1); i++) {
                initializeGraphRow(nextRow, i, graph->numVertices);
                if (j == myRank) {
                    memcpy(graph->data[myNextRow++], nextRow,
                           sizeof(int) * graph->numVertices);
                } else {
                    catch_error(MPI_Send(nextRow, graph->numVertices, MPI_INT,
                                         j, 0, MPI_COMM_WORLD));
                }
            }
        }
    } else {
        const int n = getMyFirstRow(myRank + 1) - getMyFirstRow(myRank);
        for (int i = 0; i < n; i++) {
            catch_error(MPI_Recv(graph->data[i], graph->numVertices, MPI_INT, 0,
                                 MPI_ANY_TAG, MPI_COMM_WORLD,
                                 MPI_STATUS_IGNORE));
        }
    }

    return graph;
}

void collectAndPrintGraph(Graph* graph, int numProcesses, int myRank) {
    assert(numProcesses >= 1 && myRank >= 0 && myRank < numProcesses);
    assert(graph->numVertices > 0);
    assert(graph->firstRowIdxIncl >= 0 &&
           graph->lastRowIdxExcl <= graph->numVertices);

    /* FIXME: implement */
    if (myRank == 0) {
        int row_to_print_buf[graph->numVertices], *row_to_print;
        for (int j = 0; j < numProcesses; j++) {
            for (int i = getMyFirstRow(j); i < getMyFirstRow(j + 1); i++) {
                if (j == myRank) {
                    row_to_print = graph->data[i];
                } else {
                    catch_error(MPI_Recv(row_to_print_buf, graph->numVertices,
                                         MPI_INT, j, MPI_ANY_TAG,
                                         MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                    row_to_print = row_to_print_buf;
                }
                printGraphRow(row_to_print, i, graph->numVertices);
            }
        }
    } else {
        const int n = getMyFirstRow(myRank + 1) - getMyFirstRow(myRank);
        for (int i = 0; i < n; i++) {
            catch_error(MPI_Send(graph->data[i], graph->numVertices, MPI_INT, 0,
                                 0, MPI_COMM_WORLD));
        }
    }
}

void destroyGraph(Graph* graph, int numProcesses, int myRank) {
    freeGraphPart(graph);
}
