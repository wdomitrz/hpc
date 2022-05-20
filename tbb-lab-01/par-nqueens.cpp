// Solves the n-quees puzzle on an n x x checkerboard.
//
// This sequential implementation is to be extended with TBB to get a
// parallel implementation.
//
// HPC course, MIM UW
// Krzysztof Rzadca, LGPL

#include <cmath>
#include <iostream>
#include <list>
#include <vector>

#include "tbb/tbb.h"

// Indexed by column. Value is the row where the queen is placed,
// or -1 if no queen.
typedef std::vector<int> Board;

void pretty_print(const Board& board) {
    for (int row = 0; row < (int)board.size(); row++) {
        for (const auto& loc : board) {
            if (loc == row)
                std::cout << "*";
            else
                std::cout << " ";
        }
        std::cout << std::endl;
    }
}

// Checks the location of queen in column 'col' against queens in cols [0, col).
bool check_col(Board& board, int col_prop) {
    int row_prop = board[col_prop];
    int col_queen = 0;
    for (auto i = board.begin(); (i != board.end()) && (col_queen < col_prop);
         ++i, ++col_queen) {
        int row_queen = *i;
        if (row_queen == row_prop) {
            return false;
        }
        if (abs(row_prop - row_queen) == col_prop - col_queen) {
            return false;
        }
    }
    return true;
}

void initialize(Board& board, int size) {
    board.reserve(size);
    for (int col = 0; col < size; ++col) board.push_back(-1);
}

// Solves starting from a partially-filled board (up to column col).
void recursive_solve(Board& partial_board, int col,
                     tbb::concurrent_queue<Board>& solutions) {
    // std::cout << "rec solve col " << col << std::endl;
    // pretty_print(b_partial);

    int b_size = partial_board.size();
    if (col == b_size) {
        solutions.push(partial_board);
    } else if (col >= 3) {
        for (int tested_row = 0; tested_row < b_size; tested_row++) {
            partial_board[col] = tested_row;
            if (check_col(partial_board, col))
                recursive_solve(partial_board, col + 1, solutions);
        }
    } else {
        tbb::parallel_for(  // execute a parallel for

            tbb::blocked_range<int>(0, b_size),  // pon a range from 1 to limit

            [&partial_board, col,
             &solutions](const tbb::blocked_range<int>&
                             r) {  // inside a loop, for a partial range r,
                                   // run this lambda
                Board my_partial_board = partial_board;
                for (int tested_row = r.begin(); tested_row != r.end();
                     tested_row++) {
                    my_partial_board[col] = tested_row;
                    if (check_col(my_partial_board, col))
                        recursive_solve(my_partial_board, col + 1, solutions);
                }
            });
    }
}

int main() {
    const int board_size = 13;
    Board board{};
    initialize(board, board_size);
    tbb::concurrent_queue<Board> solutions{};

    tbb::tick_count seq_start_time = tbb::tick_count::now();
    recursive_solve(board, 0, solutions);
    tbb::tick_count seq_end_time = tbb::tick_count::now();
    double seq_time = (seq_end_time - seq_start_time).seconds();
    std::cout << "seq time: " << seq_time << "[s]" << std::endl;

    std::cout << "solution count: " << solutions.unsafe_size() << std::endl;
    // for (const auto& sol : solutions) {
    //     pretty_print(sol);
    //     std::cout << std::endl;
    // }
}
