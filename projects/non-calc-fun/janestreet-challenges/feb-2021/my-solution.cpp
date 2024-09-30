/* 

This is a challenge problem for a reason. 

 Key Components of MY & JS's Solution:
 
 * 1. Constants and Data Structures:
 *    - Grid size, number of hooks, regions, target sum
 *    - Cell structure to represent grid positions
 *    - Vector of regions (19 in total)
 *    - Initial possible numbers for each hook
 *
 * 2. Hook Placement:
 *    - Generate all possible hook placements (4^8 possibilities)
 *    - Place hooks on the grid based on positions
 *    - Check initial constraints to filter valid placements
 *
 * 3. Solving the Grid:
 *    - Use recursive backtracking to fill the grid
 *    - For each cell, try placing valid numbers from the corresponding hook
 *    - Check constraints after each placement:
 *      a) 2x2 regions have at least one empty cell
 *      b) Sums in shaded regions equal the target (15)
 *      c) Filled squares form a connected region
 *
 * 4. Constraint Checking Functions:
 *    - check_initial_constraints: Verify given clues and sums
 *    - is_connected: Use BFS to ensure filled squares are connected
 *    - check_regions: Verify sums in shaded regions
 *    - check_2x2_regions: Ensure at least one empty cell in each 2x2 area
 *
 * 5. Solution Calculation:
 *    - calculate_product_of_empty_regions: Find connected empty regions and multiply their sizes
 *
 * Algorithm Flow:
 * 1. Generate all possible hook placements
 * 2. Filter placements based on initial constraints
 * 3. For each valid placement:
 *    a) Initialize the grid with hooks
 *    b) Use recursive backtracking to fill the grid
 *    c) Check all constraints during filling
 * 4. If a valid solution is found:
 *    a) Calculate the product of empty region sizes
 *    b) Return the result as the puzzle answer
 *
 * Mathematical Concepts Used:
 * - Combinatorics: Generating hook placements (4^8 possibilities)
 * - Graph Theory: Checking connectivity of filled squares (BFS)
 * - Constraint Satisfaction: Backtracking to find a valid solution
 * - Geometry: Working with L-shaped hooks and 2x2 regions
 * - Arithmetic: Calculating sums and products
 *
 * Time Complexity:
 * - Worst case: O(4^8 * 9^81), but practical runtime is much lower due to early constraint checking
 * - Generating placements: O(4^8)
 * - Solving each grid: O(9^81) theoretically, but much less in practice
 *
 * Space Complexity:
 * - O(9^2) for the grid representation
 * - O(4^8) for storing valid hook placements
 * - O(9) for recursion depth in solving


*/ 

#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <bitset>
#include <unordered_map>
#include <queue>

// Define constants
const int GRID_SIZE = 9;
const int TOTAL_CELLS = GRID_SIZE * GRID_SIZE;
const int NUM_HOOKS = 9; // Hooks from size 9 to 1
const int NUM_REGIONS = 19;
const int TARGET_SUM = 15;

// Structure to represent a cell
struct Cell {
    int x, y;
    Cell(int x_, int y_) : x(x_), y(y_) {}
};

// Regions and their corresponding cells
std::vector<std::vector<Cell>> regions = {
    { Cell(0,0), Cell(0,1), Cell(1,0), Cell(1,1) },
    { Cell(0,4), Cell(1,3) },
    { Cell(0,5), Cell(0,6), Cell(1,5), Cell(1,6) },
    { Cell(0,7), Cell(0,8), Cell(1,7), Cell(1,8) },
    { Cell(2,7), Cell(2,8) },
    { Cell(2,0), Cell(2,1), Cell(3,0) },
    { Cell(0,3), Cell(1,2), Cell(2,2), Cell(2,3), Cell(2,4), Cell(3,4) },
    { Cell(3,1), Cell(4,0), Cell(4,1) },
    { Cell(0,2), Cell(1,2), Cell(2,2), Cell(3,2), Cell(3,3), Cell(4,3) },
    { Cell(3,7), Cell(3,8), Cell(4,8) },
    { Cell(2,5), Cell(2,6), Cell(3,5), Cell(3,6), Cell(4,6) },
    { Cell(4,7), Cell(5,7), Cell(5,8) },
    { Cell(5,0), Cell(5,1), Cell(6,0) },
    { Cell(4,2), Cell(5,2), Cell(5,3), Cell(6,2), Cell(7,2) },
    { Cell(4,4), Cell(5,4), Cell(6,3), Cell(6,4), Cell(6,5), Cell(6,6), Cell(7,5) },
    { Cell(4,5), Cell(5,5), Cell(5,6), Cell(6,1), Cell(6,7), Cell(7,0), Cell(7,1),
      Cell(7,3), Cell(7,4), Cell(7,6), Cell(7,7), Cell(7,8) },
    { Cell(7,7), Cell(8,7) },
    { Cell(8,0), Cell(8,1), Cell(8,2) },
    { Cell(8,3), Cell(8,4), Cell(8,5), Cell(8,6) }
};

// Initial possible numbers for each hook
std::vector<std::vector<int>> hook_numbers = {
    {}, // Hook 0 (unused)
    {1},
    {2,2,0},
    {3,3,3,0,0},
    {4,4,4,4,0,0,0},
    {5,5,5,5,5,0,0,0,0},
    {6,6,6,6,6,6,0,0,0,0,0},
    {7,7,7,7,7,7,7,0,0,0,0,0,0},
    {8,8,8,8,8,8,8,8,0,0,0,0,0,0,0},
    {9,9,9,9,9,9,9,9,9,0,0,0,0,0,0,0,0}
};

// Function to place hooks on the grid
void place_hooks(const std::array<int, 8>& positions, int grid[GRID_SIZE][GRID_SIZE]) {
    int x_left = 0, x_right = GRID_SIZE - 1;
    int y_top = GRID_SIZE - 1, y_bottom = 0;
    for (int i = 0; i < 8; ++i) {
        int pos = positions[i];
        int hook_num = NUM_HOOKS - i;
        switch (pos) {
            case 0: // Bottom left corner
                for (int y = y_bottom; y <= y_top; ++y)
                    grid[x_left][y] = hook_num;
                for (int x = x_left; x <= x_right; ++x)
                    grid[x][y_bottom] = hook_num;
                ++x_left; ++y_bottom;
                break;
            case 1: // Top left corner
                for (int y = y_bottom; y <= y_top; ++y)
                    grid[x_left][y] = hook_num;
                for (int x = x_left; x <= x_right; ++x)
                    grid[x][y_top] = hook_num;
                ++x_left; --y_top;
                break;
            case 2: // Top right corner
                for (int y = y_bottom; y <= y_top; ++y)
                    grid[x_right][y] = hook_num;
                for (int x = x_left; x <= x_right; ++x)
                    grid[x][y_top] = hook_num;
                --x_right; --y_top;
                break;
            case 3: // Bottom right corner
                for (int y = y_bottom; y <= y_top; ++y)
                    grid[x_right][y] = hook_num;
                for (int x = x_left; x <= x_right; ++x)
                    grid[x][y_bottom] = hook_num;
                --x_right; ++y_bottom;
                break;
        }
    }
}

// Function to check initial constraints
bool check_initial_constraints(const int grid[GRID_SIZE][GRID_SIZE]) {
    return grid[4][0] + grid[4][1] == 15 &&
           grid[4][7] + grid[4][8] == 15 &&
           grid[7][2] + grid[8][2] == 15 &&
           grid[4][2] == 5 &&
           grid[4][6] == 4;
}

// Function to check if the filled squares form a connected region
bool is_connected(const int filled_grid[GRID_SIZE][GRID_SIZE]) {
    std::vector<std::vector<bool>> visited(GRID_SIZE, std::vector<bool>(GRID_SIZE, false));
    std::queue<Cell> q;
    int count = 0;

    // Find the first non-zero cell
    for (int i = 0; i < GRID_SIZE; ++i) {
        for (int j = 0; j < GRID_SIZE; ++j) {
            if (filled_grid[i][j] != 0) {
                q.push(Cell(i, j));
                visited[i][j] = true;
                ++count;
                goto bfs;
            }
        }
    }

bfs:
    while (!q.empty()) {
        Cell current = q.front();
        q.pop();

        const int dx[] = {-1, 1, 0, 0};
        const int dy[] = {0, 0, -1, 1};

        for (int dir = 0; dir < 4; ++dir) {
            int nx = current.x + dx[dir];
            int ny = current.y + dy[dir];

            if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE &&
                !visited[nx][ny] && filled_grid[nx][ny] != 0) {
                q.push(Cell(nx, ny));
                visited[nx][ny] = true;
                ++count;
            }
        }
    }

    return count == 45; // 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9
}

// Function to check if the sum of values in each region is equal to TARGET_SUM
bool check_regions(const int filled_grid[GRID_SIZE][GRID_SIZE]) {
    for (const auto& region : regions) {
        int sum = 0;
        for (const auto& cell : region) {
            sum += filled_grid[cell.x][cell.y];
        }
        if (sum != TARGET_SUM) {
            return false;
        }
    }
    return true;
}

// Function to check if every 2x2 region has at least one empty cell
bool check_2x2_regions(const int filled_grid[GRID_SIZE][GRID_SIZE]) {
    for (int i = 0; i < GRID_SIZE - 1; ++i) {
        for (int j = 0; j < GRID_SIZE - 1; ++j) {
            if (filled_grid[i][j] != 0 && filled_grid[i+1][j] != 0 &&
                filled_grid[i][j+1] != 0 && filled_grid[i+1][j+1] != 0) {
                return false;
            }
        }
    }
    return true;
}

// Recursive function to solve the grid
bool solve_grid(int grid[GRID_SIZE][GRID_SIZE],
                std::vector<int> hook_remaining[NUM_HOOKS + 1],
                int filled_grid[GRID_SIZE][GRID_SIZE],
                int cell) {
    if (cell >= TOTAL_CELLS) {
        return is_connected(filled_grid) && check_regions(filled_grid);
    }

    int x = cell % GRID_SIZE;
    int y = cell / GRID_SIZE;

    if (filled_grid[x][y] != -1) {
        return solve_grid(grid, hook_remaining, filled_grid, cell + 1);
    }

    int hook_num = grid[x][y];
    auto& remaining = hook_remaining[hook_num];

    for (size_t i = 0; i < remaining.size(); ++i) {
        int num = remaining[i];
        filled_grid[x][y] = num;
        remaining.erase(remaining.begin() + i);

        if (check_2x2_regions(filled_grid) &&
            solve_grid(grid, hook_remaining, filled_grid, cell + 1)) {
            return true;
        }

        // Backtrack
        filled_grid[x][y] = -1;
        remaining.insert(remaining.begin() + i, num);
    }

    return false;
}

// Function to calculate the product of empty regions
int calculate_product_of_empty_regions(const int filled_grid[GRID_SIZE][GRID_SIZE]) {
    std::vector<std::vector<bool>> visited(GRID_SIZE, std::vector<bool>(GRID_SIZE, false));
    int product = 1;

    for (int i = 0; i < GRID_SIZE; ++i) {
        for (int j = 0; j < GRID_SIZE; ++j) {
            if (filled_grid[i][j] == 0 && !visited[i][j]) {
                int size = 0;
                std::queue<Cell> q;
                q.push(Cell(i, j));
                visited[i][j] = true;

                while (!q.empty()) {
                    Cell current = q.front();
                    q.pop();
                    ++size;

                    const int dx[] = {-1, 1, 0, 0};
                    const int dy[] = {0, 0, -1, 1};

                    for (int dir = 0; dir < 4; ++dir) {
                        int nx = current.x + dx[dir];
                        int ny = current.y + dy[dir];

                        if (nx >= 0 && nx < GRID_SIZE && ny >= 0 && ny < GRID_SIZE &&
                            !visited[nx][ny] && filled_grid[nx][ny] == 0) {
                            q.push(Cell(nx, ny));
                            visited[nx][ny] = true;
                        }
                    }
                }

                if (size > 1) {
                    product *= size;
                }
            }
        }
    }

    return product;
}

int main() {
    // Generate all possible hook placements
    int total_positions = 1 << (2 * 8); // 4^8
    std::vector<std::array<int, 8>> valid_positions;

    for (int i = 0; i < total_positions; ++i) {
        std::array<int, 8> positions;
        int temp = i;
        for (int j = 0; j < 8; ++j) {
            positions[j] = temp & 3;
            temp >>= 2;
        }
        int grid[GRID_SIZE][GRID_SIZE] = {0};
        place_hooks(positions, grid);
        if (check_initial_constraints(grid)) {
            valid_positions.push_back(positions);
        }
    }

    std::cout << "Valid positions: " << valid_positions.size() << std::endl;

    // Try to solve each valid grid
    for (const auto& positions : valid_positions) {
        int grid[GRID_SIZE][GRID_SIZE] = {0};
        place_hooks(positions, grid);
        int filled_grid[GRID_SIZE][GRID_SIZE];
        std::fill(&filled_grid[0][0], &filled_grid[0][0] + GRID_SIZE * GRID_SIZE, -1);

        // Set initial known values
        filled_grid[4][2] = 5;
        filled_grid[4][6] = 4;

        // Initialize hook_remaining
        std::vector<int> hook_remaining[NUM_HOOKS + 1];
        for (int i = 1; i <= NUM_HOOKS; ++i)
            hook_remaining[i] = hook_numbers[i];

        // Remove the initial values from hook_remaining
        hook_remaining[grid[4][2]].erase(std::remove(hook_remaining[grid[4][2]].begin(),
                                                     hook_remaining[grid[4][2]].end(), 5),
                                         hook_remaining[grid[4][2]].end());
        hook_remaining[grid[4][6]].erase(std::remove(hook_remaining[grid[4][6]].begin(),
                                                     hook_remaining[grid[4][6]].end(), 4),
                                         hook_remaining[grid[4][6]].end());

        if (solve_grid(grid, hook_remaining, filled_grid, 0)) {
            std::cout << "Solution found!" << std::endl;
            for (int i = 0; i < GRID_SIZE; ++i) {
                for (int j = 0; j < GRID_SIZE; ++j) {
                    std::cout << filled_grid[i][j] << " ";
                }
                std::cout << std::endl;
            }
            int result = calculate_product_of_empty_regions(filled_grid);
            std::cout << "The answer is: " << result << std::endl;
            return 0;
        }
    }

    std::cout << "No solution found." << std::endl;
    return 0;
}
