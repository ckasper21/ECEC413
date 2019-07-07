README
Last Updated: 5/5/19
Chris Kasper
ECEC 413
Jacobi Solver OpenMP

Build: gcc -fopenmp -o solver solver.c solver_gold.c -O3 -Wall -std=c99 -lm (on xunil)
		gcc-8 -fopenmp -o solver solver.c solver_gold.c -O3 -Wall -std=c99 -lm (on Mac)
Execute: ./solver GRID_SIZE NUM_THREADS MIN_TEMP MAX_TEMP