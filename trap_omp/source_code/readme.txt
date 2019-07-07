README
Last Updated: 5/5/19
Chris Kasper
ECEC 413
trap_omp assignment

Build: gcc -fopenmp -o trap trap.c -O3 -std=c99 -Wall -lm (on xunil)
		gcc-8 -fopenmp -o trap trap.c -O3 -std=c99 -Wall -lm (on Mac)
		
Execute: ./trap LOWER_BOUND UPPER_BOUND NUM_TRAPS NUM_THREADS

Notes:
- None