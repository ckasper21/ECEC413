README
Last Updated: 4/26/19
Chris Kasper
ECEC 413
Programming Assignment 3

Build: gcc -o solver solver.c solver_gold.c -O3 -Wall -std=c99 -lm -lpthread -D_XOPEN_SOURCE=600
Execute: ./solver GRID_SIZE NUM_THREADS MIN_TEMP MAX_TEMP

Notes:
- For testing ./solver on xunil-05/03, use the above build command.
- If testing on MacOSX, use the other build command:
	- gcc -o solver solver.c solver_gold.c -O3 -Wall -std=c99 -lm -lpthread (on MacOSX)
	- you will also need to uncomment line 20
		- this includes a implementation of pthread barriers on MacOSX