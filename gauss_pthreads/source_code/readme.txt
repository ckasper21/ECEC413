README
Last Updated: 4/20/19
Chris Kasper
ECEC 413
Programming Assignment 2

Build: gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -O3 -Wall -std=c99 -lpthread -lm -D_XOPEN_SOURCE=600
Execute: ./gauss_eliminate

Notes:
- For testing on ./gauss_eliminate on xunil-05/03, use the above build command.
- If testing on MacOSX, use the other build command:
	- gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -O3 -Wall -std=c99 -lpthread -lm (on MacOSX)
	- you will also need to uncomment line 19
		- this includes a implementation of pthread barriers on MacOSX