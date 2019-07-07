/* Code for the Jacbi equation solver. 
 * Author: Naga Kandasamy
 * Date created: April 19, 2019
 * Date modified: April 20, 2019
 *
 * Compile as follows:
 * gcc -o solver solver.c solver_gold.c -O3 -Wall -std=c99 -lm -lpthread -D_XOPEN_SOURCE=600 (on xunil)
 *
 * If you wish to see debug info add the -D DEBUG option when compiling the code.
 */

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include "grid.h" 
#include "barrier_MACOS.h" /* Comment out when testing on xunil */

/* Structure data type of p_threads arg */
typedef struct args_thread {
    int threadID;
	int num_threads;
    grid_t *grid1;
	grid_t *grid2;
	double *mydiff;
    
} ARGS_FOR_THREAD;

int pthread_iters = 0;
int pthread_done = 0;

extern int compute_gold (grid_t *);
int compute_using_pthreads_jacobi (grid_t *, int);
void compute_grid_differences(grid_t *, grid_t *);
grid_t *create_grid (int, float, float);
grid_t *copy_grid (grid_t *);
void print_grid (grid_t *);
void print_stats (grid_t *);
double grid_mse (grid_t *, grid_t *);
void * pthreads_jacobi(void *);

pthread_barrier_t b1;
pthread_barrier_t b2;
pthread_barrier_t b3;

int 
main (int argc, char **argv)
{	
	struct timeval start, stop;	

	if (argc < 5) {
        printf ("Usage: %s grid-dimension num-threads min-temp max-temp\n", argv[0]);
        printf ("grid-dimension: The dimension of the grid\n");
        printf ("num-threads: Number of threads\n"); 
        printf ("min-temp, max-temp: Heat applied to the north side of the plate is uniformly distributed between min-temp and max-temp\n");
        exit (EXIT_FAILURE);
    }
    
    /* Parse command-line arguments. */
    int dim = atoi (argv[1]);
    int num_threads = atoi (argv[2]);
    float min_temp = atof (argv[3]);
    float max_temp = atof (argv[4]);
    
    /* Generate the grids and populate them with initial conditions. */
 	grid_t *grid_1 = create_grid (dim, min_temp, max_temp);
    /* Grid 2 should have the same initial conditions as Grid 1. */
    grid_t *grid_2 = copy_grid (grid_1); 

	/* Compute the reference solution using the single-threaded version. */
	printf ("\nUsing the single threaded version to solve the grid\n");
	gettimeofday(&start, NULL);
	int num_iter = compute_gold (grid_1);
	gettimeofday(&stop, NULL);
	printf ("Convergence achieved after %d iterations\n", num_iter);
    /* Print key statistics for the converged values. */
	printf ("Printing statistics for the interior grid points\n");
    print_stats (grid_1);
	printf ("Execution time for single threaded = %fs. \n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));

#ifdef DEBUG
    print_grid (grid_1);
#endif
	
	/* Use pthreads to solve the equation using the jacobi method. */
	printf ("\nUsing pthreads to solve the grid using the jacobi method\n");
	gettimeofday(&start, NULL);
	num_iter = compute_using_pthreads_jacobi (grid_2, num_threads);
	gettimeofday(&stop, NULL);
	printf ("Convergence achieved after %d iterations\n", num_iter);			
    printf ("Printing statistics for the interior grid points\n");
	print_stats (grid_2);
	printf ("Execution time for mult-threaded = %fs. \n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));

#ifdef DEBUG
    print_grid (grid_2);
#endif
    
    /* Compute grid differences. */
    double mse = grid_mse (grid_1, grid_2);
    printf ("MSE between the two grids: %f\n", mse);

	/* Free up the grid data structures. */
	free ((void *) grid_1->element);	
	free ((void *) grid_1); 
	free ((void *) grid_2->element);	
	free ((void *) grid_2);

	exit (EXIT_SUCCESS);
}

int 
compute_using_pthreads_jacobi (grid_t *grid, int num_threads)
{		
    int i;
	ARGS_FOR_THREAD *args_thread;
	double diffs[num_threads];

	grid_t *grid2 = copy_grid(grid);

	pthread_t *worker = (pthread_t *) malloc (num_threads *sizeof (pthread_t));
	pthread_barrier_init(&b1, 0, num_threads);
	pthread_barrier_init(&b2, 0, num_threads);
	pthread_barrier_init(&b3, 0, num_threads);

	for (i = 0; i < num_threads; i++) {
		args_thread = (ARGS_FOR_THREAD *) malloc (sizeof (ARGS_FOR_THREAD));

		args_thread->threadID = i;
		args_thread->num_threads = num_threads;
		args_thread->grid1 = grid;
		args_thread->grid2 = grid2;
		args_thread->mydiff = diffs + i;

		if ((pthread_create (&worker[i], NULL, pthreads_jacobi, (void *) args_thread)) != 0) {
            printf ("Cannot create worker thread \n");
            exit (EXIT_FAILURE);
        }
	}

	for (i = 0; i < num_threads; i++) {
		pthread_join (worker[i], NULL);
	}

	free ((void *) grid2->element);	
	free ((void *) grid2);

	pthread_barrier_destroy(&b1);
    pthread_barrier_destroy(&b2);
	pthread_barrier_destroy(&b3);

	return pthread_iters;
}

/* PThreads worker program for Jacobi Solver */
void *
pthreads_jacobi (void *args) {

    int i, j;
	double diff; 
    float eps = 1e-2; /* Convergence criteria. */

	ARGS_FOR_THREAD *grid_thread = (ARGS_FOR_THREAD *) args;

    int threadID = grid_thread->threadID;
	int num_threads = grid_thread->num_threads;
	grid_t *grid1 = grid_thread->grid1;
	grid_t *grid2 = grid_thread->grid2;

	int num_elements = (grid1->dim -2) * (grid1->dim -2);
	
	while(!pthread_done) { /* While we have not converged yet. */
		diff = 0.0;

		// First wave, update grid2 (new values)
        for (i = threadID+1; i < (grid1->dim - 1); i += num_threads) {
            for (j = 1; j < (grid1->dim - 1); j++) {
                grid2->element[i * grid1->dim + j] = 0.25 * (grid1->element[(i - 1) * grid1->dim + j] +\
                              								grid1->element[(i + 1) * grid1->dim + j] +\
                              								grid1->element[i * grid1->dim + (j + 1)] +\
                              								grid1->element[i * grid1->dim + (j - 1)]);
            }
        }

		pthread_barrier_wait(&b1);

		//Second wave - update grid1 (grid) with new values from grid2 (backup)
		for (i = threadID+1; i < (grid1->dim - 1); i += num_threads) {
            for (j = 1; j < (grid1->dim - 1); j++) {

				// Get diff first
				diff = diff + fabs(grid2->element[i * grid1->dim + j] - grid1->element[i * grid1->dim + j]);
                grid1->element[i * grid1->dim + j] = grid2->element[i * grid1->dim + j];
			}
        }

		*grid_thread->mydiff = diff;

		pthread_barrier_wait(&b2);

		// First thread will check for covarence
		if (threadID == 0) {
			double *temp;
			diff = 0.0;

			for (i = 0; i < num_threads; i++) {
				temp = grid_thread->mydiff + i;
				diff = diff + *temp;			
			}

			diff = diff/num_elements;

			if (diff < eps)
				pthread_done = 1;

			//printf ("Iteration %d. DIFF: %f.\n", pthread_iters, diff);
			pthread_iters++;
		}
		pthread_barrier_wait(&b3);
	}
	
	free ((void *) grid_thread);
    pthread_exit (NULL);

}

/* Create a grid with the specified initial conditions. */
grid_t * 
create_grid (int dim, float min, float max)
{
    grid_t *grid = (grid_t *) malloc (sizeof (grid_t));
    if (grid == NULL)
        return NULL;

    grid->dim = dim;
	printf("Creating a grid of dimension %d x %d\n", grid->dim, grid->dim);
	grid->element = (float *) malloc (sizeof (float) * grid->dim * grid->dim);
    if (grid->element == NULL)
        return NULL;

    int i, j;
	for (i = 0; i < grid->dim; i++) {
		for (j = 0; j < grid->dim; j++) {
            grid->element[i * grid->dim + j] = 0.0; 			
		}
    }

    /* Initialize the north side, that is row 0, with temperature values. */ 
    srand ((unsigned) time (NULL));
	float val;		
    for (j = 1; j < (grid->dim - 1); j++) {
        val =  min + (max - min) * rand ()/(float)RAND_MAX;
        grid->element[j] = val; 	
    }

    return grid;
}

/* Creates a new grid and copies over the contents of an existing grid into it. */
grid_t *
copy_grid (grid_t *grid) 
{
    grid_t *new_grid = (grid_t *) malloc (sizeof (grid_t));
    if (new_grid == NULL)
        return NULL;

    new_grid->dim = grid->dim;
	new_grid->element = (float *) malloc (sizeof (float) * new_grid->dim * new_grid->dim);
    if (new_grid->element == NULL)
        return NULL;

    int i, j;
	for (i = 0; i < new_grid->dim; i++) {
		for (j = 0; j < new_grid->dim; j++) {
            new_grid->element[i * new_grid->dim + j] = grid->element[i * new_grid->dim + j] ; 			
		}
    }

    return new_grid;
}

/* This function prints the grid on the screen. */
void 
print_grid (grid_t *grid)
{
    int i, j;
    for (i = 0; i < grid->dim; i++) {
        for (j = 0; j < grid->dim; j++) {
            printf ("%f\t", grid->element[i * grid->dim + j]);
        }
        printf ("\n");
    }
    printf ("\n");
}


/* Print out statistics for the converged values of the interior grid points, including min, max, and average. */
void 
print_stats (grid_t *grid)
{
    float min = INFINITY;
    float max = 0.0;
    double sum = 0.0;
    int num_elem = 0;
    int i, j;

    for (i = 1; i < (grid->dim - 1); i++) {
        for (j = 1; j < (grid->dim - 1); j++) {
            sum += grid->element[i * grid->dim + j];

            if (grid->element[i * grid->dim + j] > max) 
                max = grid->element[i * grid->dim + j];

             if(grid->element[i * grid->dim + j] < min) 
                min = grid->element[i * grid->dim + j];
             
             num_elem++;
        }
    }
                    
    printf("AVG: %f\n", sum/num_elem);
	printf("MIN: %f\n", min);
	printf("MAX: %f\n", max);
	printf("\n");
}

/* Calculate the mean squared error between elements of two grids. */
double
grid_mse (grid_t *grid_1, grid_t *grid_2)
{
    double mse = 0.0;
    int num_elem = grid_1->dim * grid_1->dim;
    int i;

    for (i = 0; i < num_elem; i++) 
        mse += (grid_1->element[i] - grid_2->element[i]) * (grid_1->element[i] - grid_2->element[i]);
                   
    return mse/num_elem; 
}



		

