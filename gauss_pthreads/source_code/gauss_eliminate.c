/* Gaussian elimination code.
 * 
 * Author: Naga Kandasamy
 * Date created: February 7
 * Date of last update: April 10, 2019
 *
 * Compile as follows: gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -O3 -Wall -std=c99 -lpthread -lm -D_XOPEN_SOURCE=600 (on xunil)
 *                     gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -O3 -Wall -std=c99 -lpthread -lm (on MacOSX)
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "gauss_eliminate.h"
#include "barrier_MACOS.h" /* Comment out when testing on xunil */

#define MIN_NUMBER 2
#define MAX_NUMBER 50
#define NUM_THREADS 32

pthread_barrier_t divBarrier;
pthread_barrier_t elimBarrier;

/* Function prototypes. */
extern int compute_gold (float *, unsigned int);
extern void print_matrix(float *, unsigned int);

Matrix allocate_matrix (int, int, int);
void gauss_eliminate_using_pthreads (Matrix);
int perform_simple_check (const Matrix);
float get_random_number (int, int);
int check_results (float *, float *, unsigned int, float);
void * compute_gold_pthread (void *); 

int
main (int argc, char **argv)
{
    /* Check command line arguments. */
    if (argc > 1) {
        printf ("Error. This program accepts no arguments.\n");
        exit (EXIT_FAILURE);
    }

    Matrix A;			    /* Input matrix. */
    Matrix U_reference;		/* Upper triangular matrix computed by reference code. */
    Matrix U_mt;			/* Upper triangular matrix computed by pthreads. */

    /* Initialize the random number generator with a seed value. */
    srand (time (NULL));

    A = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 1);	            /* Allocate and populate a random square matrix. */
    U_reference = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 0);	/* Allocate space for the reference result. */
    U_mt = allocate_matrix (MATRIX_SIZE, MATRIX_SIZE, 0);	        /* Allocate space for the multi-threaded result. */

    /* Copy the contents of the A matrix into the U matrices. */
    for (int i = 0; i < A.num_rows; i++) {
        for (int j = 0; j < A.num_rows; j++) {
            U_reference.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
            U_mt.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
        }
    }

    printf ("Performing gaussian elimination using the reference code.\n");
    struct timeval start, stop;
    gettimeofday (&start, NULL);
    
    int status = compute_gold (U_reference.elements, A.num_rows);
  
    gettimeofday (&stop, NULL);
    printf ("CPU run time (Single thread) = %0.2f s.\n", (float) (stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec) / (float) 1000000));

    if (status == 0) {
        printf ("Failed to convert given matrix to upper triangular. Try again. Exiting.\n");
        exit (EXIT_FAILURE);
    }
  
    status = perform_simple_check (U_reference);	/* Check that the principal diagonal elements are 1. */ 
    if (status == 0) {
        printf ("The upper triangular matrix is incorrect. Exiting.\n");
        exit (EXIT_FAILURE);
    }
    printf ("Single-threaded Gaussian elimination was successful.\n");
  
    gettimeofday (&start, NULL);

    /* Perform the Gaussian elimination using pthreads. The resulting upper 
     * triangular matrix should be returned in U_mt */
    gauss_eliminate_using_pthreads (U_mt);

    gettimeofday (&stop, NULL);

    /* check if the pthread result matches the reference solution within a specified tolerance. */
    int size = MATRIX_SIZE * MATRIX_SIZE;
    int res = check_results (U_reference.elements, U_mt.elements, size, 0.0001f);
    printf ("Pthread TEST %s\n", (1 == res) ? "PASSED" : "FAILED");
    printf ("CPU run time (Multi-thread) = %0.2f s.\n", (float) (stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec) / (float) 1000000));

    /* Free memory allocated for the matrices. */
    free (A.elements);
    free (U_reference.elements);
    free (U_mt.elements);

    exit (EXIT_SUCCESS);
}

void
gauss_eliminate_using_pthreads (Matrix U)
{
    int i;
    ARGS_FOR_THREAD *args_thread;

    pthread_t *worker = (pthread_t *) malloc (NUM_THREADS * sizeof (pthread_t));
    pthread_barrier_init(&divBarrier, 0, NUM_THREADS);
    pthread_barrier_init(&elimBarrier, 0, NUM_THREADS);

    for (i = 0; i < NUM_THREADS; i++) {
        args_thread = (ARGS_FOR_THREAD *) malloc (sizeof (ARGS_FOR_THREAD));

        args_thread->threadID = i;
        args_thread->U = &U;

        if ((pthread_create (&worker[i], NULL, compute_gold_pthread, (void *) args_thread)) != 0) {
            printf ("Cannot create worker thread \n");
            exit (EXIT_FAILURE);
        }
    }

    for (i = 0; i < NUM_THREADS; i++) {
        pthread_join (worker[i], NULL);
    }

    pthread_barrier_destroy(&divBarrier);
    pthread_barrier_destroy(&elimBarrier);

    return;
}

/* Perform Gaussian elimination in place on the U matrix using pthreads */
void *
compute_gold_pthread (void *args) 
{
    int i, j, k;
    int num_elements;
    float *U;

    ARGS_FOR_THREAD *matrix_thread = (ARGS_FOR_THREAD *) args;

    num_elements = matrix_thread->U->num_rows;
    U = matrix_thread->U->elements;
    int threadID = matrix_thread->threadID;

    for (k = 0; k < num_elements; k++) {
        for (j = (k + threadID + 1); j < num_elements; j+= NUM_THREADS) {
            U[num_elements * k + j] = (float) (U[num_elements * k + j] / U[num_elements * k + k]);  /* Division step. */
        }
        pthread_barrier_wait(&divBarrier);

        for (i = (threadID + k + 1); i < num_elements; i+= NUM_THREADS) {
            for (j = (k + 1); j < num_elements; j++) {
                U[num_elements * i + j] = U[num_elements * i + j] - (U[num_elements * i + k] * U[num_elements * k + j]);	/* Elimination step. */
            }
            U[num_elements * i + k] = 0;
        }
        pthread_barrier_wait(&elimBarrier);
    }

    // Each thread has rows to set the principal diagonal entry
    for (k = 0 + threadID; k < num_elements; k += NUM_THREADS) {
         U[num_elements * k + k] = 1;   /* Set the principal diagonal entry in U to be 1. */ 

    }
    
    free ((void *) matrix_thread);
    pthread_exit (NULL);
}


/* Function checks if the results generated by the single threaded and multi threaded versions match. */
int
check_results (float *A, float *B, unsigned int size, float tolerance)
{
    for (int i = 0; i < size; i++)
        if (fabsf (A[i] - B[i]) > tolerance)
            return 0;
    return 1;
}


/* Allocate a matrix of dimensions height*width
 * If init == 0, initialize to all zeroes.  
 * If init == 1, perform random initialization. 
*/
Matrix
allocate_matrix (int num_rows, int num_columns, int init)
{
    Matrix M;
    M.num_columns = M.pitch = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
    M.elements = (float *) malloc (size * sizeof (float));
  
    for (unsigned int i = 0; i < size; i++) {
        if (init == 0)
            M.elements[i] = 0;
        else
            M.elements[i] = get_random_number (MIN_NUMBER, MAX_NUMBER);
    }
  
    return M;
}


/* Returns a random floating-point number between the specified min and max values. */ 
float
get_random_number (int min, int max)
{
    return (float)
        floor ((double)
                (min + (max - min + 1) * ((float) rand () / (float) RAND_MAX)));
}

/* Performs a simple check on the upper triangular matrix. Checks to see if the principal diagonal elements are 1. */
int
perform_simple_check (const Matrix M)
{
    for (unsigned int i = 0; i < M.num_rows; i++)
        if ((fabs (M.elements[M.num_rows * i + i] - 1.0)) > 0.0001)
            return 0;
  
    return 1;
}
