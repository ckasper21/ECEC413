/*  Purpose: Calculate definite integral using trapezoidal rule.
 *
 * Input:   a, b, n, num_threads
 * Output:  Estimate of integral from a to b of f(x)
 *          using n trapezoids, with num_threads.
 *
 * Compile: gcc -o trap trap.c -O3 -std=c99 -Wall -lpthread -lm
 * Usage:   ./trap
 *
 * Note:    The function f(x) is hardwired.
 *
 * Author: Naga Kandasamy
 * Date modified: 4/1/2019
 *
 */

#ifdef _WIN32
#  define NOMINMAX 
#endif

#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

double compute_using_pthreads (float, float, int, float, int);
double compute_gold (float, float, int, float);
void * compute_gold_pthread (void *); 

/* Structure data type of p_threads arg */
typedef struct args_thread {
    int threadID;
    int numThreads;
    float a;
    float b;
    int numTrapz;
    float h;
    double *result;
} ARGS_FOR_THREAD;

int 
main (int argc, char **argv) 
{
    struct timeval start, stop;	

    if (argc < 5) {
        printf ("Usage: trap lower-limit upper-limit num-trapezoids num-threads\n");
        printf ("lower-limit: The lower limit for the integral\n");
        printf ("upper-limit: The upper limit for the integral\n");
        printf ("num-trapezoids: Number of trapeziods used to approximate the area under the curve\n");
        printf ("num-threads: Number of threads to use in the calculation\n");
        exit (EXIT_FAILURE);
    }

    float a = atoi (argv[1]); /* Lower limit */
	float b = atof (argv[2]); /* Upper limit */
	float n = atof (argv[3]); /* Number of trapezoids */

	float h = (b - a)/(float) n; /* Base of each trapezoid */  
	printf ("The base of the trapezoid is %f \n", h);

    gettimeofday (&start, NULL);
	double reference = compute_gold (a, b, n, h);
    printf ("Reference solution computed using single-threaded version = %f \n", reference);
    gettimeofday (&stop, NULL);
	printf ("Execution time for single threaded = %fs. \n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));

	/* Write this function to complete the trapezoidal rule using pthreads. */
    int num_threads = atoi (argv[4]); /* Number of threads */
	gettimeofday (&start, NULL);
    double pthread_result = compute_using_pthreads (a, b, n, h, num_threads);
	printf ("Solution computed using %d threads = %f \n", num_threads, pthread_result);
    gettimeofday (&stop, NULL);
	printf ("Execution time for multi-threaded = %fs. \n", (float)(stop.tv_sec - start.tv_sec +\
                (stop.tv_usec - start.tv_usec)/(float)1000000));

    exit(EXIT_SUCCESS);
} 


/*------------------------------------------------------------------
 * Function:    f
 * Purpose:     Defines the integrand
 * Input args:  x
 * Output: sqrt((1 + x^2)/(1 + x^4))

 */
float 
f (float x) 
{
    return sqrt ((1 + x*x)/(1 + x*x*x*x));
}

/*------------------------------------------------------------------
 * Function:    compute_gold
 * Purpose:     Estimate integral from a to b of f using trap rule and
 *              n trapezoids using a single-threaded version
 * Input args:  a, b, n, h
 * Return val:  Estimate of the integral 
 */
double 
compute_gold (float a, float b, int n, float h) 
{
   double integral;
   int k;

   integral = (f(a) + f(b))/2.0;

   for (k = 1; k <= n-1; k++)
     integral += f(a+k*h);
   
   integral = integral*h;

   return integral;
}  

/*------------------------------------------------------------------
 * Function:    compute_gold
 * Purpose:     Setup to estimate integral from a to b of f using trap rule and
 *              n trapezoids using a pthread version
 * Input args:  a, b, n, h, num_threads
 * Return val:  Estimate of the integral 
 */
double 
compute_using_pthreads (float a, float b, int n, float h, int num_threads)
{
	int i;
    double integral = 0.0;
    double data[num_threads];
    ARGS_FOR_THREAD *args_thread;

    /* Determine starting point for each thread */
    float startA = a;

    pthread_t *worker = (pthread_t *) malloc (num_threads * sizeof (pthread_t));

    /* Split up work based on number of trapezoids equally across threads */
    for (i = 0; i < num_threads; i++) {
        args_thread = (ARGS_FOR_THREAD *) malloc (sizeof (ARGS_FOR_THREAD));
        
        args_thread->threadID = i;
        args_thread->numThreads = num_threads;

        /* Determine number of trapezoids for each thread */
        if (i != num_threads-1) {
            args_thread->numTrapz = n / num_threads;
        } else {
            args_thread->numTrapz = n - (n / num_threads) * i;
        }

        args_thread->a = startA;
        args_thread->b = startA + (args_thread->numTrapz * h);
        args_thread->h = h;   

        args_thread->result = data + i;

        startA = args_thread->b;

        if ((pthread_create (&worker[i], NULL, compute_gold_pthread, (void *) args_thread)) != 0) {
            printf ("Cannot create worker thread \n");
            exit (EXIT_FAILURE);
        }
    }

    for (i = 0; i < num_threads; i++)
        pthread_join (worker[i], NULL);

    for (i = 0; i < num_threads; i++)
        integral += data[i];

    integral = integral * h;

    return integral;
}

/*------------------------------------------------------------------
 * Function:    compute_gold_pthread
 * Purpose:     Estimate integral from a to b of f using trap rule and
 *              n trapezoids using a pthread version
 * Input args:  args
 * Return val:  Estimate of the integral 
 */
void *
compute_gold_pthread (void *args) 
{
    ARGS_FOR_THREAD *args_thread = (ARGS_FOR_THREAD *) args;
    double integral = 0.0;
    int k;

    /* Special case: If there is only 1 thread */ 
    if ((args_thread->threadID == 0) && (args_thread->threadID == (args_thread->numThreads - 1))) {
        integral = (f(args_thread->a) + f(args_thread->b)) / 2.0;

        for (k = 1; k <= args_thread->numTrapz-1; k++) {
            integral += f(args_thread->a+k*args_thread->h);
        }
    
    /* First thread processing */
    } else if (args_thread->threadID == 0) {
        integral = f(args_thread->a) / 2.0;

        for (k = 1; k <= args_thread->numTrapz; k++) {
            integral += f(args_thread->a+k*args_thread->h);
        }

    /* End thread processing */
    } else if (args_thread->threadID == (args_thread->numThreads - 1)) {
        for (k = 1; k <= args_thread->numTrapz-1; k++) {
            integral += f(args_thread->a+k*args_thread->h);
        }
        
        integral += f(args_thread->b) / 2.0;

    /* Middle thread processing */
    } else {
        for (k = 1; k <= args_thread->numTrapz; k++) {
            integral += f(args_thread->a+k*args_thread->h);
        }
    }

    *args_thread->result = integral;

    free ((void *) args_thread);
    pthread_exit (NULL);
}