/* Host-side code to perform counting sort 
 * Author: Naga Kandasamy
 * Date modified: May 19, 2019
 * 
 * Compile as follows: make clean && make
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include "counting_sort.h"
#define HISTOGRAM_SIZE 256

#include "counting_sort_kernel.cu"

/* Do not change the range value. */
#define MIN_VALUE 0 
#define MAX_VALUE 255
//#define DEBUG

#define THREAD_BLOCK_SIZE 256
#define NUM_BLOCKS 40

extern "C" int counting_sort_gold (int *, int *, int, int);
int rand_int (int, int);
void print_array (int *, int);
void print_min_and_max_in_array (int *, int);
void compute_on_device (int *, int *, int, int);
int check_if_sorted (int *, int);
int compare_results (int *, int *, int);
void check_for_error (const char *);

int 
main (int argc, char **argv)
{
    if (argc < 2) {
        printf ("Usage: %s num-elements\n", argv[0]);
        exit (EXIT_FAILURE);
    }

	struct timeval start, stop;
    int num_elements = atoi (argv[1]);
    int range = MAX_VALUE - MIN_VALUE;
    int *input_array, *sorted_array_reference, *sorted_array_d;

    /* Populate the input array with random integers between [0, RANGE]. */
    printf ("Generating input array with %d elements in the range 0 to %d\n", num_elements, range);
    input_array = (int *) malloc (num_elements * sizeof (int));
    if (input_array == NULL) {
        printf ("Cannot malloc memory for the input array. \n");
        exit (EXIT_FAILURE);
    }
    srand (time (NULL));
    for (int i = 0; i < num_elements; i++)
        input_array[i] = rand_int (MIN_VALUE, MAX_VALUE);

#ifdef DEBUG
    print_array (input_array, num_elements);
    print_min_and_max_in_array (input_array, num_elements);
#endif

    /* Sort the elements in the input array using the reference implementation. 
     * The result is placed in sorted_array_reference. */
    printf ("\nSorting array on CPU\n");
    int status;
    sorted_array_reference = (int *) malloc (num_elements * sizeof (int));
    if (sorted_array_reference == NULL) {
        perror ("Malloc"); 
        exit (EXIT_FAILURE);
    }
    
	memset (sorted_array_reference, 0, num_elements);
	gettimeofday (&start, NULL);
    
	status = counting_sort_gold (input_array, sorted_array_reference, num_elements, range);
    if (status == 0) {
        exit (EXIT_FAILURE);
    }

	gettimeofday (&stop, NULL);
    printf("Execution time = %f\n",stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
	
    status = check_if_sorted (sorted_array_reference, num_elements);
    if (status == 0) {
        printf ("Error sorting the input array using the reference code\n");
        exit (EXIT_FAILURE);
    }

    printf ("Counting sort was successful on the CPU\n");

#ifdef DEBUG
    print_array (sorted_array_reference, num_elements);
#endif

    printf ("\nSorting array on GPU\n");
    sorted_array_d = (int *) malloc (num_elements * sizeof (int));
    if (sorted_array_d == NULL) {
        perror ("Malloc");
        exit (EXIT_FAILURE);
    }
    memset (sorted_array_d, 0, num_elements);
    compute_on_device (input_array, sorted_array_d, num_elements, range);

    /* Check the two results for correctness. */
    printf ("\nComparing CPU and GPU results\n");
    status = compare_results (sorted_array_reference, sorted_array_d, num_elements);
    if (status == 1)
        printf ("Test passed\n");
    else
        printf ("Test  failed\n");

    exit(EXIT_SUCCESS);
}


/* GPU implementation of counting sort. */
void 
compute_on_device (int *input_array, int *sorted_array, int num_elements, int range)
{
	struct timeval start, stop;	

	int *d_input = NULL;
	int *d_output = NULL;
	
	/* Histogram step variables */
	int *hist = (int *) malloc (sizeof (int) * HISTOGRAM_SIZE); 
	memset (hist, 0, HISTOGRAM_SIZE);
	
	int *d_hist = NULL;

	/* Prefix scan variables */
	int *scan_out = (int *) malloc (sizeof (int) * HISTOGRAM_SIZE); 
    memset (scan_out, 0, sizeof (int) * HISTOGRAM_SIZE); 

	int *d_scan = NULL;	

	/* Allocate space on GPU for the input */
	cudaMalloc((void**) &d_input, num_elements * sizeof(int));
	cudaMemcpy(d_input, input_array, num_elements * sizeof(int), cudaMemcpyHostToDevice);

	/* Allocate space on GPU for the output */
	cudaMalloc((void**) &d_output, num_elements * sizeof(int));
    cudaMemcpy(d_output, sorted_array, num_elements * sizeof(int), cudaMemcpyHostToDevice);

	/* Allocate space on GPU for histogram */
	cudaMalloc((void**) &d_hist, HISTOGRAM_SIZE * sizeof(int));
	cudaMemcpy(d_hist, hist, HISTOGRAM_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	
	/* Allocate space on GPU for scan output */
	cudaMalloc((void**) &d_scan, HISTOGRAM_SIZE * sizeof(int));
    cudaMemcpy(d_scan, scan_out, HISTOGRAM_SIZE * sizeof(int), cudaMemcpyHostToDevice);	

	/* Kernel config */
	dim3 thread_block (THREAD_BLOCK_SIZE, 1, 1);
	dim3 grid (NUM_BLOCKS, 1);

    gettimeofday (&start, NULL);
    
	/* Histogram generation */
	histogram_kernel<<<grid, thread_block>>>(d_input, d_hist, num_elements, HISTOGRAM_SIZE); 
	cudaDeviceSynchronize ();
	check_for_error ("KERNEL FAILURE: Histogram");

	/* Kernel config for scan */
	grid.x = 1;
	thread_block.x = HISTOGRAM_SIZE;
	
	int shared_mem_size = sizeof (int) * HISTOGRAM_SIZE;

	/* Inclusive prefix scan */
	scan_kernel<<< grid, thread_block, 2 * shared_mem_size >>>(d_scan, d_hist, HISTOGRAM_SIZE);
	cudaDeviceSynchronize ();
    check_for_error ("KERNEL FAILURE: Prefix scan");	

	/* Kernel config for count sort */
	grid.x = NUM_BLOCKS;
	thread_block.x = THREAD_BLOCK_SIZE;

	/* Count sort */
	counting_sort_kernel<<<grid, thread_block>>>(d_output, d_scan, HISTOGRAM_SIZE);
	cudaDeviceSynchronize ();
    check_for_error ("KERNEL FAILURE: Count sort");
	gettimeofday(&stop, NULL);

	/* Copy the result back from the GPU and store. */
    cudaMemcpy (sorted_array, d_output, num_elements * sizeof (int), cudaMemcpyDeviceToHost);

    printf("Execution time = %f\n",stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);

	cudaFree(d_input);
	cudaFree(d_hist);
	cudaFree(d_scan);
	cudaFree(d_output);

	free(hist);
	free(scan_out);

	return;
}

/* Check if the array is sorted. */
int
check_if_sorted (int *array, int num_elements)
{
    int status = 1;
    for (int i = 1; i < num_elements; i++) {
        if (array[i - 1] > array[i]) {
            status = 0;
            break;
        }
    }

    return status;
}

/* Check if the arrays elements are identical. */ 
int 
compare_results (int *array_1, int *array_2, int num_elements)
{
    int status = 1;
    for (int i = 0; i < num_elements; i++) {
        if (array_1[i] != array_2[i]) {
            status = 0;
            printf("Error at index %d\n", i);
			break;
        }
    }

    return status;
}


/* Returns a random integer between [min, max]. */ 
int
rand_int (int min, int max)
{
    float r = rand ()/(float) RAND_MAX;
    return (int) floorf (min + (max - min) * r);
}

/* Helper function to print the given array. */
void
print_array (int *this_array, int num_elements)
{
    printf ("Array: ");
    for (int i = 0; i < num_elements; i++)
        printf ("%d ", this_array[i]);
    printf ("\n");
    return;
}

/* Helper function to return the min and max values in the given array. */
void 
print_min_and_max_in_array (int *this_array, int num_elements)
{
    int i;

    int current_min = INT_MAX;
    for (i = 0; i < num_elements; i++)
        if (this_array[i] < current_min)
            current_min = this_array[i];

    int current_max = INT_MIN;
    for (i = 0; i < num_elements; i++)
        if (this_array[i] > current_max)
            current_max = this_array[i];

    printf ("Minimum value in the array = %d\n", current_min);
    printf ("Maximum value in the array = %d\n", current_max);
    return;
}

/* Check for errors during kernel execution. */
void 
check_for_error (const char *msg)
{
	cudaError_t err = cudaGetLastError ();
	if (cudaSuccess != err) {
		printf ("CUDA ERROR: %s (%s)\n", msg, cudaGetErrorString(err));
		exit (EXIT_FAILURE);
	}

    return;
} 

