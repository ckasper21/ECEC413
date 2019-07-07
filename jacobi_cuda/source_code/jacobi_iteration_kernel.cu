#include "jacobi_iteration.h"

/* This function uses a compare and swap technique to acquire a mutex/lock. */
__device__ 
void lock (int *mutex) 
{	  
    while (atomicCAS(mutex, 0, 1) != 0);
    return;
}

/* This function uses an atomic exchange operation to release the mutex/lock. */
__device__ 
void unlock (int *mutex) 
{
    atomicExch (mutex, 0);
    return;
}

/* Jacobi iteration using global memory */
__global__ void jacobi_iteration_kernel_naive (const matrix_t A, const matrix_t x, matrix_t x_update, const matrix_t B, int* mutex, double* ssd)
{
	__shared__ double ssd_per_thread[THREAD_BLOCK_SIZE];
	unsigned int i, j;
	unsigned int num_rows = A.num_rows;
	unsigned int num_cols = A.num_columns;
	float new_x;
	double thisSSD = 0.0;

	int threadY = threadIdx.y;
	int blockY = blockIdx.y;

	int row = blockDim.y * blockY + threadY;
	
	if (row < num_rows) {
		/* Initialize Jacobi sum, begin computation */
		double sum = -A.elements[row * num_cols + row] * x.elements[row];
		
		for (j = 0; j < num_cols; j++) {
			sum += (double) A.elements[row * num_cols + j] * x.elements[j];
		}	
		
		/* Find new unknown value */
		new_x = (B.elements[row] - sum) / A.elements[row * num_cols + row];

		__syncthreads();
			
		thisSSD = (double) (new_x - x.elements[row]) * (new_x - x.elements[row]);
		x_update.elements[row] = new_x;	

		ssd_per_thread[threadY] = thisSSD;
    	__syncthreads();

    	/* SSD reduction */
		i = blockDim.y/2;
    	while (i != 0) {
        	if (threadY < i)
            	ssd_per_thread[threadY] += ssd_per_thread[threadY + i];
        	__syncthreads();
        	i /= 2;
    	}

		if (threadY == 0) {
        	lock(mutex);
        	*ssd += ssd_per_thread[0];
        	unlock(mutex);
    	}
		
	}

	return;
}

/* Jacobit iteration using global and shared memory, including coalecesed memory calls */
__global__ void jacobi_iteration_kernel_optimized (const matrix_t A, const matrix_t x, matrix_t x_update, const matrix_t B, int* mutex, double* ssd)
{
	__shared__ float aTile[TILE_SIZE][TILE_SIZE];
	__shared__ float xTile[TILE_SIZE];
	__shared__ double ssd_per_thread[TILE_SIZE];

	unsigned int i, k;
    unsigned int num_rows = A.num_rows;
    unsigned int num_cols = A.num_columns;
	float new_x;
	double sum = 0.0;
	double thisSSD = 0.0;

	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	int blockY = blockIdx.y;
	int row = blockDim.y * blockY + threadY;

	if (row < num_rows) {
		for (i = 0; i < num_cols; i+= TILE_SIZE) {
			/* Bring TILE_SIZE elements per row of A into shared memory */
			aTile[threadY][threadX] = A.elements[row * num_cols + i + threadX];
		
			/* Bring TILE_SIZE elements of x and b into shared memory */
			if (threadY == 0)
				xTile[threadX] = x.elements[i + threadX];

			__syncthreads();

			/* Compute partial Jacobi sum for the current tile */
			if (threadX == 0) {
				for (k = 0; k < TILE_SIZE; k+=1)
					sum += (double) aTile[threadY][k] * xTile[k];
			}

			__syncthreads();
		}

		if (threadX == 0) {
			float aDiag = A.elements[row * num_cols + row];
    	    float xDiag = x.elements[row];
	        float bDiag = B.elements[row];

			sum += -aDiag * xDiag;
			new_x = (bDiag - sum) / aDiag;

			thisSSD = (double) (new_x - xDiag) * (new_x - xDiag);
			x_update.elements[row] = new_x;

			ssd_per_thread[threadY] = thisSSD;
			__syncthreads();
			
			i = blockDim.y/2;
       		while (i != 0) {
            	if (threadY < i)
                	ssd_per_thread[threadY] += ssd_per_thread[threadY + i];
            	__syncthreads();
            	i /= 2;
        	}

        	if (threadY == 0) {
            	lock(mutex);
            	*ssd += ssd_per_thread[0];
            	unlock(mutex);
        	}
		}
	}
	
	return;
}

/* Jacobi iteration updating x */
__global__ void jacobi_update_x (matrix_t sol_x, const matrix_t new_x)
{
    unsigned int num_rows = sol_x.num_rows;

    int threadY = threadIdx.y;
    int threadX = threadIdx.x;
    int blockY = blockIdx.y;
    int row = blockDim.y * blockY + threadY;

    if ((row < num_rows) && (threadX == 0)) {
        sol_x.elements[row] = new_x.elements[row];
    }

    return;
}

