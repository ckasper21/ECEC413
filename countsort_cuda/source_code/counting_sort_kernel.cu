/* Kernels for counting sort algorithm */

/* Counting sort: histogram generation */
__global__  void histogram_kernel(int *input_data, int *histogram, int num_elements, int histogram_size)
{
    __shared__ unsigned int s[HISTOGRAM_SIZE];
	
    /* Initialize the shared memory area. */ 
    if(threadIdx.x < histogram_size)
        s[threadIdx.x] = 0;
		
    __syncthreads();

    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
	
    while (offset < num_elements) {
        atomicAdd (&s[input_data[offset]], 1);
        offset += stride;
    }	  
	
    __syncthreads();

    /* Accumulate the histogram in shared memory into global memory. */
    if (threadIdx.x < histogram_size) 
        atomicAdd (&histogram[threadIdx.x], s[threadIdx.x]);

    return;
}

/* Counting sort: inclusive prefix scan */
__global__ void scan_kernel (int *out, int *in, int n)
{
    /* Dynamically allocated shared memory for storing the scan array. */
    extern  __shared__  int temp[];

    int tid = threadIdx.x;

    /* Indices for the ping-pong buffers. */
    int pout = 0;
    int pin = 1;

    /* Load the in array from global memory into shared memory. */
    temp[pout * n + tid] = in[tid];

    for (int offset = 1; offset < n; offset *= 2) {
        pout = 1 - pout;
        pin  = 1 - pout;
        __syncthreads();

        temp[pout * n + tid] = temp[pin * n + tid];

        if (tid >= offset)
            temp[pout * n + tid] += temp[pin * n + tid - offset];
    }

    __syncthreads();

    out[tid] = temp[pout * n + tid];
    return;
}

/* Counting sort */
__global__ void counting_sort_kernel (int *out, int *in, int n)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

	int i, j, diff, start;

	for (i = 0; i < n; i++) {
		if (i == 0) {
			diff = in[i];
			start = 0;
		} else {
			diff = in[i] - in[i-1];
			start = in[i-1];
		}	

		for (j = idx; j < diff; j+=stride)
			out[start+j] = i;
		__syncthreads();
	}
	
	return;
}

