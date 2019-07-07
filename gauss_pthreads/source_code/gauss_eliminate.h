#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

#define MATRIX_SIZE 2048

/* Matrix dimensions. */
#define NUM_COLUMNS MATRIX_SIZE     /* Number of columns in Matrix A. */
#define NUM_ROWS MATRIX_SIZE        /* Number of rows in Matrix A. */

/* Matrix Structure declaration. */
typedef struct {
    unsigned int num_columns;   /* Width of the matrix. */ 
    unsigned int num_rows;      /* Height of the matrix. */
	/* Number of elements between the beginnings of adjacent rows in the 
     * memory layout (useful for representing sub-matrices) */
    unsigned int pitch;
    float* elements;            /* Pointer to the first element of the matrix. */

} Matrix;

/* Structure data type of p_threads arg */
typedef struct args_thread {
    int threadID;
    Matrix *U;
    
} ARGS_FOR_THREAD;


#endif /* _MATRIXMUL_H_ */

