#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "grid.h"


/* This function solves the Gauss-Seidel method on the CPU using a single thread. */
int 
compute_gold (grid_t *grid)
{
    int num_iter = 0;
	int done = 0;
    int i, j;
	double diff;
	float old, new; 
    float eps = 1e-2; /* Convergence criteria. */
    int num_elements; 
	
	while(!done) { /* While we have not converged yet. */
        diff = 0.0;
        num_elements = 0;

        for (i = 1; i < (grid->dim - 1); i++) {
            for (j = 1; j < (grid->dim - 1); j++) {
                old = grid->element[i * grid->dim + j]; /* Store old value of grid point. */
                /* Apply the update rule. */	
                new = 0.25 * (grid->element[(i - 1) * grid->dim + j] +\
                              grid->element[(i + 1) * grid->dim + j] +\
                              grid->element[i * grid->dim + (j + 1)] +\
                              grid->element[i * grid->dim + (j - 1)]);

                grid->element[i * grid->dim + j] = new; /* Update the grid-point value. */
                diff = diff + fabs(new - old); /* Calculate the difference in values. */
                num_elements++;
            }
        }
		
        /* End of an iteration. Check for convergence. */
        diff = diff/num_elements;
        printf ("Iteration %d. DIFF: %f.\n", num_iter, diff);
        num_iter++;
			  
        if (diff < eps) 
            done = 1;
	}
	
    return num_iter;
}

