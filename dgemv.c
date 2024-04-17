#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>


void matrix_vector_product(double *a, double *b, double *c, int m, int n) {
	for (int i = 0; i < m; i++) {
		c[i] = 0.0;
		for (int j = 0; j < n; j++)
			c[i] += a[i * n + j] * b[j];
	}
}

double run_serial(int m, int n) {
	double *a, *b, *c;
	a = malloc(sizeof(*a) * m * n);
	b = malloc(sizeof(*b) * n);
	c = malloc(sizeof(*c) * m);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++)
			a[i * n + j] = i + j;
	}
	for (int j = 0; j < n; j++)
		b[j] = j;
	double t = omp_get_wtime();
	matrix_vector_product(a, b, c, m, n);
	t = omp_get_wtime() - t;

	printf("Elapsed time (serial): %.6f sec.\n", t);
	free(a);
	free(b);
	free(c);
	return t;
}


int get_bounds(int threadid, int k, int nthreads, int size) {

    int bound = 0;

    if (threadid < k)
        bound = threadid * ((size / nthreads) + 1);
    else
        bound = k * ((size / nthreads) + 1) + (threadid - k) * (size / nthreads);

    return bound;
}


double parallel_matrix_vector_product(double *a, double *b, double *c, int m, int n) {
    double sum = 0.0;
#pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int k = m % nthreads;
        int lb = get_bounds(threadid, k, nthreads, m);
        int ub = get_bounds(threadid + 1, k, nthreads, m);

        printf("Thread %d: get_bounds %d UB %d\n", threadid, lb, ub);

        double local_sum = 0.0;
        
        for (int i = lb; i < ub; i++) {
            c[i] = 0.0;
            for (int j = 0; j < n; j++) {
                c[i] += a[i * n + j] * b[j];
                local_sum += c[i];
            }
        }
#pragma omp atomic
        sum += local_sum;
    }
    printf("\nTotal sum: %lf\n", sum);

    return sum;
}


double run_parallel(int m, int n) {
    double *a, *b, *c;

    a = malloc(sizeof(*a) * m * n);
    b = malloc(sizeof(*b) * n);
    c = malloc(sizeof(*c) * m);
    for (int i = 0; i < m; i++) {
	    for (int j = 0; j < n; j++)
		    a[i * n + j] = i + j;
    }

    for (int j = 0; j < n; j++)
        b[j] = j;

    double t = omp_get_wtime();
    parallel_matrix_vector_product(a, b, c, m, n);
    t = omp_get_wtime() - t;
    printf("Elapsed time (parallel): %.6f sec.\n", t);
    free(a);
    free(b);
    free(c);
    return t;
}


int main(int argc, char **argv) {
	size_t m = 1000;
    size_t n = 1000;
	
	if (argc > 1)
        m = atoi(argv[1]);
    if (argc > 2)
        n = atoi(argv[2]);
	
    printf("Matrix-vector product (c[m] = a[m, n] * b[n]; m = %ld, n = %ld)\n", m, n);
    printf("Memory used: %"  PRIu64 " MiB\n", ((m * n + m + n) * sizeof(double)) >> 20);

    double tserial = run_serial(m, n);
    double tparallel = run_parallel(m,n);

    printf("Speedup: %.2f\n", tserial / tparallel);
    return 0;
}
