#include <iostream>
#include <omp.h>
#include <vector>

using namespace std;

const double tau = 0.00001;
const double epsilon = 0.000000001;
double ProductOfMatrix(const double* A, const double* x, int i, long long matrix_width) {
    double sum = 0;

    for (int j = 0; j < matrix_width; j++) {
        sum += A[i * matrix_width + j] * x[j]; // умножение матрицы на вектор
    }

    return sum;
}

void linear_equation(double* A, double* b, double* x, long long N) {
    double sum_2 = 0;

    for (int i = 0; i < N; i++) {
        sum_2 += b[i] * b[i]; // сумма квадратов
    }

    int index = 0;
    while (true) {
        index++;
        double sum, sum_1 = 0;

        double* arg = new double[N];
        for (int j = 0; j < N; j++) {
            // x^{n+1} = x^n - tau(Ax^n - b)
            // (||Ax^n-b||_2) / (||b||_2) <epsilon
            // ||x||_2 = sqrt(sum_0^{N-1} u^2_i)
            sum = ProductOfMatrix(A, x, j, N);  // сумма перемножений матрицы на вектор
            sum_1 += (sum - b[j]) * (sum - b[j]); // квадрат разности сумм (числитель)
            arg[j] = tau * (sum - b[j]);
        }

        for (int j = 0; j < N; j++) {
            x[j] -= arg[j]; // x^{n+1} = x^n - tau(Ax^n - b)
        }

        if (sum_1 / sum_2 < epsilon) {
            break;
        }
    }

    cout << x[0] << endl;
}

void parallel_linear_equation_v1(double* A, double* b, double* x, long long N) {
    double sum_1;
    double sum_2 = 0;

#pragma omp parallel
    {
        double sum_lc = 0;
#pragma omp for schedule(guided, 100) nowait
        for (int j = 0; j < N; j++) {
            sum_lc += b[j] * b[j];
        }

#pragma omp atomic
        sum_2 += sum_lc;
    }

    double* arr = new double[N];

    while (true) {
        for (int i = 0; i < N; i++) {
            arr[i] = 0;
        }

        sum_1 = 0;

#pragma omp parallel
        {
#pragma omp for schedule(guided, 100)
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    arr[i] += A[i * N + j] * x[j];
                }
            }

#pragma omp for schedule(guided, 100) reduction(+:sum_1)
            for (int i = 0; i < N; i++) {
                sum_1 += (arr[i] - b[i]) * (arr[i] - b[i]);
                x[i] -= tau * (arr[i] - b[i]);
            }
        }

        if (sum_1 / sum_2 < epsilon) {
            break;
        }
    }

    cout << x[0] << endl;
}

void parallel_linear_equation_v2(double* A, double* b, double* x, long long N) {
    double sum_1;
    double sum_2 = 0;

#pragma omp parallel
    {
#pragma omp for schedule(guided, 100) reduction(+:sum_2) nowait
        for (int j = 0; j < N; j++) {
            sum_2 += b[j] * b[j];
        }
    }

    auto* arr = new double[N];

#pragma omp parallel
    {
        double sum_lc;

        while (true) {
            sum_1 = 0;
#pragma omp for schedule(guided, 100)
            for (int i = 0; i < N; i++) {
                arr[i] = 0;
                for (int j = 0; j < N; j++) {
                    arr[i] += A[i * N + j] * x[j];
                }
            }


            sum_lc = 0;
#pragma omp for schedule(guided, 100)
            for (int i = 0; i < N; i++) {
                sum_lc += (arr[i] - b[i]) * (arr[i] - b[i]);
                x[i] -= tau * (arr[i] - b[i]);
            }

#pragma omp atomic
            sum_1 += sum_lc;

#pragma omp barrier

            if (sum_1 / sum_2 < 0.000000001)
                break;

#pragma omp barrier
        }
    }

    cout << x[0] << endl;
}

void test(long long N) {
    int length = N * N;
    auto* A = new double[N * N];
    auto* b = new double[N];
    auto* x = new double[N];

    for (int i = 0; i < length; i++)
        A[i] = ((i % N == i / N) ? 2.0 : 1.0);

    for (int i = 0; i < N; i++) {
        b[i] = N + 1;
        x[i] = 0.0;
    }

    double t = omp_get_wtime();
    linear_equation(A, b, x, N);
    double t1 = omp_get_wtime();

    cout << "serial version: " << t1 - t << endl;
}

void parallel_test(long long N) {
    int length = N * N;
    auto* A = new double[N * N];
    auto* b = new double[N];
    auto* x = new double[N];

    for (int i = 0; i < length; i++)
        A[i] = ((i % N == i / N) ? 2.0 : 1.0);

    for (int i = 0; i < N; i++) {
        b[i] = N + 1;
        x[i] = 0.0;
    }

    double t = omp_get_wtime();
    parallel_linear_equation_v1(A, b, x, N);
    double t1 = omp_get_wtime();

    cout << "parallel_v1: " << t1 - t << '\n' << endl;

    for (int i = 0; i < N; i++)
        x[i] = 0;

    t = omp_get_wtime();
    parallel_linear_equation_v2(A, b, x, N);
    t1 = omp_get_wtime();

    cout << "parallel_v2: " << t1 - t << '\n' << endl;
}

int main(int argc, char** argv) {
    size_t SIZE = 1000;

    if (argc > 1)
        SIZE = atoi(argv[1]);

    parallel_test(SIZE);
    test(SIZE);

    return 0;
}
