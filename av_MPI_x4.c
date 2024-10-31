#include <mpi.h>
#include <stdio.h>
#include <math.h>

// Function to integrate
double f(double x) {
    return pow(x, 4) - 3 * pow(x, 2) + x + 4;
}

// Compute local trapezoidal approximation
double Trap(double left_endpt, double right_endpt, int trap_count, double base_len) {
    double estimate, x;
    int i;

    estimate = (f(left_endpt) + f(right_endpt)) / 2.0;
    for (i = 1; i <= trap_count - 1; i++) {
        x = left_endpt + i * base_len;
        estimate += f(x);
    }
    estimate = estimate * base_len;

    return estimate;
}

int main(int argc, char **argv) {
    int my_rank, comm_sz, n = 16384;
    double a = 0.0, b = 2.0, h, local_a, local_b;
    int local_n;
    double local_int, total_int;
    double start, finish, elapsed, total_elapsed = 0.0;
    int iterations = 10;  // Number of repetitions

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    for (int i = 0; i < iterations; i++) {
        // Start timing the entire program
        start = MPI_Wtime();

        h = (b - a) / n;        // Width of each trapezoid
        local_n = n / comm_sz;  // Number of trapezoids for each process

        // Each process calculates its local interval
        local_a = a + my_rank * local_n * h;
        local_b = local_a + local_n * h;

        // Calculate the local integral
        local_int = Trap(local_a, local_b, local_n, h);

        // Sum up the integrals from all processes using MPI_Reduce
        MPI_Reduce(&local_int, &total_int, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // End timing the entire program
        finish = MPI_Wtime();
        elapsed = finish - start;

        // Accumulate elapsed time for averaging
        if (my_rank == 0) {
            total_elapsed += elapsed;
        }
    }

    // Calculate and output the average time
    if (my_rank == 0) {
        double average_time = total_elapsed / iterations;
        printf("With n = %d trapezoids, our estimate of the integral from %f to %f = %.15e\n", n, a, b, total_int);
        printf("Average elapsed time over %d iterations = %f seconds\n", iterations, average_time);
    }

    MPI_Finalize();
    return 0;
}
