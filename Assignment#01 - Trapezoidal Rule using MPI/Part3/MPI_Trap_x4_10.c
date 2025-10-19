#include <mpi.h>
#include <stdio.h>

// Function to integrate
double f(double x) {
    return (x * x * x * x) - 3 * (x * x) + x + 4;
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

int main(int argc, char** argv) {
    int my_rank, comm_sz, n = 10;
    double a = 0.0, b = 2.0, h, local_a, local_b;
    int local_n;
    double local_int, total_int;
    double start, finish, elapsed;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    // Start timing the entire program
    start = MPI_Wtime();

    h = (b - a) / n;  // width of each trapezoid
    local_n = n / comm_sz;  // number of trapezoids for each process

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

    // Output the result and total time
    if (my_rank == 0) {
        printf("With n = %d trapezoids, our estimate of the integral from %f to %f = %.15e\n", n, a, b, total_int);
        printf("Total elapsed time = %f seconds\n", elapsed);
    }

    MPI_Finalize();
    return 0;
}
