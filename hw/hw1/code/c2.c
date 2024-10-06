#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double timespec_diff_in_seconds(struct timespec start, struct timespec end) {
  double start_sec = start.tv_sec + start.tv_nsec / 1e9;
  double end_sec = end.tv_sec + end.tv_nsec / 1e9;
  return end_sec - start_sec;
}

// Function to calculate the mean time taken for the second half of repetitions
double calculate_mean_time_taken(double *exec_times, long reps) {
  double sum = 0.0;
  long start = reps / 2;
  for (long i = start; i < reps; i++) {
    sum += exec_times[i];
  }
  return sum / (reps - start);
}

// Function to calculate Bandwidth in GB/sec
double calculate_bandwidth(long N, double mean_time_taken) {
  double data_moved_bytes = 2.0 * N * sizeof(float);
  double data_moved_gb = data_moved_bytes / 1e9;
  double bandwidth = data_moved_gb / mean_time_taken;
  return bandwidth;
}

// Function to calculate Throughput in GFLOP/sec
double calculate_throughput(long N, double mean_time_taken) {
  double flops = 2.0 * (double)N;
  double throughput = (flops / mean_time_taken) / 1e9;
  return throughput;
}

double dpunroll(long N, float *pA, float *pB) {
  double R = 0.0;
  int j;
  for (j = 0; j < N; j += 4)
    R += pA[j] * pB[j] + pA[j + 1] * pB[j + 1] + pA[j + 2] * pB[j + 2] +
         pA[j + 3] * pB[j + 3];
  return R;
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <N> <reps>\n", argv[0]);
    return EXIT_FAILURE;
  }

  long N = atol(argv[1]);
  if (N <= 0) {
    fprintf(stderr, "Error: N must be a positive integer\n");
    return EXIT_FAILURE;
  }

  long reps = atol(argv[2]);
  if (reps <= 0) {
    fprintf(stderr, "Error: reps must be a positive integer\n");
    return EXIT_FAILURE;
  }

  float *pA = (float *)malloc(N * sizeof(float));
  float *pB = (float *)malloc(N * sizeof(float));
  double *exec_times = (double *)malloc(reps * sizeof(double));

  if (pA == NULL || pB == NULL || exec_times == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    return EXIT_FAILURE;
  }

  for (long i = 0; i < N; i++) {
    pA[i] = 1.0;
    pB[i] = 1.0;
  }

  double result = 0.0f;
  for (long i = 0; i < reps; i++) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    result = dpunroll(N, pA, pB);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_taken = timespec_diff_in_seconds(start, end);
    exec_times[i] = time_taken;
  }

  if (result != (double)N) {
    fprintf(stderr,
            "Error: Dot product result incorrect (expected %ld, got %f)\n", N,
            result);
    free(pA);
    free(pB);
    free(exec_times);
    return EXIT_FAILURE;
  }

  double mean_time_taken = calculate_mean_time_taken(exec_times, reps);
  double bandwidth = calculate_bandwidth(N, mean_time_taken);
  double throughput = calculate_throughput(N, mean_time_taken);

  printf("N: %ld <T>: %.6f sec B: %.6f GB/sec F: %.6f GFLOP/sec\n", N, mean_time_taken,
         bandwidth, throughput);
  free(pA);
  free(pB);
  free(exec_times);

  return EXIT_SUCCESS;
}
