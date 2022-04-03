#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

__global__ void kernel(float* X, int L); 

int main(int argc, char** argv) {

  if (argc < 2) {
    printf("usage: %s N L\n", argv[0]);
    return 1;
  }

  int N = atoi(argv[1]);
  int L = atoi(argv[2]);

  // timer variables
  double time_input, time_compute, time_output;

  // create host data structure
  float *X = (float*)malloc(N*sizeof(float));
  // initialize with random values
  for (int i=0; i<N; i++) {
    X[i] = (float) rand() / (RAND_MAX);
  }
  // print first and last 3 values
  for (int i=0; i<3; i++)
    printf("%f  %s", X[i], i==2 ? "...  " : "");
  for (int i=N-3; i<N; i++)
    printf("%f  %s", X[i], i==N-1 ? "\n" : "");

  // create device data structure
  float *device_X;
  cudaMalloc((void **)&device_X, N*sizeof(float));

  // copy array from host to device
  start_timer();
  cudaMemcpy(device_X, X, N*sizeof(float), cudaMemcpyHostToDevice);
  stop_timer();
  time_input = elapsed_time();
  reset_timer();

  // set CUDA topology (number of threads and blocks)
  int num_threads = 1024; 
  int num_blocks = (N / num_threads) + 1;

  // launch CUDA kernel
  start_timer();
  kernel<<<num_blocks, num_threads>>>(device_X, L);
  // wait for kernel to finish
  cudaDeviceSynchronize();
  stop_timer();
  time_compute = elapsed_time();
  reset_timer();

  // copy array back to host
  start_timer();
  cudaMemcpy(X, device_X, N*sizeof(float), cudaMemcpyDeviceToHost);
  stop_timer();
  time_output = elapsed_time();

  // print first and last 3 values
  for (int i=0; i<3; i++)
    printf("%f  %s", X[i], i==2 ? "...  " : "");
  for (int i=N-3; i<N; i++)
    printf("%f  %s", X[i], i==N-1 ? "\n" : "");

  printf("\n");
  printf("Time to copy input:  %f\n", time_input);
  printf("Time to compute:     %f\n", time_compute);
  printf("Time to copy output: %f\n", time_output);
}

