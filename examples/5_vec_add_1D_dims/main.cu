#include <stdio.h>
#include <stdlib.h>

__global__ void mykernel(float *a, float *b, float *c) {
  int i = blockIdx.x * 64 + threadIdx.x;

  c[i] = a[i] + b[i];
}

int resultNoGood(cudaError_t res) {
  return res != cudaSuccess;
}

int main(int argc, char **argv) {

  if (argc < 2) {
    printf("usage: %s N\n", argv[0]);
    return 1;
  }

  long N = atoi(argv[1]);

  // pointers to host arrays
  float *h_a, *h_b, *h_c;

  // pointers to device arrays
  float *d_a, *d_b, *d_c;

  // allocate host memory
  h_a = (float*)malloc(N * sizeof(float));
  h_b = (float*)malloc(N * sizeof(float));
  h_c = (float*)malloc(N * sizeof(float));

  // allocate device memory
  cudaMalloc((void **)&d_a, N * sizeof(float));
  cudaMalloc((void **)&d_b, N * sizeof(float));
  cudaMalloc((void **)&d_c, N * sizeof(float));

  // initialize the arrays
  for (int i=0; i<N; i++) {
    h_a[i] = i;
    h_b[i] = 100 + i;
    h_c[i] = 0;
  }

  // copy data from host arrays to device arrays
  cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c, N * sizeof(float), cudaMemcpyHostToDevice);
 
  // define the layout
  int gridDim = N / 64;  // i.e., number of blocks
  int blockDim = 64;     // i.e., number of threads per block

  // launch kernel 
  mykernel<<<gridDim, blockDim>>>(d_a, d_b, d_c);
  cudaDeviceSynchronize();

  // copy data from device arrays to host arrays
  cudaMemcpy(h_a, d_a, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_b, d_b, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i=0; i<N; i++) {
    printf("host c[%d] = %f\n", i, h_c[i]);
  }

  return 0;
}

