#include <stdio.h>
#include <stdlib.h>

#define RADIUS 2
#define BLOCK_SIZE 30

#define temp(i) temp[RADIUS + (i)]

__global__ void mykernel(int *in, int *out) {
  __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
  int gindex = threadIdx.x + blockIdx.x * blockDim.x;
  int lindex = threadIdx.x;

  // each thread reads a single value from global memory
  temp(lindex) = in[gindex];

  // first thread additionally reads halo on left
  if (lindex == 0) {
    for (int i=-RADIUS; i<0; i++)
      temp(i + lindex) = in[gindex + i];
  } 
  // last thread additionally reads halo on right
  if (lindex == blockDim.x - 1) {
    for (int i=0; i<RADIUS; i++)
      temp(lindex + i) = in[gindex + i];
  }
 
  // synchronize (ensure all the data is available)
  __syncthreads();

  // apply the stencil
  int result = 0;
  for (int i=-RADIUS; i<=RADIUS; i++) {
    result += temp(lindex + i);
  }

  // store the result
  out[gindex] = result;
}



int main(int argc, char **argv) {

  if (argc < 2) {
    printf("usage: %s N\n", argv[0]);
    return 1;
  }

  long N = atoi(argv[1]);

  if ((N-(2*RADIUS)) % BLOCK_SIZE != 0) {
    printf("N must be %d more than a multiple of %d for this example\n", 2*RADIUS, BLOCK_SIZE);
    return 1;
  }

  // pointers to host arrays
  int *h_in, *h_out;

  // pointers to device arrays
  int *d_in, *d_out;

  // allocate host memory
  h_in  = (int*)malloc(N * sizeof(int));
  h_out = (int*)malloc(N * sizeof(int));

  // allocate device memory
  cudaMalloc((void **)&d_in,  N * sizeof(int));
  cudaMalloc((void **)&d_out, N * sizeof(int));

  // initialize the arrays
  for (int i=0; i<N; i++) {
    h_in[i] = i;
  }

  // copy data from host arrays to device arrays
  cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
 
  // Do everything from the start of the first region.
  // If you don't do this, then you need to add special handling
  // for the first thread in the first block and the last thread
  // in the last block
  int *d_in_first_full_halo  = &(d_in[RADIUS]);
  int *d_out_first_full_halo = &(d_out[RADIUS]);

  // define the layout
  int gridDim = (N - (2*RADIUS+1)) / BLOCK_SIZE + 1;  // i.e., number of blocks
  int blockDim = BLOCK_SIZE;                          // i.e., number of threads per block

  // launch kernel
  mykernel<<<gridDim, blockDim>>>(d_in_first_full_halo, d_out_first_full_halo);
  cudaDeviceSynchronize();

  // copy data from device arrays to host arrays
  cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i=0; i<N; i++) {
    printf("host out[%d] = %d\n", i, h_out[i]);
  }

  return 0;
}

