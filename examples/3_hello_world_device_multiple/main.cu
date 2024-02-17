#include <stdio.h>

__global__ void mykernel(void) {
  printf("Hello from the device, block %d, thread %d!\n", blockIdx.x, threadIdx.x);
}

int main(void) {

  printf("Hello from the host!\n");

  mykernel<<<4,10>>>();
  cudaDeviceSynchronize();

  return 0;
}

