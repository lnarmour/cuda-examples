#include <stdio.h>

__global__ void mykernel(void) {
  printf("Hello from the device!\n");
}

int main(void) {

  printf("Hello from the host!\n");

  mykernel<<<1,1>>>();
  cudaDeviceSynchronize();

  return 0;
}

