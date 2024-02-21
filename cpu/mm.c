#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include "timer.h"

#define A(i,j) A[(i)*N+(j)]
#define B(i,j) B[(i)*N+(j)]
#define C(i,j) C[(i)*N+(j)]

int main(int argc, char **argv) {

  int N = (argc<2) ? 1200 : atoi(argv[1]);

  if (N%8 != 0) {
    printf("N must be a multiple of 8 in this example\n");
    return 1;
  }

  float *A = (float*)malloc(N*N*sizeof(float));
  float *B = (float*)malloc(N*N*sizeof(float));
  float *C = (float*)malloc(N*N*sizeof(float));

  srand(0);

  // initialization
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A(i,j) = rand() % 100;
      B(i,j) = rand() % 100;
      C(i,j) = 0;
    }
  }

  // start Timer
  double time;
  initialize_timer();
  start_timer();
  
  // compute
#ifndef SIMD
  for (int i=0; i<N; i++) {
    for (int k=0; k<N; k++) {
      for (int j=0; j<N; j++) {
        C(i,j) += A(i,k) * B(k,j);
      }
    }
  }
#else
  for (int i=0; i<N; i++) {
    for (int k=0; k<N; k++) {
      for (int j=0; j<N; j+=8) {
        // load 8 32-bit floats into the vector registers
        __m256 a = _mm256_loadu_ps(&(A(i,k)));
        __m256 b = _mm256_loadu_ps(&(B(k,j)));
        __m256 c = _mm256_loadu_ps(&(C(i,j)));
        
        // vector fused multiply add 
        __m256 z = _mm256_fmadd_ps(a, b, c);
         
        // store 8 32-bit floats from the result back into C
        _mm256_storeu_ps(&C(i,j), z);
      }
    }
  }
#endif

  // stop timer
  stop_timer();
  time = elapsed_time();
 
  printf("elapsed time = %lf\n", time);
  return 0;

}
