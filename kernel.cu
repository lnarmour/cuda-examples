__global__ void kernel(float *X, int L){

  // threadIdx.x  - thread ID
  // blockIdx.x   - block ID
  // gridDim.x    - total number of blocks
  // blockDim.x   - total number of threads in each block

  // 
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;

  for (int cnt=0; cnt<L; cnt++) {
    X[i] += 1;
  }
}
