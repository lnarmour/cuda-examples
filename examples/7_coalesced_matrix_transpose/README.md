# output
```
$ make
nvcc main.cu -o transpose
$ ./transpose 

Device : NVIDIA GeForce GTX 1060 6GB
Matrix size: 1024 1024, Block size: 32 8, Tile size: 32 32
dimGrid: 32 32 1. dimBlock: 32 8 1
                  Routine         Bandwidth (GB/s)
                     copy              147.71
       shared memory copy              150.75
          naive transpose               54.53
      coalesced transpose              113.02
  conflict-free transpose              151.45
```
