# Some example CUDA code

Does the following:
1. creates an array of random values of length N 
2. copies the array from the host to the device (GPU)
3. adds 1 to each element in the array L times
4. copies the array from the device back to the host
5. prints first and last 3 values before and after
6. prints timing info for steps 2, 3, and 4 above

To compile run:
```
$ nvcc main.cu kernel.cu timer.cu -o runner
```

This will produce a binary called "runner", which you can run as:
```
$ ./runner N L
```

For example:
```
(py36) eel:~/git/md/cuda$ make
nvcc main.cu kernel.cu timer.cu -o runner
(py36) eel:~/git/md/cuda$ 
(py36) eel:~/git/md/cuda$ 
(py36) eel:~/git/md/cuda$ ./runner 100000000 100
0.840188  0.394383  0.783099  ...  0.252915  0.190237  0.988865  
100.840187  100.394379  100.783096  ...  100.252914  100.190239  100.988861  

Time to copy input:  0.048772
Time to compute:     0.008145
Time to copy output: 0.030898
(py36) eel:~/git/md/cuda$ 
(py36) eel:~/git/md/cuda$ 
(py36) eel:~/git/md/cuda$ ./runner 100000000 10000
0.840188  0.394383  0.783099  ...  0.252915  0.190237  0.988865  
10000.839844  10000.394531  10000.783203  ...  10000.252930  10000.190430  10000.988281  

Time to copy input:  0.049360
Time to compute:     0.526128
Time to copy output: 0.030999
```
