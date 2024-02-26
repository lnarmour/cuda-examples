# output

```
$ make
nvcc -g -G main.cu -o vadd
$ ./vadd 10
host a[0] = 0.000000
host a[1] = 1.000000
host a[2] = 2.000000
host a[3] = 3.000000
host a[4] = 4.000000
host a[5] = 5.000000
host a[6] = 6.000000
host a[7] = 7.000000
host a[8] = 8.000000
host a[9] = 9.000000
device a[0] = 0.000000
device a[6] = 6.000000
device a[2] = 2.000000
device a[8] = 8.000000
device a[4] = 4.000000
device a[1] = 1.000000
device a[7] = 7.000000
device a[3] = 3.000000
device a[9] = 9.000000
device a[5] = 5.000000

after copying back device->host:
host c[0] = 100.000000
host c[1] = 102.000000
host c[2] = 104.000000
host c[3] = 106.000000
host c[4] = 108.000000
host c[5] = 110.000000
host c[6] = 112.000000
host c[7] = 114.000000
host c[8] = 116.000000
host c[9] = 118.000000
```
