all: matmult matmult.simd

timer.o: timer.c
	gcc -c timer.c -o timer.o

matmult: mm.c timer.o
	gcc -O0 -fno-tree-vectorize $^ -o $@ 

matmult.simd: mm.c timer.o
	gcc -O0 -fno-tree-vectorize $^ -o $@ -DSIMD -mavx2 -mfma

clean:
	rm -f *.o matmult
