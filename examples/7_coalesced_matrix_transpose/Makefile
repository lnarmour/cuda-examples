all: transpose

transpose: main.cu
	nvcc $^ -o transpose

clean:
	rm -f transpose* *.o
