all:
	nvcc main.cu kernel.cu timer.cu -o runner
