#include <iostream>
#include <math.h>

//Kernel function to add the elements of 2 arrays
__global__
void add(int n, float *x, float *y) {
	//Add in to go through array with parallel threads
	int index = threadIdx.x;
	int stride = blockDim.x;

	for (int i = index; i < n; i += stride)
		y[i] = x[i] + y[i];
}

int main(void) {
	int N = 1<<20;
	float *x, *y;

	//allocate unified memory - accessible from CPU or GPU
	cudaMallocManaged(&x, N*sizeof(float));
	cudaMallocManaged(&y, N*sizeof(float));

	//initialize x and y arrays on the host
	for( int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}
	
	//run kernel on 1 million elements on the GPU
	add<<<1, 256>>>(N, x, y);

	//wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	//check for errors (all values should be 3.0f)
	float maxError = 0.0f;
	for( int i = 0; i < N; i++)
		maxError = fmax(maxError, fabs(y[i]-3.0f));
	std::cout << "Max error: " << maxError << std::endl;

	//free memory
	cudaFree(x);
	cudaFree(y);

	return 0;
}
