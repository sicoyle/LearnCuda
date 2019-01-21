#include <iostream>
#include <math.h>

//function to add the elements of 2 arrays
void add (int n, float *x, float *y) {
	for (int i = 0; i < n; i++)
		y[i] = x[i] + y[i];
}

int main(void) {
	int N = 1<<20; //1 million elements

	float *x = new float[N];
	float *y = new float[N];

	//initialize x and y arrays on the host
	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}
}
