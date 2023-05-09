#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void handleCudaError(cudaError error) {
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA error: %s!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}

void handleAllocationError(void *ptr) {
	if (ptr == NULL) {
		fprintf(stderr, "Allocation failed!\n");
		exit(EXIT_FAILURE);
	}
}
