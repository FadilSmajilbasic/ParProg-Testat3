#include "cuda_helper.cuh"

// C = A * B

const int C_ROWS = 1000;
const int C_COLS = 2000;
const int A_COLS = 3000;

const int A_ROWS = C_ROWS;
const int B_ROWS = A_COLS;
const int B_COLS = C_COLS;


__global__ void matrixMultKernel(float *A, float *B, float *C) {
      __shared__ float Asub[TILE_SIZE][TILE_SIZE];
      __shared__ float Bsub[TILE_SIZE][TILE_SIZE];

      int tx = threadIdx.x, ty = threadIdx.y;
      int col = blockIdx.x * TILE_SIZE + tx;
      int row = blockIdx.y * TILE_SIZE + ty;
      int nofTiles = (A_COLS + TILE_SIZE - 1) / TILE_SIZE;
      float sum = 0
	  for (int i = 0; i < A_COLS; i++)
	  {
		sum += A[col * A_COLS + i] * B[i * B_COLS + row];
	  }
	  C[col * C_COLS + row] = sum;
}

void cudaMatrixMult(float *A, float *B, float *C, int repetitions, bool warmup) {
	clock_t start = clock();

	for (int i = 0; i < repetitions; i++)
	{
		// TODO: Implement parallel tiled matrix multiplication on CUDA
	}
	if (!warmup)
	{
		float diff = float(clock() - start) / (CLOCKS_PER_SEC * repetitions);
		printf("CUDA: %.3lf seconds\n", diff);
	}
}

void fillRandomArray(float *A, int numElements) {
	for (int i = 0; i < numElements; i++) {
		A[i] = rand() / (float)RAND_MAX;
	}
}

void verifyResults(float *A, float *B, float *C) {
	printf("Verifying ...");
	for (int row = 0; row < C_ROWS; row++) {
		for (int col = 0; col < C_COLS; col++) {
			float sum = 0.0;
			for (int k = 0; k < A_COLS; k++) {
				sum += A[row * A_COLS + k] * B[k * B_COLS + col];
			}
			if (fabsf(C[row * C_COLS + col] - sum) > 1e-3f) {
				fprintf(stderr, "Result verification failed at element %d: %f vs. %f!\n", row, C[row * C_COLS + col], sum);
				exit(EXIT_FAILURE);
			}
		}
	}
	printf(" done\n");
}

void sequentialMatrixMult(float *A, float *B, float *C) {
	clock_t start = clock();

	for (int row = 0; row < C_ROWS; row++) {
		for (int col = 0; col < C_COLS; col++) {
			float sum = 0.0;
			for (int k = 0; k < A_COLS; k++) {
				sum += A[row * A_COLS + k] * B[k * B_COLS + col];
			}
			C[row * C_COLS + col] = sum;
		}
	}

	float diff = float(clock() - start) / CLOCKS_PER_SEC;
	printf("Sequential: %.3lf seconds\n", diff);
}

int main() {
	int nofElemA = A_ROWS * A_COLS;
	float *h_A = (float *)malloc(nofElemA * sizeof(float));
	handleAllocationError(h_A);
	fillRandomArray(h_A, nofElemA);

	int nofElemB = B_ROWS * B_COLS;
	float *h_B = (float *)malloc(nofElemB * sizeof(float));
	handleAllocationError(h_B);
	fillRandomArray(h_B, nofElemB);
	
	int nofElemC = C_ROWS * C_COLS;
	float *h_C = (float *)malloc(nofElemC * sizeof(float));
	handleAllocationError(h_C);

	cudaMatrixMult(h_A, h_B, h_C, 2, true);
	cudaMatrixMult(h_A, h_B, h_C, 4, false);
	verifyResults(h_A, h_B, h_C);

	sequentialMatrixMult(h_A, h_B, h_C);

	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}
