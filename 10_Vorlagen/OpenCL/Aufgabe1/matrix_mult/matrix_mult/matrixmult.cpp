#include "opencl_helper.h"
#include "math.h"

// C = A * B

const int C_ROWS = 1000;
const int C_COLS = 2000;
const int A_COLS = 3000;

const int A_ROWS = C_ROWS;
const int B_ROWS = A_COLS;
const int B_COLS = C_COLS;

const int BLOCK_SIZE = 8;

const char* KERNEL_FILE = "matrixmult.cl";
const char* KERNEL_FUNCTION = "MatrixMultKernel";

void openCLMatrixMult(const char* executablePath, float* A, float* B, float* C, int repetitions, bool warmup) {
	cl_int error = 0;
	cl_platform_id platform;
	handleClError(clGetPlatformIDs(1, &platform, NULL));

	cl_device_id device;
	handleClError(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL));

	cl_context context = clCreateContext(0, 1, &device, NULL, NULL, &error);
	handleClError(error);

	cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, &error);
	handleClError(error);

	char* kernelSource = NULL;
	size_t kernelLength = 0;
	readSourceFromFile(KERNEL_FILE, &kernelSource, &kernelLength);
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, &kernelLength, &error);
	handleClError(error);
	handleClError(clBuildProgram(program, 0, NULL, NULL, NULL, NULL));
	cl_kernel kernel = clCreateKernel(program, KERNEL_FUNCTION, &error);
	handleClError(error);

	clock_t start = clock();
	for (int i = 0; i < repetitions; i++) {
		// TODO: Implement GPU-parallel matrix multiplication on OpenCL (simple version)
	}
	handleClError(clReleaseKernel(kernel));
	handleClError(clReleaseProgram(program));
	handleClError(clReleaseCommandQueue(commandQueue));
	handleClError(clReleaseContext(context));
	if (!warmup) {
		float diff = float(clock() - start) / (CLOCKS_PER_SEC * repetitions);
		printf("OpenCL: %.3lf seconds\n", diff);
	}
}

void fillRandomArray(float* A, int numElements) {
	for (int i = 0; i < numElements; i++) {
		A[i] = rand() / (float)RAND_MAX;
	}
}

void verifyResults(float* A, float* B, float* C) {
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

void sequentialMatrixMult(float* A, float* B, float* C) {
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

int main(int argc, char** argv) {
	int nofElemA = A_ROWS * A_COLS;
	float* h_A = (float*)malloc(nofElemA * sizeof(float));
	handleAllocationError(h_A);
	fillRandomArray(h_A, nofElemA);

	int nofElemB = B_ROWS * B_COLS;
	float* h_B = (float*)malloc(nofElemB * sizeof(float));
	handleAllocationError(h_B);
	fillRandomArray(h_B, nofElemB);

	int nofElemC = C_ROWS * C_COLS;
	float* h_C = (float*)malloc(nofElemC * sizeof(float));
	handleAllocationError(h_C);

	openCLMatrixMult(argv[0], h_A, h_B, h_C, 2, true);
	openCLMatrixMult(argv[0], h_A, h_B, h_C, 4, false);
	verifyResults(h_A, h_B, h_C);

	sequentialMatrixMult(h_A, h_B, h_C);

	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}
