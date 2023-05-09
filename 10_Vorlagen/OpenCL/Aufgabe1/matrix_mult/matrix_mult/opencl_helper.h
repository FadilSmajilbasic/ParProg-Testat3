#pragma once

#include <stdlib.h>
#include <time.h>

#include <stdio.h>
#include <CL/opencl.h>

void handleClError(cl_int errorCode) {
	if (errorCode != CL_SUCCESS) {
		fprintf(stderr, "Open CL failed: %i\n", errorCode);
		exit(EXIT_FAILURE);
	}
}

void handleAllocationError(void* ptr) {
	if (ptr == NULL) {
		fprintf(stderr, "Allocation failed!\n");
		exit(EXIT_FAILURE);
	}
}

void readSourceFromFile(const char* fileName, char** source, size_t* sourceSize) {
	FILE* fp = NULL;
	fopen_s(&fp, fileName, "rb");
	if (fp == NULL)
	{
		fprintf(stderr, "Error: Couldn't find program source file '%s'.\n", fileName);
		exit(EXIT_FAILURE);
	}
	fseek(fp, 0, SEEK_END);
	*sourceSize = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	*source = new char[*sourceSize];
	if (*source == NULL)
	{
		fprintf(stderr, "Error: Couldn't allocate %d bytes for program source from file '%s'.\n", (int)*sourceSize, fileName);
		exit(EXIT_FAILURE);
	}
	fread(*source, 1, *sourceSize, fp);
}
