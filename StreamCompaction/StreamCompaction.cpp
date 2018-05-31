#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

// Avoid Visual Studio LNK2019 compiler error
#pragma comment(lib, "OpenCL.lib")

using namespace std;

bool predicate(int num) {
	return num >= 5;
}

void serialCompact(int *input, int *output, unsigned int length, unsigned int *resultLength) {
	unsigned int j = 0;
	for (unsigned int i = 0; i < length; i++) {
		if (predicate(input[i])) {
			output[j] = input[i];
			j++;
		}
	}
	*resultLength = j;
}

bool isPowerOfTwo(int n) {
	return (n & (n - 1)) == 0;
}

int main() {

	/*
	int test[10] = {1, 5, 6, 3, 12, 3, 6, 8, 2, 1};
	int testResult[10];
	unsigned int resultLength = 0;

	serialCompact(test, testResult, 10, &resultLength);

	for (unsigned int i = 0; i < resultLength; i++) {
		printf("at %d: %d\n", i, testResult[i]);
	}

	return 0;
	*/

	cl_int err;
	cl_platform_id platforms[8];
	cl_uint numPlatforms;
	cl_device_id device;

	err = clGetPlatformIDs(8, platforms, &numPlatforms);
	if (err != CL_SUCCESS) {
		cerr << "OpenCL platforms not found." << std::endl;
		return 1;
	}

	err = clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	if (err != CL_SUCCESS) {
		cerr << "Device not found." << std::endl;
		return 1;
	}

	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
	cl_command_queue queue = clCreateCommandQueue(context, device, (cl_command_queue_properties)0, &err);
	if (err != CL_SUCCESS) {
		cerr << "Couldn't create command queue." << std::endl;
		return 1;
	}

	ifstream kernelFS("Compact.cl");
	string kernelSourceString((istreambuf_iterator<char>(kernelFS)), (istreambuf_iterator<char>()));
	const char *kernelSource = &kernelSourceString[0u];

	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, &err);
	if (err != CL_SUCCESS) {
		cerr << "Couldn't create program object." << std::endl;
		return 1;
	}

	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {

		fprintf(stderr, "Couldn't build program (%d).\n", err);
		return 1;
	}

	//cl_kernel scan = clCreateKernel(program, "blellochScan", &err);
	cl_kernel compactKernel = clCreateKernel(program, "compact", &err);
	if (err != CL_SUCCESS) {
		cerr << "Couldn't create kernel." << std::endl;
		return 1;
	}

	vector<int> scanInput = { 1, 2, 3, 24, 5, 6, 7, 8, 11, 1, 20};
	vector<int> scanOutput(scanInput.size(), 0);
	uint32_t scanArraySize = scanInput.size() * sizeof(int);

	// vector size: power of 2!
	while (!isPowerOfTwo(scanInput.size()))
		scanInput.push_back(0);

	cl_mem scanInBuffer =
		clCreateBuffer(context, CL_MEM_READ_ONLY, scanArraySize, NULL, &err);
	if (err != CL_SUCCESS) {
		cerr << "Couldn't create kernel input buffer." << std::endl;
		return 1;
	}

	cl_mem scanOutBuffer =
		clCreateBuffer(context, CL_MEM_READ_WRITE, scanArraySize, NULL, &err);
	if (err != CL_SUCCESS) {
		cerr << "Couldn't create kernel output buffer." << std::endl;
		return 1;
	}

	cl_mem passedElementBuffer =
		clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned int), NULL, &err);
	if (err != CL_SUCCESS) {
		cerr << "Couldn't create kernel output buffer." << std::endl;
		return 1;
	}

	err = clEnqueueWriteBuffer(queue, scanInBuffer, CL_TRUE, 0, scanArraySize, scanInput.data(), 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		cerr << "Couldn't write image input buffer." << std::endl;
		return 1;
	}

	unsigned int numberOfElements = scanInput.size();

	clSetKernelArg(compactKernel, 0, sizeof(cl_mem), (void *)&scanInBuffer);
	clSetKernelArg(compactKernel, 1, sizeof(cl_mem), (void *)&scanOutBuffer);
	clSetKernelArg(compactKernel, 2, sizeof(cl_uint), (void *)&numberOfElements);
	clSetKernelArg(compactKernel, 3, sizeof(cl_mem), (void *)&passedElementBuffer);

	//clSetKernelArg(scan, 0, sizeof(cl_mem), (void *)&scanInBuffer);
	//clSetKernelArg(scan, 1, sizeof(cl_mem), (void *)&scanOutBuffer);
	//clSetKernelArg(scan, 2, sizeof(cl_int), (void *)&numberOfElements);

	size_t globalItemSize = scanInput.size();
	size_t localItemSize = globalItemSize;

	err = clEnqueueNDRangeKernel(queue, compactKernel, 1, 0, &globalItemSize, &localItemSize, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		cerr << "Couldn't enqueue command to execute on kernel." << std::endl;
		return 1;
	}

	err = clEnqueueReadBuffer(queue, scanOutBuffer, CL_TRUE, 0, scanArraySize, scanOutput.data(), NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		cerr << "Couldn't enqueue command to read from kernel." << std::endl;
		return 1;
	}

	unsigned int numberOfPassedElements = 0;

	err = clEnqueueReadBuffer(queue, passedElementBuffer, CL_TRUE, 0, sizeof(unsigned int), &numberOfPassedElements, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		cerr << "Couldn't enqueue command to read from kernel." << std::endl;
		return 1;
	}

	//printf("numberOfPassedElements: %d\n", numberOfPassedElements);

	if (clFinish(queue) != CL_SUCCESS) {
		cerr << "Couldn't finish queue." << std::endl;
		return 1;
	}

	printf("output size: %d\n", numberOfPassedElements);

	for (unsigned int i = 0; i < numberOfPassedElements; i++) {
		printf("%d, ", scanOutput[i]);
	}
	printf("\n");

	return 0;
}