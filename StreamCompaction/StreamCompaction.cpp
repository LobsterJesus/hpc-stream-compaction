#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#define SERIAL false

// Avoid Visual Studio LNK2019 compiler error
#pragma comment(lib, "OpenCL.lib")

using namespace std;

cl_context _context;
cl_command_queue _queue;
cl_program _program;
unsigned int _argCount = 0;

bool predicate(int num) 
{
	return num >= 5;
}

void serialCompact(
	vector<int> input, 
	vector<int> *output, 
	unsigned int *resultLength) 
{
	unsigned int j = 0;
	for (unsigned int i = 0; i < input.size(); i++) {
		if (predicate(input[i])) {
			(*output)[j] = input[i];
			j++;
		}
	}
	*resultLength = j;
}

bool isPowerOfTwo(int n) 
{
	return (n & (n - 1)) == 0;
}

int setupDevice() 
{
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

	_context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
	_queue = clCreateCommandQueue(_context, device, (cl_command_queue_properties)0, &err);
	if (err != CL_SUCCESS) {
		cerr << "Couldn't create command queue." << std::endl;
		return 1;
	}
	return 0;
}

int compileProgram(const char *filename)
{
	cl_int err;
	ifstream kernelFS(filename);
	string kernelSourceString((istreambuf_iterator<char>(kernelFS)), (istreambuf_iterator<char>()));
	const char *kernelSource = &kernelSourceString[0u];

	_program = clCreateProgramWithSource(_context, 1, (const char**)&kernelSource, NULL, &err);
	if (err != CL_SUCCESS) {
		cerr << "Couldn't create program object." << std::endl;
		return 1;
	}

	err = clBuildProgram(_program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {

		fprintf(stderr, "Couldn't build program (%d).\n", err);
		return 1;
	}
	return 0;
}

int createKernel(const char *functionName, cl_kernel *kernel)
{
	cl_int err;
	*kernel = clCreateKernel(_program, functionName, &err);
	if (err != CL_SUCCESS) {
		cerr << "Couldn't create kernel." << std::endl;
		return 1;
	}
	return 0;
}

int setInBufferArg(
	const cl_kernel kernel, 
	const size_t size,
	const void *data)
{
	cl_int err;
	cl_mem buffer =
		clCreateBuffer(_context, CL_MEM_READ_ONLY, size, NULL, &err);
	if (err != CL_SUCCESS) {
		cerr << "Couldn't create kernel input buffer." << std::endl;
		return 1;
	}
	err = clEnqueueWriteBuffer(_queue, buffer, CL_TRUE, 0, size, data, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		cerr << "Couldn't write image input buffer." << std::endl;
		return 1;
	}
	clSetKernelArg(kernel, _argCount++, sizeof(cl_mem), (void *)&buffer);
	return 0;
}

int setOutBufferArg(
	const cl_kernel kernel,
	const size_t size,
	cl_mem *buffer)
{
	cl_int err;
	*buffer = clCreateBuffer(_context, CL_MEM_READ_WRITE, size, NULL, &err);

	if (err != CL_SUCCESS) {
		cerr << "Couldn't create kernel input buffer." << std::endl;
		return 1;
	}
	clSetKernelArg(kernel, _argCount++, sizeof(cl_mem), (void *)buffer);
	return 0;
}

int readBuffer(const cl_mem buffer, const size_t size, void *data)
{
	cl_int err;
	err = clEnqueueReadBuffer(_queue, buffer, CL_TRUE, 0, size, data, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		cerr << "Couldn't enqueue command to read from kernel." << std::endl;
		return 1;
	}
	return 0;
}

int runKernel(cl_kernel kernel, size_t size)
{
	cl_int err;
	err = clEnqueueNDRangeKernel(_queue, kernel, 1, 0, &size, &size, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		cerr << "Couldn't enqueue command to execute on kernel." << std::endl;
		return 1;
	}
	return 0;
}

int finish()
{
	if (clFinish(_queue) != CL_SUCCESS) {
		cerr << "Couldn't finish queue." << std::endl;
		return 1;
	}
}

int main() 
{
	vector<int> scanInput = { 1, 2, 3, 24, 5, 6, 7, 8, 11, 1, 20 };
	vector<int> scanOutput(scanInput.size(), 0);
	uint32_t scanArraySize = scanInput.size() * sizeof(int);
	unsigned int numberOfPassedElements = 0;

	// vector size: pad to power of 2
	while (!isPowerOfTwo(scanInput.size()))
		scanInput.push_back(0);

	if (!SERIAL) 
	{
		cl_kernel compactKernel;
		cl_mem scanOutBuffer;
		cl_mem passedElementBuffer;
		unsigned int numberOfElements = scanInput.size();
		
		if (setupDevice() != 0) return 1;
		if (compileProgram("Compact.cl") != 0) return 1;
		if (createKernel("compact", &compactKernel) != 0) return 1;

		setInBufferArg(compactKernel, scanArraySize, scanInput.data());
		setOutBufferArg(compactKernel, scanArraySize, &scanOutBuffer);
		clSetKernelArg(compactKernel, _argCount++, sizeof(cl_uint), (void *)&numberOfElements);
		setOutBufferArg(compactKernel, sizeof(unsigned int), &passedElementBuffer);

		runKernel(compactKernel, scanInput.size());

		readBuffer(scanOutBuffer, scanArraySize, scanOutput.data());
		readBuffer(passedElementBuffer, sizeof(unsigned int), &numberOfPassedElements);

		finish();
	}
	else
	{
		serialCompact(scanInput, &scanOutput, &numberOfPassedElements);
	}

	printf("output size: %d\n", numberOfPassedElements);
	for (unsigned int i = 0; i < numberOfPassedElements; i++) {
		printf("%d, ", scanOutput[i]);
	}
	printf("\n");

	return 0;
}