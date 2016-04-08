
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "RadialSymmetryImage.h"
#include <Windows.h>
#include <ctime>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

using namespace std;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

int main()
{
	float *x_c;
	float *y_c;
	float a,b;
	x_c=&a;
	y_c=&b;

	char * charImage=(char *)calloc(160*160,sizeof(char)); //File
	uint8_t * image=(uint8_t *)calloc(160*160,sizeof(uint8_t)); //File
	uint8_t * image_512=(uint8_t *)calloc(512*256,sizeof(uint8_t)); //File

	ifstream imageFile("C:\\Users\\y\\Desktop\\HSMT\\test_bin\\2.p.bin",ios::in|ios::binary|ios::ate);

	size_t size=imageFile.tellg();
	imageFile.seekg (0, ios::beg);

	printf("file sizeis %d\n",size);
	imageFile.read(charImage,size);

	for (size_t y=0;y<160;y++)
		for (size_t x=0;x<160;x++)
			image[160*y+x]=(uint8_t)charImage[160*y+x];

	RadialSymmetryImage RC(image,160,160);

	for(int i=0;i<10;i++)
	{
		RC.GetCenter(x_c, y_c);
		printf("%f, %f\n", *x_c, *y_c);

	}
	imageFile.close();


	LARGE_INTEGER frequency;
	LARGE_INTEGER start;
	LARGE_INTEGER end;

	double elapsedSeconds;
	QueryPerformanceFrequency(&frequency);


	clock_t begin_time = clock();
	QueryPerformanceCounter(&start);

	RadialSymmetryImage RC2(image,128,128);

	for(size_t i=0;i<1000;i++)
	{
		RC2.UpdateCenter();
	}
	QueryPerformanceCounter(&end);

	printf("%lf or %f, %f clokcs for 128x128 UpdateCenter()\n",(end.QuadPart - start.QuadPart)/(double)1000 / (double)frequency.QuadPart,float( clock () - begin_time ) /  CLOCKS_PER_SEC,float( clock () - begin_time ) );

	begin_time = clock();
	QueryPerformanceCounter(&start);

	RadialSymmetryImage RC3(image,128,38);

	for(size_t i=0;i<1000;i++)
	{
		RC3.UpdateCenter();
	}
	QueryPerformanceCounter(&end);

	printf("%lf or %f, %f clokcs for 128x38 UpdateCenter()\n",(end.QuadPart - start.QuadPart)/(double)1000 / (double)frequency.QuadPart,float( clock () - begin_time ) /  CLOCKS_PER_SEC,float( clock () - begin_time ) );

	begin_time = clock();
	QueryPerformanceCounter(&start);

	for(size_t i=0;i<1000;i++)
	{
		RC.UpdateCenter();
	}
	QueryPerformanceCounter(&end);

	printf("%lf or %f, %f clokcs for 160x160 UpdateCenter()\n",(end.QuadPart - start.QuadPart)/(double)1000 / (double)frequency.QuadPart,float( clock () - begin_time ) /  CLOCKS_PER_SEC,float( clock () - begin_time ) );

	free(image);
	free(charImage);

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
