
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "RadialSymmetryImage.h"
#include "RSImage_GPU.h"
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
	//image dimension
	size_t width=160;
	size_t height=160;
	size_t ROI_width=128;
	size_t ROI_height=96;
	size_t x_off=(width-ROI_width)/2;
	size_t y_off=(height-ROI_height)/2;

	//GPU thread structure
	dim3 grid, block;
	size_t warpSize=16;
	grid.x=ROI_width/warpSize;
	grid.y=ROI_height/warpSize;
	block.x=warpSize;
	block.y=warpSize;

	size_t max_stream=10;
	cudaStream_t *streams=new cudaStream_t[max_stream];

	RSImage_GPU **RSImage=(RSImage_GPU**)malloc(max_stream*sizeof*RSImage);

	char * charImage=(char *)calloc(width*height,sizeof(char)); //an image from a file
	//pinned images
	uint8_t ** image=(uint8_t**)malloc(max_stream*sizeof*image);
	float **di=(float**)malloc(max_stream*sizeof*di);
	float **di2=(float**)malloc(max_stream*sizeof*di2);

	for(size_t i=0;i<max_stream;i++)
	{
		cudaMallocHost(&image[i],width*height*sizeof*(image[i]));
		cudaMallocHost(&di[i],(ROI_width-1)*(ROI_height)*sizeof*(di[i]));
		cudaMallocHost(&di2[i],(ROI_width-1)*(ROI_height)*sizeof*(di2[i]));

		cudaStreamCreate(&(streams[i]));
		RSImage[i]=(RSImage_GPU*)malloc(sizeof*RSImage[i]);
	}
	
	//read image
	ifstream imageFile("C:\\2.p.bin",ios::in|ios::binary|ios::ate);

	size_t size=imageFile.tellg();
	imageFile.seekg (0, ios::beg);

	printf("file sizeis %d\n",size);
	imageFile.read(charImage,size);

	for (size_t y=0;y<height;y++)
		for (size_t x=0;x<width;x++)
		{	
			for(size_t i=0;i<max_stream;i++)
			{
				image[i][width*y+x]=(uint8_t)charImage[width*y+x];
			}
		}

	//CPU image
	RadialSymmetryImage RC(image[0],160,160);
	
	//GPU image
	for(size_t i=0;i<max_stream;i++)
	{
		initRSImage(RSImage[i],image[i],width,height,ROI_width,ROI_height,x_off,y_off,RC.m_x_centroid,RC.m_y_centroid);
	}
	
	LARGE_INTEGER frequency;
	LARGE_INTEGER start;
	LARGE_INTEGER end;

	QueryPerformanceFrequency(&frequency);


	clock_t begin_time = clock();
	QueryPerformanceCounter(&start);

	for(size_t i=0;i<1000;i++)
	{
		RC.UpdateCenter();
	}
	QueryPerformanceCounter(&end);

	printf("%lf or %f, %f clokcs for 160x160 UpdateCenter()\n",(end.QuadPart - start.QuadPart)/(double)1000 / (double)frequency.QuadPart,float( clock () - begin_time ) /  CLOCKS_PER_SEC/(double)1000,float( clock () - begin_time ) );

	//warming-up
	transferRSImageHtoD(RSImage[0],streams[0]);
	updateRSImageCenter(RSImage[0], grid, block,streams[0]);
	transferRSImageDtoH(RSImage[0],streams[0]);
	calcCenter(RSImage[0],streams[0]);

	size_t cpu_holding=0;
	for(size_t trial=0;trial<10;trial++){
		cudaDeviceSynchronize();
		begin_time = clock();
		QueryPerformanceCounter(&start);

		for(size_t i=1;i<1000;i++)
		{
			//printf("Cursor: %d\n",i);
			transferRSImageHtoD(RSImage[i%max_stream],streams[i%max_stream]);
			updateRSImageCenter(RSImage[i%max_stream], grid, block,streams[i%max_stream]);
			transferRSImageDtoH(RSImage[i%max_stream],streams[i%max_stream]);
			calcCenter(RSImage[(i-cpu_holding)%max_stream],streams[(i-cpu_holding)%max_stream]);
			//calcCenter<<<1,1>>>(RSImage->d_x_c, RSImage->d_y_c, RSImage->d_x_c_old, RSImage->d_y_c_old, RSImage->d_det, RSImage->d_sw, RSImage->d_smmw, RSImage->d_smw, RSImage->d_smbw, RSImage->d_sbw);
		}

		for(size_t i=0;i<cpu_holding;i++) calcCenter(RSImage[(i-1)%max_stream],streams[(i-1)%max_stream]);

		cudaDeviceSynchronize();

		QueryPerformanceCounter(&end);

		printf("%lf or %f, %f clokcs for %dx%d CUDA\n",(end.QuadPart - start.QuadPart)/(double)1000 / (double)frequency.QuadPart,float( clock () - begin_time ) /  CLOCKS_PER_SEC/(double)1000,float( clock () - begin_time ) , ROI_width, ROI_height);
	}
	//free memory

	printf("Center position_GPU is (%f, %f)\nCenter position_CPU is (%f, %f)\n",RSImage[0]->h_x_c,RSImage[0]->h_y_c,RC.m_x_c,RC.m_y_c);

	for(size_t i=0;i<max_stream;i++)
	{
		cudaFreeHost(image[i]);
		cudaFreeHost(di[i]);
		cudaFreeHost(di2[i]);
		freeRSImage(RSImage[i]);
	}
	//free streamed arrays
	free(image);
	free(di);
	free(di2);
	free(RSImage);
	free(streams);

	//free image files
	free(charImage);
	imageFile.close();

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
