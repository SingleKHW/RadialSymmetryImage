
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
	/*memory allocation test*/

	float *x_c;
	float *y_c;
	float a,b;
	x_c=&a;
	y_c=&b;

	char * charImage=(char *)calloc(160*160,sizeof(char)); //File
	//uint8_t * image=(uint8_t *)calloc(160*160,sizeof(uint8_t)); //File
	uint8_t * image;
	cudaMallocHost(&image,160*160*sizeof*image);
	uint8_t ** image2D=(uint8_t **)malloc(sizeof*image2D*160);
	uint8_t * image2=(uint8_t *)calloc(160*160,sizeof(uint8_t));
	float *di=(float*)malloc(160*160*sizeof(float));
	for(size_t y=0;y<160;y++)
		image2D[y]=image+160*y;

	ifstream imageFile("C:\\2.p.bin",ios::in|ios::binary|ios::ate);

	size_t size=imageFile.tellg();
	imageFile.seekg (0, ios::beg);

	printf("file sizeis %d\n",size);
	imageFile.read(charImage,size);

	for (size_t y=0;y<160;y++)
		for (size_t x=0;x<160;x++)
			image[160*y+x]=(uint8_t)charImage[160*y+x];


	RadialSymmetryImage RC(image,160,160);


	float * h_x_c;
	float * h_y_c;
	float dummy1, dummy2;

	dummy1=-5.0f;

	h_x_c=&dummy1;
	h_y_c=&dummy2;

	size_t width=160;
	size_t height=160;
	size_t ROI_width=128;
	size_t ROI_height=128;
	size_t x_off=(width-ROI_width)/2;
	size_t y_off=(height-ROI_height)/2;

	dim3 grid, block;
	size_t warpSize=16;
	grid.x=ROI_width/warpSize;
	grid.y=ROI_height/warpSize;
	block.x=warpSize;
	block.y=warpSize;

	RSImage_GPU *RSImage=(RSImage_GPU*)malloc(sizeof*RSImage);

	initRSImage(RSImage,width,height,ROI_width,ROI_height,x_off,y_off,RC.m_x_centroid,RC.m_y_centroid);

	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = RSImage->d_du;
	resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
	resDesc.res.linear.desc.x = 32; // bits per channel
	resDesc.res.linear.sizeInBytes = (ROI_width-1)*(ROI_height-1)*sizeof(float);

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;

	cudaTextureObject_t tex=0;
	cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

	cudaMemcpy(RSImage->d_image,image,width*height*sizeof*RSImage->d_image,cudaMemcpyHostToDevice);
	//cudaMemcpy(RSImage->d_dervs,h_x_c,160*160*sizeof*h_x_c,cudaMemcpyHostToDevice);

	calcDervs<<<grid,block>>>(RSImage->d_image,RSImage->d_du,RSImage->d_dv,RSImage->d_x_off,RSImage->d_y_off,RSImage->d_width, RSImage->d_height, RSImage->d_ROIwidth, RSImage->d_ROIheight);
	calcDervsF<<<grid,block>>>(RSImage->d_du,RSImage->d_duF,RSImage->d_dv,RSImage->d_dvF, RSImage->d_ROIwidth, RSImage->d_ROIheight,RSImage->d_smw, RSImage->d_smmw, RSImage->d_sw, RSImage->d_smbw, RSImage->d_sbw);
	calcGrads<<<grid,block>>>(RSImage->d_duF, RSImage->d_dvF, RSImage->d_ROIwidth, RSImage->d_ROIheight, RSImage->d_x_c, RSImage->d_y_c, RSImage->d_x_c_old, RSImage->d_y_c_old, RSImage->d_sw, RSImage->d_smmw, RSImage->d_smw, RSImage->d_smbw, RSImage->d_sbw);
	calcCenter<<<1,1>>>(RSImage->d_x_c, RSImage->d_y_c, RSImage->d_x_c_old, RSImage->d_y_c_old, RSImage->d_det, RSImage->d_sw, RSImage->d_smmw, RSImage->d_smw, RSImage->d_smbw, RSImage->d_sbw);

	cudaMemcpy(h_x_c,RSImage->d_x_c,sizeof*h_x_c,cudaMemcpyDeviceToHost);

	printf("d_x_c=%f, RC.m_x_c=%f\n",*h_x_c, RC.m_x_c);


	
	LARGE_INTEGER frequency;
	LARGE_INTEGER start;
	LARGE_INTEGER end;

	double elapsedSeconds;
	QueryPerformanceFrequency(&frequency);


	clock_t begin_time = clock();
	QueryPerformanceCounter(&start);

	for(size_t i=0;i<1000;i++)
	{
		RC.UpdateCenter();
	}
	QueryPerformanceCounter(&end);

	printf("%lf or %f, %f clokcs for 160x160 UpdateCenter()\n",(end.QuadPart - start.QuadPart)/(double)1000 / (double)frequency.QuadPart,float( clock () - begin_time ) /  CLOCKS_PER_SEC/(double)1000,float( clock () - begin_time ) );

	cudaStream_t stream1, stream2, stream3;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);

	cudaDeviceSynchronize();
	begin_time = clock();
	QueryPerformanceCounter(&start);

	for(size_t i=0;i<1000;i++)
	{
		calcDervs<<<grid,block>>>(RSImage->d_image,RSImage->d_du,RSImage->d_dv,RSImage->d_x_off,RSImage->d_y_off,RSImage->d_width, RSImage->d_height, RSImage->d_ROIwidth, RSImage->d_ROIheight);
		calcDervsF<<<grid,block>>>(RSImage->d_du,RSImage->d_duF,RSImage->d_dv,RSImage->d_dvF, RSImage->d_ROIwidth, RSImage->d_ROIheight,RSImage->d_smw, RSImage->d_smmw, RSImage->d_sw, RSImage->d_smbw, RSImage->d_sbw);
		calcGrads<<<grid,block>>>(RSImage->d_duF, RSImage->d_dvF, RSImage->d_ROIwidth, RSImage->d_ROIheight, RSImage->d_x_c, RSImage->d_y_c, RSImage->d_x_c_old, RSImage->d_y_c_old, RSImage->d_sw, RSImage->d_smmw, RSImage->d_smw, RSImage->d_smbw, RSImage->d_sbw);
		//calcCenter<<<1,1>>>(RSImage->d_x_c, RSImage->d_y_c, RSImage->d_x_c_old, RSImage->d_y_c_old, RSImage->d_det, RSImage->d_sw, RSImage->d_smmw, RSImage->d_smw, RSImage->d_smbw, RSImage->d_sbw);
	}
	cudaDeviceSynchronize();

	QueryPerformanceCounter(&end);

	printf("%lf or %f, %f clokcs for %dx%d CUDA\n",(end.QuadPart - start.QuadPart)/(double)1000 / (double)frequency.QuadPart,float( clock () - begin_time ) /  CLOCKS_PER_SEC/(double)1000,float( clock () - begin_time ) , ROI_width, ROI_height);

	cudaDeviceSynchronize();
	begin_time = clock();
	QueryPerformanceCounter(&start);

	for(size_t i=0;i<1000;i++)
	{
		calcDervs<<<grid,block,0,stream1>>>(RSImage->d_image,RSImage->d_du,RSImage->d_dv,RSImage->d_x_off,RSImage->d_y_off,RSImage->d_width, RSImage->d_height, RSImage->d_ROIwidth, RSImage->d_ROIheight);
		calcDervsF<<<grid,block,0,stream2>>>(RSImage->d_du,RSImage->d_duF,RSImage->d_dv,RSImage->d_dvF, RSImage->d_ROIwidth, RSImage->d_ROIheight,RSImage->d_smw, RSImage->d_smmw, RSImage->d_sw, RSImage->d_smbw, RSImage->d_sbw);
		calcGrads<<<grid,block,0,stream3>>>(RSImage->d_duF, RSImage->d_dvF, RSImage->d_ROIwidth, RSImage->d_ROIheight, RSImage->d_x_c, RSImage->d_y_c, RSImage->d_x_c_old, RSImage->d_y_c_old, RSImage->d_sw, RSImage->d_smmw, RSImage->d_smw, RSImage->d_smbw, RSImage->d_sbw);
		//calcCenter<<<1,1>>>(RSImage->d_x_c, RSImage->d_y_c, RSImage->d_x_c_old, RSImage->d_y_c_old, RSImage->d_det, RSImage->d_sw, RSImage->d_smmw, RSImage->d_smw, RSImage->d_smbw, RSImage->d_sbw);
	}
	cudaDeviceSynchronize();

	QueryPerformanceCounter(&end);

	printf("%lf or %f, %f clokcs for Multistream %dx%d CUDA\n",(end.QuadPart - start.QuadPart)/(double)1000 / (double)frequency.QuadPart,float( clock () - begin_time ) /  CLOCKS_PER_SEC/(double)1000,float( clock () - begin_time ) , ROI_width, ROI_height);
	
	cudaDeviceSynchronize();
	begin_time = clock();
	QueryPerformanceCounter(&start);

	image[0]=10;
	printf("\n Pinned 0=%d",image[0]);
	for(size_t i=0;i<1000;i++)
	{
		//cudaMemcpy(di,RSImage->d_du,(ROI_width-1)*(ROI_height-1)*sizeof*di,cudaMemcpyDeviceToHost);
		cudaMemcpyAsync(RSImage->d_image,image,width*height*sizeof*RSImage->d_image,cudaMemcpyHostToDevice,stream1);
	}
	cudaStreamSynchronize(stream1);
	QueryPerformanceCounter(&end);

	printf("Host to device, %lf or %f, %f clokcs for %dx%d CUDA\n",(end.QuadPart - start.QuadPart)/(double)1000 / (double)frequency.QuadPart,float( clock () - begin_time ) /  CLOCKS_PER_SEC/(double)1000,float( clock () - begin_time ) , ROI_width, ROI_height);

	begin_time = clock();
	QueryPerformanceCounter(&start);

	float sum_t=0;
	for(size_t i=0;i<(1<<24);i++)
	{
		sum_t=0;
		for(size_t y=0;y<ROI_height-1;y++)
			for(size_t x=0;x<ROI_width-1;x++)
				sum_t=di[y*(ROI_width-1)+x];
	}
	QueryPerformanceCounter(&end);

	printf("Summation speed %lf or %f, %f clokcs for %dx%d CUDA\n",(end.QuadPart - start.QuadPart)/(double)1000 / (double)frequency.QuadPart,float( clock () - begin_time ) /  CLOCKS_PER_SEC/(double)1000,float( clock () - begin_time ) , ROI_width, ROI_height);
	

	cudaFreeHost(image);
	free(charImage);
	imageFile.close();

	freeRSImage(RSImage);
	free(RSImage);
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
