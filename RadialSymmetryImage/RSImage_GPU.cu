/*

Project		:	RSImage_GPU
Description	:	GPU implementation of radial symmetry center finding method.
				
				This is a C/C++ translation of the original work,
				<doi:10.1038/nmeth.2071>.

*

Copyright (C) 2016 Janghyun Yoo

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "RSImage_GPU.h"

#include <stdlib.h>
#include <cmath>
#include <stdio.h>

#include <Windows.h>
#include <ctime>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <stdint.h>
#include <stdio.h>

#define StackSize 32

using namespace std;

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort /* =true */)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void calcDervs(uint8_t *d_image, float *d_du, float *d_dv, size_t *d_x_off, size_t *d_y_off,  size_t *d_width, size_t *d_height, size_t *d_ROIwidth, size_t *d_ROIheight)
{
	// ROI coordinates
	int i=blockDim.x*blockIdx.x+threadIdx.x;
	int j=blockDim.y*blockIdx.y+threadIdx.y;

	if(i>*d_ROIwidth-2 || j>*d_ROIheight-2)  return;

	// Image coordinates
	int abs_i=i+(*d_x_off);
	int abs_j=j+(*d_y_off);


	d_du[(*d_ROIwidth-1)*j+i]=float(d_image[*d_width*(abs_j)+abs_i+1])-float(d_image[*d_width*(abs_j+1)+abs_i]);
	d_dv[(*d_ROIwidth-1)*j+i]=float(d_image[*d_width*(abs_j)+abs_i])-float(d_image[*d_width*(abs_j+1)+abs_i+1]);

}

__global__ void calcDervsF(float *d_du, float *d_duF, float *d_dv, float *d_dvF, size_t *d_ROIwidth, size_t *d_ROIheight, float *d_sw)
{
	int i=blockDim.x*blockIdx.x+threadIdx.x;
	int j=blockDim.y*blockIdx.y+threadIdx.y;


	if(i>*d_ROIwidth-2 || j>*d_ROIheight-2)
	{
		return;
	}

	d_sw[((*d_ROIwidth-1)*j+i)/StackSize]=0;
	d_sw[(*d_ROIwidth)*(*d_ROIheight)*1/StackSize+((*d_ROIwidth-1)*j+i)/StackSize]=0;
	d_sw[(*d_ROIwidth)*(*d_ROIheight)*2/StackSize+((*d_ROIwidth-1)*j+i)/StackSize]=0;
	d_sw[(*d_ROIwidth)*(*d_ROIheight)*3/StackSize+((*d_ROIwidth-1)*j+i)/StackSize]=0;
	d_sw[(*d_ROIwidth)*(*d_ROIheight)*4/StackSize+((*d_ROIwidth-1)*j+i)/StackSize]=0;

	//Smoothing. 3x3 average. Not a boundary pixel
	if (i>0 && i<*d_ROIwidth-2 && j>0 && j<*d_ROIheight-2)
	{
			d_duF[(*d_ROIwidth-1)*j+i]=(	d_du[(*d_ROIwidth-1)*(j-1)+i-1]+	d_du[(*d_ROIwidth-1)*(j-1)+i]+	d_du[(*d_ROIwidth-1)*(j-1)+i+1]+ \
													d_du[(*d_ROIwidth-1)*j+i-1]+	d_du[(*d_ROIwidth-1)*j+i]+		d_du[(*d_ROIwidth-1)*j+i+1]+ \
													d_du[(*d_ROIwidth-1)*(j+1)+i-1]+	d_du[(*d_ROIwidth-1)*(j+1)+i]+	d_du[(*d_ROIwidth-1)*(j+1)+i+1])/9;
			d_dvF[(*d_ROIwidth-1)*j+i]=(	d_dv[(*d_ROIwidth-1)*(j-1)+i-1]+	d_dv[(*d_ROIwidth-1)*(j-1)+i]+	d_dv[(*d_ROIwidth-1)*(j-1)+i+1]+ \
													d_dv[(*d_ROIwidth-1)*j+i-1]+	d_dv[(*d_ROIwidth-1)*j+i]+		d_dv[(*d_ROIwidth-1)*j+i+1]+ \
													d_dv[(*d_ROIwidth-1)*(j+1)+i-1]+	d_dv[(*d_ROIwidth-1)*(j+1)+i]+	d_dv[(*d_ROIwidth-1)*(j+1)+i+1])/9;
			return;
	}
	
	//option2: no smoothing edges
	/*
	d_duF[(*d_ROIwidth-1)*j+i]=d_du[(*d_ROIwidth-1)*j+i];
	d_dvF[(*d_ROIwidth-1)*j+i]=d_dv[(*d_ROIwidth-1)*j+i];

	return;
	*/

	//Smoothing edges
	if (j==0)
	{
		if(i==0) //left upper corner
		{
			d_duF[0]=(d_du[0]+	d_du[1]+	d_du[(*d_ROIwidth-1)*1+0]+	d_du[(*d_ROIwidth-1)*1+1])/4;
			d_dvF[0]=(d_dv[0]+	d_dv[1]+	d_dv[(*d_ROIwidth-1)*1+0]+	d_dv[(*d_ROIwidth-1)*1+1])/4;
			return;
		}

		if(i==*d_ROIwidth-2) // right upper corner
		{
			d_duF[i]=(d_du[i-1]+	d_du[i]+	d_du[(*d_ROIwidth-1)*1+i-1]+	d_du[(*d_ROIwidth-1)*1+i])/4;
			d_dvF[i]=(d_dv[i-1]+	d_dv[i]+	d_dv[(*d_ROIwidth-1)*1+i-1]+	d_dv[(*d_ROIwidth-1)*1+i])/4;
			return;
		}

		//the top line
		if(i<*d_ROIwidth-1)
		{
			d_duF[(*d_ROIwidth-1)*j+i]=(	d_du[i-1]+	d_du[i]+		d_du[i+1]+	\
													d_du[(*d_ROIwidth-1)*1+i-1]+	d_du[(*d_ROIwidth-1)*1+i]+	d_du[(*d_ROIwidth-1)*1+i+1])/6;
			d_dvF[(*d_ROIwidth-1)*j+i]=(	d_dv[i-1]+	d_dv[i]+		d_dv[i+1]+	\
													d_dv[(*d_ROIwidth-1)*1+i-1]+	d_dv[(*d_ROIwidth-1)*1+i]+	d_dv[(*d_ROIwidth-1)*1+i+1])/6;
			return;
		}
	}

	if (j==*d_ROIwidth-2)
	{
		if(i==0) //bottom left
		{
			d_duF[(*d_ROIwidth-1)*(j)+(0)]=(	d_du[(*d_ROIwidth-1)*(j-1)+(0)]+	d_du[(*d_ROIwidth-1)*(j-1)+(1)]+	\
													d_du[(*d_ROIwidth-1)*(j)+(0)]+		d_du[(*d_ROIwidth-1)*(j)+(1)])/4;
			d_dvF[(*d_ROIwidth-1)*(j)+(0)]=(	d_dv[(*d_ROIwidth-1)*(j-1)+(0)]+	d_dv[(*d_ROIwidth-1)*(j-1)+(1)]+	\
													d_dv[(*d_ROIwidth-1)*(j)+(0)]+		d_dv[(*d_ROIwidth-1)*(j)+(1)])/4;
			return;
		}

		if(i==*d_ROIwidth-2) //bottom right
		{
			d_duF[(*d_ROIwidth-1)*j+i]=(	d_du[(*d_ROIwidth-1)*(j-1)+i-1]+	d_du[(*d_ROIwidth-1)*(j-1)+i]+		\
													d_du[(*d_ROIwidth-1)*j+i-1]+	d_du[(*d_ROIwidth-1)*j+i])/4;
			d_dvF[(*d_ROIwidth-1)*j+i]=(	d_dv[(*d_ROIwidth-1)*(j-1)+i-1]+	d_dv[(*d_ROIwidth-1)*(j-1)+i]+	\
													d_dv[(*d_ROIwidth-1)*(j-1)+i-1]+	d_dv[(*d_ROIwidth-1)*j+i])/4;
			return;
		}

		if(i<*d_ROIwidth-1)// the bottom line
		{
			d_duF[(*d_ROIwidth-1)*j+i]=(	d_du[(*d_ROIwidth-1)*(j-1)+i-1]+		d_du[(*d_ROIwidth-1)*(j-1)+i]+	d_du[(*d_ROIwidth-1)*(j-1)+i+1]+	\
												d_du[(*d_ROIwidth-1)*j+i-1]+		d_du[(*d_ROIwidth-1)*j+i]+		d_du[(*d_ROIwidth-1)*j+i+1])/6;
			d_dvF[(*d_ROIwidth-1)*j+i]=(	d_dv[(*d_ROIwidth-1)*(j-1)+i-1]+		d_dv[(*d_ROIwidth-1)*(j-1)+i]+	d_dv[(*d_ROIwidth-1)*(j-1)+i+1]+	\
												d_dv[(*d_ROIwidth-1)*j+i-1]+		d_dv[(*d_ROIwidth-1)*j+i]+		d_dv[(*d_ROIwidth-1)*j+i+1])/6;
			return;
		}
	}

	if(i==0) //the left line
	{
		d_duF[(*d_ROIwidth-1)*(j)+(0)]=(	d_du[(*d_ROIwidth-1)*(j-1)+(0)]+	d_du[(*d_ROIwidth-1)*(j-1)+(1)]+ \
												d_du[(*d_ROIwidth-1)*(j)+(0)]+		d_du[(*d_ROIwidth-1)*(j)+(1)]+ \
												d_du[(*d_ROIwidth-1)*(j+1)+(0)]+	d_du[(*d_ROIwidth-1)*(j+1)+(1)])/6;
		d_dvF[(*d_ROIwidth-1)*(j)+(0)]=(	d_dv[(*d_ROIwidth-1)*(j-1)+(0)]+	d_dv[(*d_ROIwidth-1)*(j-1)+(1)]+ \
												d_dv[(*d_ROIwidth-1)*(j)+(0)]+		d_dv[(*d_ROIwidth-1)*(j)+(1)]+ \
												d_dv[(*d_ROIwidth-1)*(j+1)+(0)]+	d_dv[(*d_ROIwidth-1)*(j+1)+(1)])/6;
		return;
	}

	if(i<*d_ROIwidth-1 && j<*d_ROIheight-1)//the right line
	{
		d_duF[(*d_ROIwidth-1)*j+i]=(	d_du[(*d_ROIwidth-1)*(j-1)+i-1]+		d_du[(*d_ROIwidth-1)*(j-1)+i]+	\
												d_du[(*d_ROIwidth-1)*j+i-1]+		d_du[(*d_ROIwidth-1)*j+i]+		\
												d_du[(*d_ROIwidth-1)*(j+1)+i-1]+		d_du[(*d_ROIwidth-1)*(j+1)+i])/6;
		d_dvF[(*d_ROIwidth-1)*j+i]=(	d_dv[(*d_ROIwidth-1)*(j-1)+i-1]+		d_dv[(*d_ROIwidth-1)*(j-1)+i]+	\
												d_dv[(*d_ROIwidth-1)*j+i-1]+		d_dv[(*d_ROIwidth-1)*j+i]+		\
											d_dv[(*d_ROIwidth-1)*(j+1)+i-1]+		d_dv[(*d_ROIwidth-1)*(j+1)+i])/6;
		return;
	}
}

__global__ void calcGrads(float *d_duF, float *d_dvF, size_t *d_ROIwidth, size_t *d_ROIheight, float *d_x_c_old, float *d_y_c_old, float *sw)
{
	int i=blockDim.x*blockIdx.x+threadIdx.x;
	int j=blockDim.y*blockIdx.y+threadIdx.y;

	if(i>(*d_ROIwidth-2) || j>(*d_ROIheight-2)) return;

	float gradDenominator=d_duF[(*d_ROIwidth-1)*j+i]-d_dvF[(*d_ROIwidth-1)*j+i];
	float m=1, b=0;

	if (gradDenominator==0.f)
		m=-(gradDenominator+2*d_dvF[(*d_ROIwidth-1)*j+i])/minF;
	else
		m=-(gradDenominator+2*d_dvF[(*d_ROIwidth-1)*j+i])/gradDenominator;

	float gridY=-float(*d_ROIheight-1)/2.0f+0.5f+float(j);
	float gridX=-float(*d_ROIwidth-1)/2.0f+0.5f+float(i);
	float gradMag=d_duF[(*d_ROIwidth-1)*j+i]*d_duF[(*d_ROIwidth-1)*j+i]+d_dvF[(*d_ROIwidth-1)*j+i]*d_dvF[(*d_ROIwidth-1)*j+i];

	b=gridY-m*gridX;

	float wm2p1=gradMag/sqrt((gridX-*d_x_c_old)*(gridX-*d_x_c_old)+(gridY-*d_y_c_old)*(gridY-*d_y_c_old))/(m*m+1);

	/*
	sw[(*d_ROIwidth-1)*j+i]=wm2p1;
	smmw[(*d_ROIwidth-1)*j+i]=m*m*wm2p1;
	smw[(*d_ROIwidth-1)*j+i]=m*wm2p1;
	smbw[(*d_ROIwidth-1)*j+i]=m*b*wm2p1;
	sbw[(*d_ROIwidth-1)*j+i]=b*wm2p1;
	*/

	atomicAdd(&sw[((*d_ROIwidth-1)*j+i)/StackSize],wm2p1); //sw
	atomicAdd(&sw[(*d_ROIwidth)*(*d_ROIheight)/StackSize+((*d_ROIwidth-1)*j+i)/StackSize],m*m*wm2p1); //smmw
	atomicAdd(&sw[(*d_ROIwidth)*(*d_ROIheight)*2/StackSize+((*d_ROIwidth-1)*j+i)/StackSize],m*wm2p1); //smw
	atomicAdd(&sw[(*d_ROIwidth)*(*d_ROIheight)*3/StackSize+((*d_ROIwidth-1)*j+i)/StackSize],m*b*wm2p1); //smbw
	atomicAdd(&sw[(*d_ROIwidth)*(*d_ROIheight)*4/StackSize+((*d_ROIwidth-1)*j+i)/StackSize],b*wm2p1); //sbw
}

void calcCenter(RSImage_GPU * RSImage, cudaStream_t stream)
{
	//init variables
	RSImage->sw=0;
	RSImage->smmw=0;
	RSImage->smw=0;
	RSImage->smbw=0;
	RSImage->sbw=0;

	//order = {sw, smmw, smw, smbw, sbw}
	for(size_t i=0;i<((RSImage->ROIwidth-1)*(RSImage->ROIheight-1)-1)/StackSize+1;i++)
	{
		RSImage->sw+=RSImage->h_sw[i];
	}
	for(size_t i=0;i<((RSImage->ROIwidth-1)*(RSImage->ROIheight-1)-1)/StackSize+1;i++)
	{
		RSImage->smmw+=RSImage->h_sw[(RSImage->ROIwidth)*(RSImage->ROIheight)/StackSize+i];
	}
	for(size_t i=0;i<((RSImage->ROIwidth-1)*(RSImage->ROIheight-1)-1)/StackSize+1;i++)
	{
		RSImage->smw+=RSImage->h_sw[(RSImage->ROIwidth)*(RSImage->ROIheight)/StackSize*2+i];
	}
	for(size_t i=0;i<((RSImage->ROIwidth-1)*(RSImage->ROIheight-1)-1)/StackSize+1;i++)
	{
		RSImage->smbw+=RSImage->h_sw[(RSImage->ROIwidth)*(RSImage->ROIheight)/StackSize*3+i];
	}
	for(size_t i=0;i<((RSImage->ROIwidth-1)*(RSImage->ROIheight-1)-1)/StackSize+1;i++)
	{
		RSImage->sbw+=RSImage->h_sw[(RSImage->ROIwidth)*(RSImage->ROIheight)/StackSize*4+i];
	}

	RSImage->det=RSImage->smw*RSImage->smw-RSImage->smmw*RSImage->sw;
	RSImage->h_x_c=(RSImage->smbw*RSImage->sw-RSImage->smw*RSImage->sbw)/RSImage->det;
	RSImage->h_y_c=(RSImage->smbw*RSImage->smw-RSImage->smmw*RSImage->sbw)/RSImage->det;

	//update the previous center
	// should relay the position to the next RSImage

	//cudaCheck(cudaMemcpyAsync(RSImage->d_x_c_old,&RSImage->h_x_c,sizeof*RSImage->d_x_c_old,cudaMemcpyHostToDevice,stream));
	//cudaCheck(cudaMemcpyAsync(RSImage->d_y_c_old,&RSImage->h_y_c,sizeof*RSImage->d_y_c_old,cudaMemcpyHostToDevice,stream));
}


void updateRSImageCenter(RSImage_GPU * RSImage, dim3 grid, dim3 block,cudaStream_t stream){
	calcDervs<<<grid,block,0,stream>>>(RSImage->d_image,RSImage->d_du,RSImage->d_dv,RSImage->d_x_off,RSImage->d_y_off,RSImage->d_width, RSImage->d_height, RSImage->d_ROIwidth, RSImage->d_ROIheight);
	calcDervsF<<<grid,block,0,stream>>>(RSImage->d_du,RSImage->d_duF,RSImage->d_dv,RSImage->d_dvF, RSImage->d_ROIwidth, RSImage->d_ROIheight, RSImage->d_sw);
	calcGrads<<<grid,block,0,stream>>>(RSImage->d_duF, RSImage->d_dvF, RSImage->d_ROIwidth, RSImage->d_ROIheight, RSImage->d_x_c_old, RSImage->d_y_c_old, RSImage->d_sw);
}

//transfer image from host to device
void transferRSImageHtoD(RSImage_GPU * RSImage, cudaStream_t stream)
{
	cudaCheck(cudaMemcpyAsync(RSImage->d_image,RSImage->h_image,RSImage->width*RSImage->height*sizeof*RSImage->d_image,cudaMemcpyHostToDevice,stream));
} 

//transfer derivative images from device to host
void transferRSImageDtoH(RSImage_GPU * RSImage, cudaStream_t stream)
{
	cudaCheck(cudaMemcpyAsync(RSImage->h_sw,RSImage->d_sw,(RSImage->ROIwidth)*(RSImage->ROIheight)*sizeof*RSImage->d_sw/StackSize*5,cudaMemcpyDeviceToHost,stream));
	//cudaCheck(cudaMemcpyAsync(RSImage->h_smmw,RSImage->d_smmw,(RSImage->ROIwidth-1)*(RSImage->ROIheight-1)*sizeof*RSImage->d_smmw,cudaMemcpyDeviceToHost,stream));
	//cudaCheck(cudaMemcpyAsync(RSImage->h_smw,RSImage->d_smw,(RSImage->ROIwidth-1)*(RSImage->ROIheight-1)*sizeof*RSImage->h_smw,cudaMemcpyDeviceToHost,stream));
	//cudaCheck(cudaMemcpyAsync(RSImage->h_smbw,RSImage->d_smbw,(RSImage->ROIwidth-1)*(RSImage->ROIheight-1)*sizeof*RSImage->d_smbw,cudaMemcpyDeviceToHost,stream));
	//cudaCheck(cudaMemcpyAsync(RSImage->h_sbw,RSImage->d_sbw,(RSImage->ROIwidth-1)*(RSImage->ROIheight-1)*sizeof*RSImage->d_sbw,cudaMemcpyDeviceToHost,stream));
}

void initRSImage(RSImage_GPU *RSImage, uint8_t *h_image, size_t width,size_t height, size_t ROIwidth, size_t ROIheight, size_t x_off, size_t y_off, float x_c_old/* =0.1f */, float y_c_old /* =0.1f */)
{
	//bind image
	RSImage->h_image=h_image;
	RSImage->ROIwidth=ROIwidth;
	RSImage->ROIheight=ROIheight;
	RSImage->width=width;
	RSImage->height=height;

	//the image info
	cudaCheck(cudaMalloc(&RSImage->d_width,sizeof*RSImage->d_width));
	cudaCheck(cudaMalloc(&RSImage->d_height,sizeof*RSImage->d_height));
	cudaCheck(cudaMalloc(&RSImage->d_ROIwidth,sizeof*RSImage->d_ROIwidth));
	cudaCheck(cudaMalloc(&RSImage->d_ROIheight,sizeof*RSImage->d_ROIheight));

	cudaCheck(cudaMemcpy(RSImage->d_width,&width,sizeof(width),cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(RSImage->d_height,&height,sizeof(height),cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(RSImage->d_ROIwidth,&ROIwidth,sizeof(ROIwidth),cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(RSImage->d_ROIheight,&ROIheight,sizeof(ROIheight),cudaMemcpyHostToDevice));

	//bead position
	cudaCheck(cudaMalloc((void**)&RSImage->d_x_off,sizeof*RSImage->d_x_off));
	cudaCheck(cudaMalloc((void**)&RSImage->d_y_off,sizeof*RSImage->d_y_off));

	cudaCheck(cudaMemcpy(RSImage->d_x_off,&x_off,sizeof*RSImage->d_x_off,cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(RSImage->d_y_off,&y_off,sizeof*RSImage->d_y_off,cudaMemcpyHostToDevice));

	//device images
	cudaCheck(cudaMalloc((void**)&RSImage->d_image,width*height*sizeof*RSImage->d_image));
	cudaCheck(cudaMalloc((void**)&RSImage->d_du,(ROIwidth-1)*(ROIheight-1)*sizeof*RSImage->d_du));
	cudaCheck(cudaMalloc((void**)&RSImage->d_dv,(ROIwidth-1)*(ROIheight-1)*sizeof*RSImage->d_dv));
	cudaCheck(cudaMalloc((void**)&RSImage->d_duF,(ROIwidth-1)*(ROIheight-1)*sizeof*RSImage->d_duF));
	cudaCheck(cudaMalloc((void**)&RSImage->d_dvF,(ROIwidth-1)*(ROIheight-1)*sizeof*RSImage->d_dvF));
	cudaCheck(cudaMalloc((void**)&RSImage->d_grads,(ROIwidth-1)*(ROIheight-1)*sizeof*RSImage->d_grads));

	//host images for center calculation
	cudaCheck(cudaMallocHost(&RSImage->h_sw,(RSImage->ROIwidth)*(RSImage->ROIheight)*sizeof*RSImage->h_sw/StackSize*5));//merge 5 variables into an array
//	cudaCheck(cudaMallocHost(&RSImage->h_smw,(RSImage->ROIwidth-1)*(RSImage->ROIheight-1)*sizeof*RSImage->h_smw));
//	cudaCheck(cudaMallocHost(&RSImage->h_smmw,(RSImage->ROIwidth-1)*(RSImage->ROIheight-1)*sizeof*RSImage->h_smmw));
//	cudaCheck(cudaMallocHost(&RSImage->h_smbw,(RSImage->ROIwidth-1)*(RSImage->ROIheight-1)*sizeof*RSImage->h_smbw));
//	cudaCheck(cudaMallocHost(&RSImage->h_sbw,(RSImage->ROIwidth-1)*(RSImage->ROIheight-1)*sizeof*RSImage->h_sbw));

	//the previous center
	cudaCheck(cudaMalloc((void**)&RSImage->d_x_c_old,sizeof*RSImage->d_x_c_old))
	cudaCheck(cudaMalloc((void**)&RSImage->d_y_c_old,sizeof*RSImage->d_y_c_old));

	cudaCheck(cudaMemcpy(RSImage->d_x_c_old,&x_c_old,sizeof*RSImage->d_x_c_old,cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(RSImage->d_y_c_old,&y_c_old,sizeof*RSImage->d_y_c_old,cudaMemcpyHostToDevice));

	cudaCheck(cudaMalloc(&RSImage->d_sw,(RSImage->ROIwidth)*(RSImage->ROIheight)*sizeof*RSImage->d_sw/StackSize*5)); //merge 5 variables into an array
//	cudaCheck(cudaMalloc(&RSImage->d_smw,(RSImage->ROIwidth-1)*(RSImage->ROIheight-1)*sizeof*RSImage->d_smw));
//	cudaCheck(cudaMalloc(&RSImage->d_smmw,(RSImage->ROIwidth-1)*(RSImage->ROIheight-1)*sizeof*RSImage->d_smmw));
//	cudaCheck(cudaMalloc(&RSImage->d_smbw,(RSImage->ROIwidth-1)*(RSImage->ROIheight-1)*sizeof*RSImage->d_smbw));
//	cudaCheck(cudaMalloc(&RSImage->d_sbw,(RSImage->ROIwidth-1)*(RSImage->ROIheight-1)*sizeof*RSImage->d_sbw));
}

void freeRSImage(RSImage_GPU *RSImage)
{
	//the image info
	cudaCheck(cudaFree(RSImage->d_width));
	cudaCheck(cudaFree(RSImage->d_height));
	cudaCheck(cudaFree(RSImage->d_ROIwidth));
	cudaCheck(cudaFree(RSImage->d_ROIheight));
	//bead position
	cudaCheck(cudaFree(RSImage->d_x_off));
	cudaCheck(cudaFree(RSImage->d_y_off));

	//device images
	cudaCheck(cudaFree(RSImage->d_image));

	cudaCheck(cudaFree(RSImage->d_du));
	cudaCheck(cudaFree(RSImage->d_dv));
	cudaCheck(cudaFree(RSImage->d_duF));
	cudaCheck(cudaFree(RSImage->d_dvF));
	cudaCheck(cudaFree(RSImage->d_grads));

	//host derivative images
	cudaCheck(cudaFreeHost(RSImage->h_sw));
/*	cudaCheck(cudaFreeHost(RSImage->h_smw));
	cudaCheck(cudaFreeHost(RSImage->h_smmw));
	cudaCheck(cudaFreeHost(RSImage->h_smbw));
	cudaCheck(cudaFreeHost(RSImage->h_sbw));	*/

	//the previous center
	cudaCheck(cudaFree(RSImage->d_x_c_old));
	cudaCheck(cudaFree(RSImage->d_y_c_old));

	cudaCheck(cudaFree(RSImage->d_sw));
/*	cudaCheck(cudaFree(RSImage->d_smmw));
	cudaCheck(cudaFree(RSImage->d_smw));
	cudaCheck(cudaFree(RSImage->d_smbw));
	cudaCheck(cudaFree(RSImage->d_sbw)); */
}