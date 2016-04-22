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

#pragma once
#ifndef __RSIMAGE_GPU_H__
#define __RSIMAGE_GPU_H__

#define Coord(RSImage,j,i)		*RSImage->d_width*j+i
#define DCoord(RSImage,j,i)		(*RSImage->d_ROIwidth-1)*j+i


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include <stdio.h>

struct RSImage_GPU{
	uint8_t* d_image;	//an acquired image
	size_t *d_width, *d_height; //the acquired image dimension
	size_t h_width, h_height;

	float *d_du,*d_dv,*d_duF,*d_dvF,*d_grads; //derivative images of ROI
	size_t *d_ROIwidth, *d_ROIheight; //ROI to analysis
	size_t h_ROIwidth, h_ROIheight;
	size_t *d_x_off, *d_y_off; //ROI offset in the image

	float *d_x_c, *d_y_c;	//the center
	float *d_x_c_old, *d_y_c_old; //the previously calculated center
	

};


__global__ void calcDervs(RSImage_GPU* RSImage)
{
	// ROI coordinates
	int i=blockDim.x*blockIdx.x+threadIdx.x;
	int j=blockDim.y*blockIdx.y+threadIdx.y;

	// Image coordinates
	int abs_i=i+(*RSImage->d_x_off);
	int abs_j=j+(*RSImage->d_y_off);

	if (i<*RSImage->d_width-1 && j<*RSImage->d_height-1) //dervs images have width-1 X height-1 dimension
	{
		RSImage->d_du[DCoord(RSImage,j,i)]=float(RSImage->d_image[Coord(RSImage,abs_j,abs_i+1)])-float(RSImage->d_image[Coord(RSImage,abs_j+1,abs_i)]);
		RSImage->d_dv[DCoord(RSImage,j,i)]=float(RSImage->d_image[Coord(RSImage,abs_j,abs_i)])-float(RSImage->d_image[Coord(RSImage,abs_j+1,abs_i+1)]);
	}
};

__global__ void calcDervsF(RSImage_GPU* RSImage)
{
	int i=blockDim.x*blockIdx.x+threadIdx.x;
	int j=blockDim.y*blockIdx.y+threadIdx.y;

	//Smoothing. 3x3 average. Not a boundary pixel
	if (i>1 && i<*RSImage->d_ROIwidth-2 && j>1 && j<*RSImage->d_ROIheight-1)
	{
			RSImage->d_duF[DCoord(RSImage,j,i)]=(	RSImage->d_du[DCoord(RSImage,j-1,i-1)]+	RSImage->d_du[DCoord(RSImage,j-1,i)]+	RSImage->d_du[DCoord(RSImage,j-1,i+1)]+ \
													RSImage->d_du[DCoord(RSImage,j,i-1)]+	RSImage->d_du[DCoord(RSImage,j,i)]+		RSImage->d_du[DCoord(RSImage,j,i+1)]+ \
													RSImage->d_du[DCoord(RSImage,j+1,i-1)]+	RSImage->d_du[DCoord(RSImage,j+1,i)]+	RSImage->d_du[DCoord(RSImage,j+1,i+1)])/9;
			RSImage->d_dvF[DCoord(RSImage,j,i)]=(	RSImage->d_dv[DCoord(RSImage,j-1,i-1)]+	RSImage->d_dv[DCoord(RSImage,j-1,i)]+	RSImage->d_dv[DCoord(RSImage,j-1,i+1)]+ \
													RSImage->d_dv[DCoord(RSImage,j,i-1)]+	RSImage->d_dv[DCoord(RSImage,j,i)]+		RSImage->d_dv[DCoord(RSImage,j,i+1)]+ \
													RSImage->d_dv[DCoord(RSImage,j+1,i-1)]+	RSImage->d_dv[DCoord(RSImage,j+1,i)]+	RSImage->d_dv[DCoord(RSImage,j+1,i+1)])/9;
			return;
	}
	

	//Smoothing edges
	if (j==0)
	{
		if(i==0) //left upper corner
		{
			RSImage->d_duF[DCoord(RSImage,0,0)]=(RSImage->d_du[DCoord(RSImage,0,0)]+	RSImage->d_du[DCoord(RSImage,0,1)]+	RSImage->d_du[DCoord(RSImage,1,0)]+	RSImage->d_du[DCoord(RSImage,1,1)])/4;
			RSImage->d_dvF[DCoord(RSImage,0,0)]=(RSImage->d_dv[DCoord(RSImage,0,0)]+	RSImage->d_dv[DCoord(RSImage,0,1)]+	RSImage->d_dv[DCoord(RSImage,1,0)]+	RSImage->d_dv[DCoord(RSImage,1,1)])/4;
			return;
		}

		if(i==*RSImage->d_ROIwidth-2) // right upper corner
		{
			RSImage->d_duF[DCoord(RSImage,0,i)]=(RSImage->d_du[DCoord(RSImage,0,i-1)]+	RSImage->d_du[DCoord(RSImage,0,i)]+	RSImage->d_du[DCoord(RSImage,1,i-1)]+	RSImage->d_du[DCoord(RSImage,1,i)])/4;
			RSImage->d_dvF[DCoord(RSImage,0,i)]=(RSImage->d_dv[DCoord(RSImage,0,i-1)]+	RSImage->d_dv[DCoord(RSImage,0,i)]+	RSImage->d_dv[DCoord(RSImage,1,i-1)]+	RSImage->d_dv[DCoord(RSImage,1,i)])/4;
			return;
		}

		//the top line
		RSImage->d_duF[DCoord(RSImage,j,i)]=(	RSImage->d_du[DCoord(RSImage,0,i-1)]+	RSImage->d_du[DCoord(RSImage,0,i)]+		RSImage->d_du[DCoord(RSImage,0,i+1)]+	\
												RSImage->d_du[DCoord(RSImage,0+1,i-1)]+	RSImage->d_du[DCoord(RSImage,0+1,i)]+	RSImage->d_du[DCoord(RSImage,0+1,i+1)])/6;
		RSImage->d_dvF[DCoord(RSImage,j,i)]=(	RSImage->d_dv[DCoord(RSImage,0,i-1)]+	RSImage->d_dv[DCoord(RSImage,0,i)]+		RSImage->d_dv[DCoord(RSImage,0,i+1)]+	\
												RSImage->d_dv[DCoord(RSImage,0+1,i-1)]+	RSImage->d_dv[DCoord(RSImage,0+1,i)]+	RSImage->d_dv[DCoord(RSImage,0+1,i+1)])/6;
		return;
	}

	if (j==*RSImage->d_ROIwidth-2)
	{
		if(i==0) //bottom left
		{
			RSImage->d_duF[DCoord(RSImage,j,0)]=(	RSImage->d_du[DCoord(RSImage,j-1,0)]+	RSImage->d_du[DCoord(RSImage,j-1,1)]+	\
													RSImage->d_du[DCoord(RSImage,j,0)]+		RSImage->d_du[DCoord(RSImage,j,1)])/4;
			RSImage->d_dvF[DCoord(RSImage,j,0)]=(	RSImage->d_dv[DCoord(RSImage,j-1,0)]+	RSImage->d_dv[DCoord(RSImage,j-1,1)]+	\
													RSImage->d_dv[DCoord(RSImage,j,0)]+		RSImage->d_dv[DCoord(RSImage,j,1)])/4;
			return;
		}

		if(i==*RSImage->d_ROIwidth-2) //bottom right
		{
			RSImage->d_duF[DCoord(RSImage,j,i)]=(	RSImage->d_du[DCoord(RSImage,j-1,i-1)]+	RSImage->d_du[DCoord(RSImage,j-1,i)]+		\
													RSImage->d_du[DCoord(RSImage,j,i-1)]+	RSImage->d_du[DCoord(RSImage,j,i)])/4;
			RSImage->d_dvF[DCoord(RSImage,j,i)]=(	RSImage->d_dv[DCoord(RSImage,j-1,i-1)]+	RSImage->d_dv[DCoord(RSImage,j-1,i)]+	\
													RSImage->d_dv[DCoord(RSImage,j-1,i-1)]+	RSImage->d_dv[DCoord(RSImage,j,i)])/4;
			return;
		}

		// the bottom line
		RSImage->d_duF[DCoord(RSImage,j,i)]=(	RSImage->d_du[DCoord(RSImage,j-1,i-1)]+		RSImage->d_du[DCoord(RSImage,j-1,i)]+	RSImage->d_du[DCoord(RSImage,j-1,i+1)]+	\
												RSImage->d_du[DCoord(RSImage,j,i-1)]+		RSImage->d_du[DCoord(RSImage,j,i)]+		RSImage->d_du[DCoord(RSImage,j,i+1)])/6;
		RSImage->d_dvF[DCoord(RSImage,j,i)]=(	RSImage->d_dv[DCoord(RSImage,j-1,i-1)]+		RSImage->d_dv[DCoord(RSImage,j-1,i)]+	RSImage->d_dv[DCoord(RSImage,j-1,i+1)]+	\
												RSImage->d_dv[DCoord(RSImage,j,i-1)]+		RSImage->d_dv[DCoord(RSImage,j,i)]+		RSImage->d_dv[DCoord(RSImage,j,i+1)])/6;
		return;
	}

	if(i==0) //the left line
	{
		RSImage->d_duF[DCoord(RSImage,j,0)]=(	RSImage->d_du[DCoord(RSImage,j-1,0)]+	RSImage->d_du[DCoord(RSImage,j-1,0+1)]+ \
												RSImage->d_du[DCoord(RSImage,j,0)]+		RSImage->d_du[DCoord(RSImage,j,0+1)]+ \
												RSImage->d_du[DCoord(RSImage,j+1,0)]+	RSImage->d_du[DCoord(RSImage,j+1,0+1)])/6;
		RSImage->d_dvF[DCoord(RSImage,j,0)]=(	RSImage->d_dv[DCoord(RSImage,j-1,0)]+	RSImage->d_dv[DCoord(RSImage,j-1,0+1)]+ \
												RSImage->d_dv[DCoord(RSImage,j,0)]+		RSImage->d_dv[DCoord(RSImage,j,0+1)]+ \
												RSImage->d_dv[DCoord(RSImage,j+1,0)]+	RSImage->d_dv[DCoord(RSImage,j+1,0+1)])/6;
		return;
	}

	//the right line
	RSImage->d_duF[DCoord(RSImage,j,i)]=(	RSImage->d_du[DCoord(RSImage,j-1,i-1)]+		RSImage->d_du[DCoord(RSImage,j-1,i)]+	\
											RSImage->d_du[DCoord(RSImage,j,i-1)]+		RSImage->d_du[DCoord(RSImage,j,i)]+		\
											RSImage->d_du[DCoord(RSImage,j+1,i-1)]+		RSImage->d_du[DCoord(RSImage,j+1,i)])/6;
	RSImage->d_dvF[DCoord(RSImage,j,i)]=(	RSImage->d_dv[DCoord(RSImage,j-1,i-1)]+		RSImage->d_dv[DCoord(RSImage,j-1,i)]+	\
											RSImage->d_dv[DCoord(RSImage,j,i-1)]+		RSImage->d_dv[DCoord(RSImage,j,i)]+		\
											RSImage->d_dv[DCoord(RSImage,j+1,i-1)]+		RSImage->d_dv[DCoord(RSImage,j+1,i)])/6;
};
__global__ void calcGrads(RSImage_GPU* RSImage)
{
};
__global__ void calcCenter(RSImage_GPU* RSImage)
{
};


__global__ void updateCenter(RSImage_GPU * RSImage){
	//Due to an unknown reason, __device__ function calling does not work properly.
	//So I put all the codes into a kernel

};

void initRSImage(RSImage_GPU *RSImage,size_t width, size_t height, size_t ROIwidth, size_t ROIheight, size_t x_off, size_t y_off)
{
	//host image dimensions for cudaMalloc
	RSImage->h_width=width;
	RSImage->h_height=height;
	RSImage->h_ROIwidth=ROIwidth;
	RSImage->h_ROIheight=ROIheight;
	
	//Device image dimensions
	cudaMalloc(&RSImage->d_width,sizeof*RSImage->d_width);
	cudaMalloc(&RSImage->d_height,sizeof*RSImage->d_height);
	cudaMalloc(&RSImage->d_ROIwidth,sizeof*RSImage->d_ROIwidth);
	cudaMalloc(&RSImage->d_ROIheight,sizeof*RSImage->d_ROIheight);

	cudaMemcpy(RSImage->d_width,&width,sizeof*RSImage->d_width,cudaMemcpyHostToDevice);
	cudaMemcpy(RSImage->d_height,&height,sizeof*RSImage->d_height,cudaMemcpyHostToDevice);
	cudaMemcpy(RSImage->d_ROIwidth,&ROIwidth,sizeof*RSImage->d_ROIwidth,cudaMemcpyHostToDevice);
	cudaMemcpy(RSImage->d_ROIheight,&ROIheight,sizeof*RSImage->d_ROIheight,cudaMemcpyHostToDevice);

	cudaMalloc(&RSImage->d_x_off,sizeof*RSImage->d_x_off);
	cudaMalloc(&RSImage->d_y_off,sizeof*RSImage->d_y_off);

	cudaMemcpy(RSImage->d_x_off,&x_off,sizeof*RSImage->d_x_off,cudaMemcpyHostToDevice);
	cudaMemcpy(RSImage->d_y_off,&y_off,sizeof*RSImage->d_y_off,cudaMemcpyHostToDevice);

	//Center poisitions
	cudaMalloc(&RSImage->d_x_c,sizeof*RSImage->d_x_c);
	cudaMalloc(&RSImage->d_y_c,sizeof*RSImage->d_y_c);
	cudaMalloc(&RSImage->d_x_c_old,sizeof*RSImage->d_x_c_old);
	cudaMalloc(&RSImage->d_y_c_old,sizeof*RSImage->d_y_c_old);

	//device images
	cudaMalloc(&RSImage->d_image,RSImage->h_width*RSImage->h_height*sizeof*RSImage->d_image);

	cudaMalloc(&RSImage->d_du,(RSImage->h_ROIwidth-1)*(RSImage->h_ROIheight-1)*sizeof*RSImage->d_du);
	cudaMalloc(&RSImage->d_dv,(RSImage->h_ROIwidth-1)*(RSImage->h_ROIheight-1)*sizeof*RSImage->d_dv);
	cudaMalloc(&RSImage->d_duF,(RSImage->h_ROIwidth-1)*(RSImage->h_ROIheight-1)*sizeof*RSImage->d_duF);
	cudaMalloc(&RSImage->d_dvF,(RSImage->h_ROIwidth-1)*(RSImage->h_ROIheight-1)*sizeof*RSImage->d_dvF);
	cudaMalloc(&RSImage->d_grads,(RSImage->h_ROIwidth-1)*(RSImage->h_ROIheight-1)*sizeof*RSImage->d_grads);
};

void freeRSImage(RSImage_GPU *RSImage)
{
	//Device image dimensions
	cudaFree(RSImage->d_width);
	cudaFree(RSImage->d_height);
	cudaFree(RSImage->d_ROIwidth);
	cudaFree(RSImage->d_ROIheight);

	cudaFree(&RSImage->d_x_off);
	cudaFree(&RSImage->d_y_off);

	//Center poisitions
	cudaFree(RSImage->d_x_c);
	cudaFree(RSImage->d_y_c);
	cudaFree(RSImage->d_x_c_old);
	cudaFree(RSImage->d_y_c_old);

	//device images
	cudaFree(RSImage->d_image);

	cudaFree(RSImage->d_du);
	cudaFree(RSImage->d_dv);
	cudaFree(RSImage->d_duF);
	cudaFree(RSImage->d_dvF);
	cudaFree(RSImage->d_grads);
};
#endif