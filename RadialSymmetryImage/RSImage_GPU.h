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

#define minF	0.00001001f

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <stdint.h>
#include <stdio.h>

using namespace std;

struct RSImage_GPU{
	uint8_t* d_image;	//an acquired image
	size_t *d_width, *d_height; //the acquired image dimension

	float *d_du,*d_dv,*d_duF,*d_dvF,*d_grads; //derivative images of ROI
	size_t *d_ROIwidth, *d_ROIheight; //ROI to analysis
	size_t *d_x_off, *d_y_off; //ROI offset in the image

	float *d_x_c, *d_y_c;	//the center
	float *d_x_c_old, *d_y_c_old; //the previously calculated center

	float *d_det, *d_smw,*d_smmw,*d_sw;
	float *d_smbw,*d_sbw;
};


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

};

__global__ void calcDervsF(float *d_du, float *d_duF, float *d_dv, float *d_dvF, size_t *d_ROIwidth, size_t *d_ROIheight, float *d_smw, float *d_smmw, float *d_sw, float *d_smbw, float *d_sbw)
{
	int i=blockDim.x*blockIdx.x+threadIdx.x;
	int j=blockDim.y*blockIdx.y+threadIdx.y;


	if(i>*d_ROIwidth-2)
	{
		if(j>*d_ROIheight-2)
		{
			*d_smw=0;
			*d_smmw=0;
			*d_sw=0;
			*d_smbw=0;
			*d_sbw=0;
			return;
		}
		return;
	}

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
	d_duF[(*d_ROIwidth-1)*j+i]=d_du[(*d_ROIwidth-1)*j+i];
	d_dvF[(*d_ROIwidth-1)*j+i]=d_dv[(*d_ROIwidth-1)*j+i];

	return;

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
};
__global__ void calcGrads(float *d_duF, float *d_dvF, size_t *d_ROIwidth, size_t *d_ROIheight, float *d_x_c, float *d_y_c, float *d_x_c_old, float *d_y_c_old, float *sw, float *smmw, float *smw, float *smbw, float *sbw)
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

	//atomicAdd(sw,wm2p1);
	//*sw+=wm2p1;
	//atomicAdd(smmw,m*m*wm2p1);
	//*smmw+=m*m*wm2p1;
	//atomicAdd(smw,m*wm2p1);
	//*smw+=m*wm2p1;
	//atomicAdd(smbw,m*b*wm2p1);
	//*smbw+=m*b*wm2p1;
	//atomicAdd(sbw,b*wm2p1);
	//*sbw+=b*wm2p1;
};
__global__ void calcCenter(float *d_x_c, float *d_y_c, float *d_x_c_old, float *d_y_c_old, float *det, float *sw, float *smmw, float *smw, float *smbw, float *sbw)
{
	*det=(*smw) * (*smw) - (*smmw) * (*sw);
	*d_x_c=((*smbw) * (*sw) - (*smw) * (*sbw))/(*det);
	*d_y_c=((*smbw)* (*smw)- (*smmw) * (*sbw))/(*det);

	*d_x_c_old=*d_x_c;
	*d_y_c_old=*d_y_c;
};


__global__ void updateCenter(RSImage_GPU * RSImage){
	//Due to an unknown reason, __device__ function calling does not work properly.
	//So I put all the codes into a kernel

};

void initRSImage(RSImage_GPU *RSImage,size_t width, size_t height, size_t ROIwidth, size_t ROIheight, size_t x_off, size_t y_off, float x_c_old=0.1f, float y_c_old=0.1f)
{
	//the image info
	cudaMalloc(&RSImage->d_width,sizeof*RSImage->d_width);
	cudaMalloc(&RSImage->d_height,sizeof*RSImage->d_height);
	cudaMalloc(&RSImage->d_ROIwidth,sizeof*RSImage->d_ROIwidth);
	cudaMalloc(&RSImage->d_ROIheight,sizeof*RSImage->d_ROIheight);

	cudaMemcpy(RSImage->d_width,&width,sizeof(width),cudaMemcpyHostToDevice);
	cudaMemcpy(RSImage->d_height,&height,sizeof(height),cudaMemcpyHostToDevice);
	cudaMemcpy(RSImage->d_ROIwidth,&ROIwidth,sizeof(ROIwidth),cudaMemcpyHostToDevice);
	cudaMemcpy(RSImage->d_ROIheight,&ROIheight,sizeof(ROIheight),cudaMemcpyHostToDevice);

	//bead position
	cudaMalloc((void**)&RSImage->d_x_off,sizeof*RSImage->d_x_off);
	cudaMalloc((void**)&RSImage->d_y_off,sizeof*RSImage->d_y_off);

	cudaMemcpy(RSImage->d_x_off,&x_off,sizeof*RSImage->d_x_off,cudaMemcpyHostToDevice);
	cudaMemcpy(RSImage->d_y_off,&y_off,sizeof*RSImage->d_y_off,cudaMemcpyHostToDevice);

	//device images
	cudaMalloc((void**)&RSImage->d_image,width*height*sizeof*RSImage->d_image);

	cudaMalloc((void**)&RSImage->d_du,(ROIwidth-1)*(ROIheight-1)*sizeof*RSImage->d_du);
	cudaMalloc((void**)&RSImage->d_dv,(ROIwidth-1)*(ROIheight-1)*sizeof*RSImage->d_dv);
	cudaMalloc((void**)&RSImage->d_duF,(ROIwidth-1)*(ROIheight-1)*sizeof*RSImage->d_duF);
	cudaMalloc((void**)&RSImage->d_dvF,(ROIwidth-1)*(ROIheight-1)*sizeof*RSImage->d_dvF);
	cudaMalloc((void**)&RSImage->d_grads,(ROIwidth-1)*(ROIheight-1)*sizeof*RSImage->d_grads);

	//center
	cudaMalloc((void**)&RSImage->d_x_c,sizeof*RSImage->d_x_c);
	cudaMalloc((void**)&RSImage->d_y_c,sizeof*RSImage->d_y_c);

	cudaMalloc((void**)&RSImage->d_x_c_old,sizeof*RSImage->d_x_c_old);
	cudaMalloc((void**)&RSImage->d_y_c_old,sizeof*RSImage->d_y_c_old);

	cudaMemcpy(RSImage->d_x_c_old,&x_c_old,sizeof*RSImage->d_x_c_old,cudaMemcpyHostToDevice);
	cudaMemcpy(RSImage->d_y_c_old,&y_c_old,sizeof*RSImage->d_y_c_old,cudaMemcpyHostToDevice);

	cudaMalloc(&RSImage->d_det,sizeof*RSImage->d_det);
	cudaMalloc(&RSImage->d_smw,sizeof*RSImage->d_smw);
	cudaMalloc(&RSImage->d_smmw,sizeof*RSImage->d_smmw);
	cudaMalloc(&RSImage->d_sw,sizeof*RSImage->d_sw);
	cudaMalloc(&RSImage->d_smbw,sizeof*RSImage->d_smbw);
	cudaMalloc(&RSImage->d_sbw,sizeof*RSImage->d_sbw);
};

void freeRSImage(RSImage_GPU *RSImage)
{
	//the image info
	cudaFree(RSImage->d_width);
	cudaFree(RSImage->d_height);
	cudaFree(RSImage->d_ROIwidth);
	cudaFree(RSImage->d_ROIheight);
	//bead position
	cudaFree(RSImage->d_x_off);
	cudaFree(RSImage->d_y_off);

	//device images
	cudaFree(RSImage->d_image);

	cudaFree(RSImage->d_du);
	cudaFree(RSImage->d_dv);
	cudaFree(RSImage->d_duF);
	cudaFree(RSImage->d_dvF);
	cudaFree(RSImage->d_grads);

	//Center poisitions
	cudaFree(RSImage->d_x_c);
	cudaFree(RSImage->d_y_c);
	cudaFree(RSImage->d_x_c_old);
	cudaFree(RSImage->d_y_c_old);

	cudaFree(RSImage->d_det);
	cudaFree(RSImage->d_smw);
	cudaFree(RSImage->d_smmw);
	cudaFree(RSImage->d_sw);
	cudaFree(RSImage->d_smbw);
	cudaFree(RSImage->d_sbw);
};

void freeImageDim()
{
	;
};

#endif