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

//CUDA error check from http://stackoverflow.com/a/14038590
#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <stdint.h>
#include <stdio.h>

using namespace std;

//CUDA error check from http://stackoverflow.com/a/14038590
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);

struct RSImage_GPU{
	uint8_t *h_image;	//host image
	uint8_t *d_image;	//device image
	size_t *d_width, *d_height; //the acquired image dimension

	//image info
	size_t ROIwidth, ROIheight;
	size_t width, height;

	float *d_du,*d_dv,*d_duF,*d_dvF,*d_grads; //derivative images of ROI
	size_t *d_ROIwidth, *d_ROIheight;  //Image info for device access
	
	size_t *d_x_off, *d_y_off; //ROI offset in the image

	float h_x_c, h_y_c;	//the center
	float *d_x_c_old, *d_y_c_old; //the previously calculated center

	float sw, smmw, smw, smbw, sbw, det;
	float *h_sw;//, *h_smw,*h_smmw, *h_smbw,*h_sbw;
	float *d_sw;//*d_smw,*d_smmw,, *d_smbw,*d_sbw;
};

//wrapper functions
void updateRSImageCenter(RSImage_GPU * RSImage, dim3 grid, dim3 block, cudaStream_t stream);
void transferRSImageHtoD(RSImage_GPU * RSImage, cudaStream_t stream); //transfer image from host to device
void transferRSImageDtoH(RSImage_GPU * RSImage, cudaStream_t stream); //transfer derivative images from device to host
void calcCenter(RSImage_GPU * RSImage, cudaStream_t stream);//calculate center on CPU

//GPU kernels
__global__ void calcDervs(uint8_t *d_image, float *d_du, float *d_dv, size_t *d_x_off, size_t *d_y_off,  size_t *d_width, size_t *d_height, size_t *d_ROIwidth, size_t *d_ROIheight);
__global__ void calcDervsF(float *d_du, float *d_duF, float *d_dv, float *d_dvF, size_t *d_ROIwidth, size_t *d_ROIheight, float *d_sw);
__global__ void calcGrads(float *d_duF, float *d_dvF, size_t *d_ROIwidth, size_t *d_ROIheight, float *d_x_c_old, float *d_y_c_old, float *sw);

//init and destroy RSImage_GPU struct
void initRSImage(RSImage_GPU *RSImage, uint8_t *h_image, size_t width, size_t height, size_t ROIwidth, size_t ROIheight, size_t x_off, size_t y_off, float x_c_old=0.1f, float y_c_old=0.1f);
void freeRSImage(RSImage_GPU *RSImage);

//

#endif