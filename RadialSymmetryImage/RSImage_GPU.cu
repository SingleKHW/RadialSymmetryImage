/*

Project		:	RadialSymmetryImage
Description	:	RadialSymmetryImage class to calculate the center of 
				a 2D image which has radial symmetry.
				This is a C/C++  translation of the original work,
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
#include <cuda_runtime.h>
#include "RSImage_GPU.h"

#include <stdlib.h>
#include <cmath>
#include <stdio.h>

#include <Windows.h>
#include <ctime>

using namespace std;

RSImage_GPU::RSImage_GPU(uint8_t * image, size_t width, size_t height)
{
	m_maxWarp=32;
	m_maxThreads=1024;

	m_width=width;
	m_height=height;
	m_size=m_width*m_height*sizeof(*d_image);
	//Malloc CUDA image
	cudaMalloc(&d_image,m_size);
	cudaMemcpy(d_image,image,m_size,cudaMemcpyHostToDevice);

	//Define block structure
	m_block.x=m_maxWarp; // is 32
	m_block.y=m_maxThreads/m_maxWarp; // is 32

	m_grid.x=width/m_block.x;	// Image dimension should be multiples of 32
	m_grid.y=height/m_block.y;

	//Initialize dervative images
	InitVars();

	//Calculate derivatives 
	CalcDervs();

	//Update gradient field image
	CalcGradField();

	//Centroid calculation as a poor approximation

	//Initialize centroid
	m_x_centroid=0;
	m_y_centroid=0;

	//Find centroid to update @m_weight
	for(size_t y=0;y<m_height-1;y++)
	{
		for(size_t x=0;x<m_width-1;x++)
		{
			m_x_centroid+=m_gradMag[DCoord(y,x)]*m_gridX[DCoord(y,x)];
			m_y_centroid+=m_gradMag[DCoord(y,x)]*m_gridY[DCoord(y,x)];
		}
	}
	m_x_centroid/=m_gradMass;
	m_y_centroid/=m_gradMass;

	m_x_c=m_x_centroid;
	m_y_c=m_y_centroid;

	//Calculate a center position
	CalcCenter();
}



RSImage_GPU::~RSImage_GPU()
{
	free(m_gridX);
	free(m_gridY);

	free(m_du);
	free(m_dv);

	free(m_duFiltered);
	free(m_dvFiltered);

	free(m_gradMag);
	free(m_gradSlope);
	free(m_gradIntercept);
}

void RSImage_GPU::InitVars()
{
	//Allocate memories
	m_gridX=(float*) malloc((m_width-1)*(m_height-1)*sizeof(float));
	m_gridY=(float*) malloc((m_width-1)*(m_height-1)*sizeof(float));

	m_du=(float*) malloc((m_width-1)*(m_height-1)*sizeof(float));
	m_dv=(float*) malloc((m_width-1)*(m_height-1)*sizeof(float));

	m_duFiltered=(float*) malloc((m_width-1)*(m_height-1)*sizeof(float));
	m_dvFiltered=(float*) malloc((m_width-1)*(m_height-1)*sizeof(float));

	m_gradMag=(float*) malloc((m_width-1)*(m_height-1)*sizeof(float));
	m_gradSlope=(float*) malloc((m_width-1)*(m_height-1)*sizeof(float));
	m_gradIntercept=(float*) malloc((m_width-1)*(m_height-1)*sizeof(float));

	m_gradNumerator=1;
	m_gradNumerator=1;

	minimumFloat=0.0001f; //Typically, 1 pixel ~ 100 nm. 0.0001f ~ 0.01 nm. Float types can have 7 significant digits.

	for(size_t y=0;y<m_height-1;y++)
	{
		for(size_t x=0;x<m_width-1;x++)
		{
			m_gridX[DCoord(y,x)]=-float(m_width-1)/2.0f+0.5f+float(x);
			m_gridY[DCoord(y,x)]=-float(m_height-1)/2.0f+0.5f+float(y); //Y increases downward
		}
	}
}

__device__ void RSImage_GPU::CalcDervs()
{
	/* CPU code

	for(size_t y=0;y<m_height-1;y++)
	{
		for(size_t x=0;x<m_width-1;x++)
		{
			// pi/4 tilted derivatives
			m_du[DCoord(y,x)]=float(m_image[Coord(y,x+1)])-float(m_image[Coord(y+1,x)]);
			m_dv[DCoord(y,x)]=float(m_image[Coord(y,x)])-float(m_image[Coord(y+1,x+1)]);
		}
	}

	//Smoothing. 3x3 average
	for(size_t y=1;y<m_height-2;y++)
	{
		for(size_t x=1;x<m_width-2;x++)
		{
			m_duFiltered[DCoord(y,x)]=(	m_du[DCoord(y-1,x-1)]+	m_du[DCoord(y-1,x)]+	m_du[DCoord(y-1,x+1)]+ \
				m_du[DCoord(y,x-1)]+	m_du[DCoord(y,x)]+		m_du[DCoord(y,x+1)]+ \
				m_du[DCoord(y+1,x-1)]+	m_du[DCoord(y+1,x)]+	m_du[DCoord(y+1,x+1)])/9;
			m_dvFiltered[DCoord(y,x)]=(	m_dv[DCoord(y-1,x-1)]+	m_dv[DCoord(y-1,x)]+	m_dv[DCoord(y-1,x+1)]+ \
				m_dv[DCoord(y,x-1)]+	m_dv[DCoord(y,x)]+		m_dv[DCoord(y,x+1)]+ \
				m_dv[DCoord(y+1,x-1)]+	m_dv[DCoord(y+1,x)]+	m_dv[DCoord(y+1,x+1)])/9;
		}
	}

	//Smoothing edges
	for(size_t x=1;x<m_width-2;x++)
	{
		m_duFiltered[DCoord(0,x)]=(	m_du[DCoord(0,x-1)]+	m_du[DCoord(0,x)]+		m_du[DCoord(0,x+1)]+	\
									m_du[DCoord(0+1,x-1)]+	m_du[DCoord(0+1,x)]+	m_du[DCoord(0+1,x+1)])/6;
		m_dvFiltered[DCoord(0,x)]=(	m_dv[DCoord(0,x-1)]+	m_dv[DCoord(0,x)]+		m_dv[DCoord(0,x+1)]+	\
									m_dv[DCoord(0+1,x-1)]+	m_dv[DCoord(0+1,x)]+	m_dv[DCoord(0+1,x+1)])/6;

		m_duFiltered[DCoord(m_height-2,x)]=(	m_du[DCoord(m_height-2-1,x-1)]+	m_du[DCoord(m_height-2-1,x)]+	m_du[DCoord(m_height-2-1,x+1)]+	\
			m_du[DCoord(m_height-2,x-1)]+	m_du[DCoord(m_height-2,x)]+		m_du[DCoord(m_height-2,x+1)])/6;
		m_dvFiltered[DCoord(m_height-2,x)]=(	m_dv[DCoord(m_height-2-1,x-1)]+	m_dv[DCoord(m_height-2-1,x)]+	m_dv[DCoord(m_height-2-1,x+1)]+	\
			m_dv[DCoord(m_height-2,x-1)]+	m_dv[DCoord(m_height-2,x)]+		m_dv[DCoord(m_height-2,x+1)])/6;
	}

	for(size_t y=1;y<m_height-2;y++)
	{
		m_duFiltered[DCoord(y,0)]=(	m_du[DCoord(y-1,0)]+	m_du[DCoord(y-1,0+1)]+ \
			m_du[DCoord(y,0)]+		m_du[DCoord(y,0+1)]+ \
			m_du[DCoord(y+1,0)]+	m_du[DCoord(y+1,0+1)])/6;
		m_dvFiltered[DCoord(y,0)]=(	m_dv[DCoord(y-1,0)]+	m_dv[DCoord(y-1,0+1)]+ \
			m_dv[DCoord(y,0)]+		m_dv[DCoord(y,0+1)]+ \
			m_dv[DCoord(y+1,0)]+	m_dv[DCoord(y+1,0+1)])/6;

		m_duFiltered[DCoord(y,m_width-2)]=(	m_du[DCoord(y-1,m_width-2-1)]+	m_du[DCoord(y-1,m_width-2)]+	\
			m_du[DCoord(y,m_width-2-1)]+	m_du[DCoord(y,m_width-2)]+		\
			m_du[DCoord(y+1,m_width-2-1)]+	m_du[DCoord(y+1,m_width-2)])/6;
		m_dvFiltered[DCoord(y,m_width-2)]=(	m_dv[DCoord(y-1,m_width-2-1)]+	m_dv[DCoord(y-1,m_width-2)]+	\
			m_dv[DCoord(y,m_width-2-1)]+	m_dv[DCoord(y,m_width-2)]+		\
			m_dv[DCoord(y+1,m_width-2-1)]+	m_dv[DCoord(y+1,m_width-2)])/6;
	}

	//Smoothing corners
	m_duFiltered[DCoord(0,0)]					=(m_du[DCoord(0,0)]+	m_du[DCoord(0,1)]+	m_du[DCoord(1,0)]+	m_du[DCoord(1,1)])/4;
	m_duFiltered[DCoord(0,m_width-2)]			=(m_du[DCoord(0,m_width-2)]+m_du[DCoord(0,m_width-2-1)]+m_du[DCoord(1,m_width-2)]+m_du[DCoord(1,m_width-2-1)])/4;
	m_duFiltered[DCoord(m_height-2,0)]			=(m_du[DCoord(m_height-2,0)]+m_du[DCoord(m_height-2,1)]+m_du[DCoord(m_height-2-1,0)]+m_du[DCoord(m_height-2-1,1)])/4;
	m_duFiltered[DCoord(m_height-2,m_width-2)]	=(m_du[DCoord(m_height-2,m_width-2)]+m_du[DCoord(m_height-2,m_width-2-1)]+	\
		m_du[DCoord(m_height-2-1,m_width-2)]+m_du[DCoord(m_height-2-1,m_width-2-1)])/4;

	*/
}

__device__ void RSImage_GPU::CalcGradField()
{
	/*CPU code
	m_gradMass=0;
	for(size_t y=0;y<m_height-1;y++)
	{
		for(size_t x=0;x<m_width-1;x++)
		{
			// pi/4 tilted gradient
			//Magnitude
			m_gradMag[DCoord(y,x)]=m_duFiltered[DCoord(y,x)]*m_duFiltered[DCoord(y,x)]+m_dvFiltered[DCoord(y,x)]*m_dvFiltered[DCoord(y,x)];
			m_gradMass+=m_gradMag[DCoord(y,x)];

			//Slope
			m_gradDenominator=m_duFiltered[DCoord(y,x)]-m_dvFiltered[DCoord(y,x)];
			m_gradNumerator=m_duFiltered[DCoord(y,x)]+m_dvFiltered[DCoord(y,x)];

			//if divide by 0
			// set small enough denominator.

			if (m_gradDenominator==0)
				m_gradDenominator=minimumFloat;

			m_gradSlope[DCoord(y,x)]=-m_gradNumerator/m_gradDenominator;
			m_gradIntercept[DCoord(y,x)]=m_gridY[DCoord(y,x)]-(-m_gradNumerator/m_gradDenominator)*m_gridX[DCoord(y,x)];
		}
	}
	*/
}

__device__ void RSImage_GPU::CalcCenter()
{
	//The previous center positions are @m_x_centroid, @m_y_centroid
	//It assume that the bead does not move too much

	/*CPU code
	m_x_centroid=m_x_c;
	m_y_centroid=m_y_c;

	//Initialize variables
	wm2p1=0;
	sw=0;
	smmw=0;
	smw=0;
	smbw=0;
	sbw=0;
	det=0;

	m=0; //slope
	b=0; //intercept
	w=0; //weight

	for(size_t y=0;y<m_height-1;y++)
	{
		for(size_t x=0;x<m_width-1;x++)
		{
			w=m_gradMag[DCoord(y,x)]/ \
				sqrt(pow(m_gridX[DCoord(y,x)]-m_x_centroid,2)+pow(m_gridY[DCoord(y,x)]-m_y_centroid,2));

			m=m_gradSlope[DCoord(y,x)];
			b=m_gradIntercept[DCoord(y,x)];

			wm2p1=w/(m*m+1);

			sw+=wm2p1;
			smmw+=m*m*wm2p1;
			smw+=m*wm2p1;
			smbw+=m*b*wm2p1;
			sbw+=b*wm2p1;
		}
	}

	det=smw*smw-smmw*sw;
	m_x_c=(smbw*sw-smw*sbw)/det;
	m_y_c=(smbw*smw-smmw*sbw)/det;
	*/
}

__global__ void RSImage_GPU::UpdateCenter()
{
	//Calculate derivatives 
	CalcDervs();

	//Update gradient field image
	CalcGradField();

	//Calculate a center position
	CalcCenter();
}

void RSImage_GPU::GetCenter(float * pX_c, float * pY_c)
{
	*pX_c=m_x_c+(m_width+1)/2.0;
	*pY_c=m_y_c+(m_height+1)/2.0;
}

size_t RSImage_GPU::Coord(size_t y, size_t x)
{
	//Return 1D index for 2D image
	return (y*m_width)+x;
}

size_t RSImage_GPU::DCoord(size_t y, size_t x)
{
	//Return 1D index for derivative 2D image
	return (y*(m_width-1))+x;
}

