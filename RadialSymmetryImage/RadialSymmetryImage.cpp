/*

Project		:	RadialSymmetryImage
Description	:	RadialSymmetryImage class to calculate the center of 
				a 2D image which has radial symmetry.
				This is a C++ porting. To see the original work,
				check <doi:10.1038/nmeth.2071>.

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

#include "RadialSymmetryImage.h"
#include <stdlib.h>
#include <cmath>
#include <stdio.h>

using namespace std;

RadialSymmetryImage::RadialSymmetryImage(uint8_t * image, size_t width, size_t height)
{
	m_image=image;
	m_width=width;
	m_height=height;

	//Initialize dervative images
	InitVars();
	
	//Calculate derivatives 
	UpdateDervs();
	
	//and update gradient field image
	UpdateGradField();
	
	//Calculate a center position
	UpdateCenter();
}



RadialSymmetryImage::~RadialSymmetryImage()
{
	free(m_gridX);
	free(m_gridY);

	free(m_du);
	free(m_dv);

	free(m_duFiltered);
	free(m_dvFiltered);

	free(m_gradMag);
	free(m_gradSlope);
}

void RadialSymmetryImage::InitVars()
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

	m_gradNumerator=1;
	m_gradNumerator=1;
	
	minimumFloat=0.0001f;
	m_gradMax=(float)m_height/minimumFloat;

	for(size_t y=0;y<m_height-1;y++)
	{
		for(size_t x=0;x<m_width-1;x++)
		{
			m_gridX[DCoord(y,x)]=-float(m_width-1)/2.0f+0.5f+float(x);
			m_gridY[DCoord(y,x)]=-float(m_height-1)/2.0f+0.5f+float(y); //Y increases downward
		}
	}
}

void RadialSymmetryImage::UpdateDervs()
{
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
		size_t y=0;
		m_duFiltered[DCoord(y,x)]=(	m_du[DCoord(y,x-1)]+	m_du[DCoord(y,x)]+		m_du[DCoord(y,x+1)]+	\
									m_du[DCoord(y+1,x-1)]+	m_du[DCoord(y+1,x)]+	m_du[DCoord(y+1,x+1)])/6;
		m_dvFiltered[DCoord(y,x)]=(	m_dv[DCoord(y,x-1)]+	m_dv[DCoord(y,x)]+		m_dv[DCoord(y,x+1)]+	\
									m_dv[DCoord(y+1,x-1)]+	m_dv[DCoord(y+1,x)]+	m_dv[DCoord(y+1,x+1)])/6;

		y=m_height-2;
		m_duFiltered[DCoord(y,x)]=(	m_du[DCoord(y-1,x-1)]+	m_du[DCoord(y-1,x)]+	m_du[DCoord(y-1,x+1)]+	\
									m_du[DCoord(y,x-1)]+	m_du[DCoord(y,x)]+		m_du[DCoord(y,x+1)])/6;
		m_dvFiltered[DCoord(y,x)]=(	m_dv[DCoord(y-1,x-1)]+	m_dv[DCoord(y-1,x)]+	m_dv[DCoord(y-1,x+1)]+	\
									m_dv[DCoord(y,x-1)]+	m_dv[DCoord(y,x)]+		m_dv[DCoord(y,x+1)])/6;
	}

	for(size_t y=1;y<m_height-2;y++)
	{
		size_t x=0;
		m_duFiltered[DCoord(y,x)]=(	m_du[DCoord(y-1,x)]+	m_du[DCoord(y-1,x+1)]+ \
									m_du[DCoord(y,x)]+		m_du[DCoord(y,x+1)]+ \
									m_du[DCoord(y+1,x)]+	m_du[DCoord(y+1,x+1)])/6;
		m_dvFiltered[DCoord(y,x)]=(	m_dv[DCoord(y-1,x)]+	m_dv[DCoord(y-1,x+1)]+ \
									m_dv[DCoord(y,x)]+		m_dv[DCoord(y,x+1)]+ \
									m_dv[DCoord(y+1,x)]+	m_dv[DCoord(y+1,x+1)])/6;

		x=m_width-2;
		m_duFiltered[DCoord(y,x)]=(	m_du[DCoord(y-1,x-1)]+	m_du[DCoord(y-1,x)]+	\
									m_du[DCoord(y,x-1)]+	m_du[DCoord(y,x)]+		\
									m_du[DCoord(y+1,x-1)]+	m_du[DCoord(y+1,x)])/6;
		m_dvFiltered[DCoord(y,x)]=(	m_dv[DCoord(y-1,x-1)]+	m_dv[DCoord(y-1,x)]+	\
									m_dv[DCoord(y,x-1)]+	m_dv[DCoord(y,x)]+		\
									m_dv[DCoord(y+1,x-1)]+	m_dv[DCoord(y+1,x)])/6;
	}

	//Smoothing corners
	m_duFiltered[DCoord(0,0)]					=(m_du[DCoord(0,0)]+	m_du[DCoord(0,1)]+	m_du[DCoord(1,0)]+	m_du[DCoord(1,1)])/4;
	m_duFiltered[DCoord(0,m_width-2)]			=(m_du[DCoord(0,m_width-2)]+m_du[DCoord(0,m_width-2-1)]+m_du[DCoord(1,m_width-2)]+m_du[DCoord(1,m_width-2-1)])/4;
	m_duFiltered[DCoord(m_height-2,0)]			=(m_du[DCoord(m_height-2,0)]+m_du[DCoord(m_height-2,1)]+m_du[DCoord(m_height-2-1,0)]+m_du[DCoord(m_height-2-1,1)])/4;
	m_duFiltered[DCoord(m_height-2,m_width-2)]	=(m_du[DCoord(m_height-2,m_width-2)]+m_du[DCoord(m_height-2,m_width-2-1)]+	\
												m_du[DCoord(m_height-2-1,m_width-2)]+m_du[DCoord(m_height-2-1,m_width-2-1)])/4;
}

void RadialSymmetryImage::UpdateGradField()
{
	for(size_t x=0;x<m_width-1;x++)
	{
		for(size_t y=0;y<m_height-1;y++)
		{
			// pi/4 tilted gradient
			//Magnitude
			m_gradMag[DCoord(y,x)]=m_duFiltered[DCoord(y,x)]*m_duFiltered[DCoord(y,x)]+m_dvFiltered[DCoord(y,x)]*m_dvFiltered[DCoord(y,x)];

			//Slope
			m_gradDenominator=m_duFiltered[DCoord(y,x)]-m_dvFiltered[DCoord(y,x)];
			m_gradNumerator=m_duFiltered[DCoord(y,x)]+m_dvFiltered[DCoord(y,x)];

			//if the Slope is too big, set appropriate size. Criteria is @m_height/@minimumFloat
			if(abs(m_gradNumerator) > m_gradMax) 
			{
				m_gradSlope[DCoord(y,x)]=m_gradMax;
				continue;
			}

			//to avoid divide by 0, compare the size of denominator with minimumFloat
			if (m_gradDenominator<0)
				m_gradDenominator=m_gradDenominator<-minimumFloat?m_gradDenominator:minimumFloat;
			else
				m_gradDenominator=m_gradDenominator>minimumFloat?m_gradDenominator:minimumFloat;

			m_gradSlope[DCoord(y,x)]=-m_gradNumerator/m_gradDenominator;
		}
	}
}

void RadialSymmetryImage::UpdateCenter()
{
	;
}

void RadialSymmetryImage::GetCenter(float * pX_c, float * pY_c)
{
	pX_c=&m_x_c;
	pY_c=&m_y_c;
}

size_t RadialSymmetryImage::Coord(size_t y, size_t x)
{
//Return 1D index for 2D image

	return (y*m_width)+x;
}

size_t RadialSymmetryImage::DCoord(size_t y, size_t x)
{
	//Return 1D index for derivative 2D image
	return (y*(m_width-1))+x;
}

