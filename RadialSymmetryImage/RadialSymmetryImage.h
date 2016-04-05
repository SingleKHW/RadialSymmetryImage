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

#pragma once
#ifndef __RADIALSYMMETRYIMAGE_H__
#define __RADIALSYMMETRYIMAGE_H__

#define _AMD64_

#include <stdint.h>
#include <WinDef.h>

class RadialSymmetryImage
{
public:
	// Constructor
	// Take an image
	RadialSymmetryImage(uint8_t * image, size_t width, size_t height);

	// Update the center. You will call it when new image acquired
	void UpdateCenter();

	// Get center poisition
	void GetCenter(float * pX_c, float * pY_c);

	// Destructor
	virtual ~RadialSymmetryImage();

protected:
	uint8_t * m_image; //This is your image. It made to be updated externally upon an image acquisition.

	size_t m_width;
	size_t m_height;
	float m_x_c;	//X center
	float m_y_c;	//Y center
	float m_x_centroid;	//X center
	float m_y_centroid;	//Y center


	//midpoint grid coordinates
	float * m_gridX;
	float * m_gridY;

	//derivatives along 45-degree shifted coordinates (u and v)
	float * m_du;
	float * m_dv;

	//Filtered derivatives
	float * m_duFiltered;
	float * m_dvFiltered;

	//Gradient magnitude and slope
	float * m_gradMag;
	float * m_gradSlope;
	float * m_gradIntercept; //y=@m_gradSlope * x + @m_gradIntercept is for a gradient vector
	float m_gradMass;//Sum of m_gradMag

	float m_gradDenominator;
	float m_gradNumerator;
	float m_gradMax;
	float minimumFloat;

	//Variables for center fitting
	float wm2p1;
	float sw;
	float smmw;
	float smw;
	float smbw;
	float sbw;
	float det;
	float m; //slope
	float b; //intercept
	float w; //weight

	//Allocate memory
	void InitVars();
	//Update derivative images
	void CalcDervs();
	//Update gradient field
	void CalcGradField();
	//Update center poisition
	void CalcCenter();

	//1D coord for @m_width X @m_height dimension image
	size_t Coord(size_t y, size_t x);
	//1D coord for @m_width-1 X @m_height-1 dimension image, i.e., derivative images.
	size_t DCoord(size_t y, size_t x);
};


#endif