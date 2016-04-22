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
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//#include "RSImage_GPU.h"

#include <stdlib.h>
#include <cmath>
#include <stdio.h>

#include <Windows.h>
#include <ctime>

using namespace std;
