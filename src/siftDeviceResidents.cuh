#pragma once

extern __constant__ int d_MaxNumPoints;
extern __device__   unsigned int d_PointCounter[ 8 * 2 + 1 ];
extern __constant__ float d_ScaleDownKernel[5]; 
extern __constant__ float d_LowPassKernel[2*4 +1]; 
extern __constant__ float d_LaplaceKernel[8*12*16]; 