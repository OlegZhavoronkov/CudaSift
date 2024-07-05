#pragma once

extern __constant__ int d_MaxNumPoints;
extern __device__ unsigned int d_PointCounter[  ];
extern __constant__ float d_ScaleDownKernel[]; 
extern __constant__ float d_LowPassKernel[]; 
extern __constant__ float d_LaplaceKernel[]; 