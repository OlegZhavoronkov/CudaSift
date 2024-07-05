constexpr const unsigned int LOWPASS_R = 4;
__constant__ int d_MaxNumPoints;
__device__ unsigned int d_PointCounter[ 8 * 2 + 1 ];
__constant__ float d_ScaleDownKernel[5]; 
__constant__ float d_LowPassKernel[2*LOWPASS_R+1]; 
__constant__ float d_LaplaceKernel[8*12*16]; 