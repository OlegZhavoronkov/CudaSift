//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  
#pragma once
#ifndef CUDASIFTD_H
#define CUDASIFTD_H
#include <cudasift/cudaSift.h>
constexpr const unsigned int NUM_SCALES =5;

// Scale down thread block width
//#define SCALEDOWN_W    64 // 60 
constexpr const unsigned int  SCALEDOWN_W = 64;

// Scale down thread block height
//#define SCALEDOWN_H    16 // 8
constexpr const unsigned int SCALEDOWN_H = 16;

// Scale up thread block width
//#define SCALEUP_W      64
constexpr const unsigned int SCALEUP_W = 64;

// Scale up thread block height
//#define SCALEUP_H       8
constexpr const unsigned int SCALEUP_H = 8;

// Find point thread block width
//#define MINMAX_W       30 //32 
constexpr const unsigned int MINMAX_W = 30;//32 

// Find point thread block height
//#define MINMAX_H        8 //16 
constexpr const unsigned int MINMAX_H = 8; //16 

// Laplace thread block width
//#define LAPLACE_W     128 // 56
constexpr const unsigned int LAPLACE_W = 128; // 56

// Laplace rows per thread
//#define LAPLACE_H       4
constexpr const unsigned int LAPLACE_H = 4;

// Number of laplace scales
//#define LAPLACE_S   (NUM_SCALES+3)
constexpr const unsigned int LAPLACE_S = ( NUM_SCALES + 3 );

// Laplace filter kernel radius
//#define LAPLACE_R       4
constexpr const int LAPLACE_R = 4;

//#define LOWPASS_W      24 //56
constexpr const int LOWPASS_W = 24; //56
//#define LOWPASS_H      32 //16
constexpr const int LOWPASS_H = 32;
//#define LOWPASS_R       4
constexpr const int LOWPASS_R = 4;

//====================== Number of threads ====================//
// ScaleDown:               SCALEDOWN_W + 4
// LaplaceMulti:            (LAPLACE_W+2*LAPLACE_R)*LAPLACE_S
// FindPointsMulti:         MINMAX_W + 2
// ComputeOrientations:     128
// ExtractSiftDescriptors:  256

//====================== Number of blocks ====================//
// ScaleDown:               (width/SCALEDOWN_W) * (height/SCALEDOWN_H)
// LaplceMulti:             (width+2*LAPLACE_R)/LAPLACE_W * height
// FindPointsMulti:         (width/MINMAX_W)*NUM_SCALES * (height/MINMAX_H)
// ComputeOrientations:     numpts
// ExtractSiftDescriptors:  numpts

__global__ void ScaleDownDenseShift( float* d_Result , float* d_Data , int width , int pitch , int height , int newpitch );
__global__ void ScaleDownDense( float* d_Result , float* d_Data , int width , int pitch , int height , int newpitch );
__global__ void ScaleDown( float* d_Result , float* d_Data , int width , int pitch , int height , int newpitch );
__global__ void ScaleUp( float* d_Result , float* d_Data , int width , int pitch , int height , int newpitch );
__global__ void ExtractSiftDescriptors( cudaTextureObject_t texObj , SiftPoint* d_sift , int fstPts , float subsampling );
__device__ float FastAtan2( float y , float x );
__global__ void ExtractSiftDescriptorsCONSTNew( cudaTextureObject_t texObj , SiftPoint* d_sift , float subsampling , int octave );
__global__ void ExtractSiftDescriptorsCONST( cudaTextureObject_t texObj , SiftPoint* d_sift , float subsampling , int octave );
__global__ void ExtractSiftDescriptorsOld( cudaTextureObject_t texObj , SiftPoint* d_sift , int fstPts , float subsampling );
__device__ void ExtractSiftDescriptor( cudaTextureObject_t texObj , SiftPoint* d_sift , float subsampling , int octave , int bx );
__global__ void RescalePositions( SiftPoint* d_sift , int numPts , float scale );
__global__ void ComputeOrientations( cudaTextureObject_t texObj , SiftPoint* d_Sift , int fstPts );
__global__ void ComputeOrientationsCONSTNew( float* image , int w , int p , int h , SiftPoint* d_Sift , int octave );
__global__ void ComputeOrientationsCONST( cudaTextureObject_t texObj , SiftPoint* d_Sift , int octave );
__global__ void OrientAndExtractCONST( cudaTextureObject_t texObj , SiftPoint* d_Sift , float subsampling , int octave );
__global__ void FindPointsMultiTest( float* d_Data0 , SiftPoint* d_Sift , int width , int pitch , int height , float subsampling , float lowestScale , float thresh , float factor , float edgeLimit , int octave );
__global__ void FindPointsMultiNew( float* d_Data0 , SiftPoint* d_Sift , int width , int pitch , int height , float subsampling , float lowestScale , float thresh , float factor , float edgeLimit , int octave );
__global__ void FindPointsMulti( float* d_Data0 , SiftPoint* d_Sift , int width , int pitch , int height , float subsampling , float lowestScale , float thresh , float factor , float edgeLimit , int octave );
__global__ void FindPointsMultiOld( float* d_Data0 , SiftPoint* d_Sift , int width , int pitch , int height , float subsampling , float lowestScale , float thresh , float factor , float edgeLimit , int octave );
__global__ void LaplaceMultiTex( cudaTextureObject_t texObj , float* d_Result , int width , int pitch , int height , int octave );
__global__ void LaplaceMultiMem( float* d_Image , float* d_Result , int width , int pitch , int height , int octave );
__global__ void LaplaceMultiMemWide( float* d_Image , float* d_Result , int width , int pitch , int height , int octave );
__global__ void LaplaceMultiMemTest( float* d_Image , float* d_Result , int width , int pitch , int height , int octave );
__global__ void LaplaceMultiMemOld( float* d_Image , float* d_Result , int width , int pitch , int height , int octave );
__global__ void LowPass( float* d_Image , float* d_Result , int width , int pitch , int height );
__global__ void LowPassBlockOld( float* d_Image , float* d_Result , int width , int pitch , int height );
__global__ void LowPassBlock( float* d_Image , float* d_Result , int width , int pitch , int height );

#endif
