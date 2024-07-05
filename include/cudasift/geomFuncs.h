#pragma once
#include <cudasift/cudaSift.h>


int ImproveHomography( SiftData& data , float* homography , int numLoops , float minScore , float maxAmbiguity , float thresh );