#ifndef INIT_CUDA_H_
#define INIT_CUDA_H_

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//void printDeviceProp(const cudaDeviceProp &prop);

//#include "book.h"
#include "cpu_bitmap.h"
void InitCUDA();



#endif /* INIT_CUDA_H_ */
