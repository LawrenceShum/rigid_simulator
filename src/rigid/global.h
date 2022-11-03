// Copyright (C) 2019 Xiao Zhai

#pragma once
#include<vector_types.h>
typedef int IndexType;
typedef double ScalarType;
#define type3 double3
#define make_type3 make_double3
#define type4 double4
#define make_type4 make_double4
//typedef float ScalarType;
//#define type3 float3
//#define make_type3 make_float3
//#define type4 float4
//#define make_type4 make_float4
enum  Scene{ impact = 0, drop, thin_glass, rod, beam, cloth, spring };

const int block_size = 256;
#define EPSILON (1e-6f)
#define PI (3.14159265358979323846f)
#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { printf("CUDA Error at %s:%d\t Error code = %d\n",__FILE__,__LINE__,x);}} while(0) 
//#define CUDA_CALL(x) do { x ;} while(0) 
#define CHECK_KERNEL(); 	{cudaError_t err = cudaGetLastError();if(err)printf("CUDA Error at %s:%d:\t%s\n",__FILE__,__LINE__,cudaGetErrorString(err));}
#define MAX_A (1000.0f)

//#define nThreadMax_perBlock  1024;

