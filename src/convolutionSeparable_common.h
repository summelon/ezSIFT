/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



#ifndef CONVOLUTIONSEPARABLE_COMMON_H
#define CONVOLUTIONSEPARABLE_COMMON_H



// #define KERNEL_RADIUS 8
// #define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)
#define   ROWS_BLOCKDIM_X 16
#define   ROWS_BLOCKDIM_Y 8
#define ROWS_RESULT_STEPS 8
#define   ROWS_HALO_STEPS 1

#define   COLUMNS_BLOCKDIM_X 16
#define   COLUMNS_BLOCKDIM_Y 16
#define COLUMNS_RESULT_STEPS 8
#define   COLUMNS_HALO_STEPS 1


////////////////////////////////////////////////////////////////////////////////
// GPU convolution
////////////////////////////////////////////////////////////////////////////////
extern "C" void setConvolutionKernel(float *h_Kernel, int kernel_radius);

extern "C" void convolutionRowsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int kernel_radius
);

extern "C" void convolutionColumnsGPU(
    float *d_Dst,
    float *d_Src,
    int imageW,
    int imageH,
    int kernel_radius
);



#endif
