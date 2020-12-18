#include "image_utility.h"
#include "common.h"
#include "image.h"
#include "ezsift.h"

#include <list>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 64


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


__global__ void row_gauss_kernel(
        int w, int h, int gR,
        float *dev_src, float *dev_dst, float *dev_coef,
        float *dev_buf)
{
    int height_now = blockIdx.x * blockDim.x + threadIdx.x;
    if (height_now >= h) return;

    float partialSum;
    float *coef;
    // float *row_buf = new float[w + gR * 2];
    float *row_buf = dev_buf + height_now * (w + gR * 2);
    float *row_start = dev_src + height_now * w;
    memcpy(row_buf + gR, row_start, sizeof(float) * w);
    float firstData = *(row_start);
    float lastData = *(row_start + w - 1);

    for (int i = 0; i < gR; i++) {
        row_buf[i] = firstData;
        row_buf[i + w + gR] = lastData;
    }

    float *prow = row_buf;
    float *dstData = dev_dst + height_now;
    for (int c = 0; c < w; c++) {
        partialSum = 0.0f;
        coef = dev_coef;

        for (int i = -gR; i <= gR; i++) {
            partialSum += (*coef++) * (*prow++);
        }

        prow -= 2 * gR;
        *dstData = partialSum;
        dstData += h;
    }

    // delete [] row_buf;
    // row_buf = nullptr;
}


void row_filter_transpose_gpu(
        float *src, float *dst, int w, int h,
        float *coef1d, int gR)
{
    // Allocate and copy to device memory
    float *dev_src, *dev_dst, *dev_coef;
    float *dev_buf;
    int img_size = w * h;
    int buf_size = h * (w + gR * 2);

    gpuErrchk( cudaMalloc( (void**)&dev_src, sizeof(float)*img_size ) );
    gpuErrchk( cudaMalloc( (void**)&dev_dst, sizeof(float)*img_size ) );
    gpuErrchk( cudaMalloc( (void**)&dev_coef, sizeof(float)*2*gR ) );
    gpuErrchk( cudaMalloc( (void**)&dev_buf, sizeof(float)*buf_size) );
    gpuErrchk( cudaMemcpy(dev_src, src, sizeof(float)*img_size, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(dev_coef, coef1d, sizeof(float)*2*gR, cudaMemcpyHostToDevice) );

    // Set block and grid size
    int dimBlock = TILE_SIZE;
    int dimGrid = h/TILE_SIZE+1;

    // Run kernel function
    row_gauss_kernel<<<dimGrid, dimBlock>>>(w, h, gR, dev_src, dev_dst, dev_coef, dev_buf);
    gpuErrchk( cudaPeekAtLastError() );

    // Transfer from device mem to host and Clean
    cudaMemcpy(dst, dev_dst, sizeof(float)*img_size, cudaMemcpyDeviceToHost);
    cudaFree(dev_src); cudaFree(dev_dst); cudaFree(dev_coef);
    cudaFree(dev_buf);
}
