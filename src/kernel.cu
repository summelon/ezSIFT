#include "image_utility.h"
#include "common.h"
#include "image.h"
#include "ezsift.h"

#include <list>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#define HEIGHT_TILE 8
#define WIDTH_TILE 8


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


__global__ void cal_width_elem(int w, int h, int height_now, int gR,
                             float *coef, float *dev_buf, float *dev_dst)
{
    int width_now = blockIdx.x * blockDim.x + threadIdx.x;
    if (width_now >= w) return;

    float *prow = dev_buf + height_now * (w + gR * 2) + width_now;
    float *dstData = dev_dst + height_now + width_now * h;
    float partialSum = 0.0f;

    for (int i = -gR; i <= gR; i++) {
        partialSum += (*coef++) * (*prow++);
    }

    // prow -= 2 * gR;
    *dstData = partialSum;
}

__global__ void row_gauss_kernel(
        int w, int h, int gR,
        float *dev_dst, float *dev_coef, float *dev_buf)
{
    int height_now = blockIdx.x * blockDim.x + threadIdx.x;
    if (height_now >= h) return;

    // int dimBlock = WIDTH_TILE;
    // int dimGrid = w / WIDTH_TILE + 1;
    // cal_width_elem<<<dimGrid, dimBlock>>>(w, h, height_now, gR, dev_coef, dev_buf, dev_dst);

    float partialSum;
    float *coef;
    float *prow = dev_buf + height_now * (w + gR * 2);
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
}


void row_filter_transpose_gpu(
        float *host_src, float *host_dst, int w, int h,
        float *coef1d, int gR)
{
    int img_size = w * h;
    int buf_size = h * (w + gR * 2);
    // Allocate and copy to device memory
    float *dev_buf, *dev_dst, *dev_coef;
    float *row_buf, *row_start;
    float *host_buf = new float[buf_size];

    for (int height = 0; height < h; height++)
    {
        row_buf = host_buf + height * (w + gR * 2);
        row_start = host_src + height * w;
        memcpy(row_buf+gR, row_start, sizeof(float)*w);
        std::fill(row_buf, row_buf+gR, *row_start);
        std::fill( row_buf+w+gR, row_buf+w+2*gR, *(row_start+w-1) );
    }

    gpuErrchk( cudaMalloc( (void**)&dev_dst, sizeof(float)*img_size ) );
    gpuErrchk( cudaMalloc( (void**)&dev_coef, sizeof(float)*2*gR ) );
    gpuErrchk( cudaMalloc( (void**)&dev_buf, sizeof(float)*buf_size) );
    gpuErrchk( cudaMemcpy(dev_buf, host_buf, sizeof(float)*buf_size, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(dev_coef, coef1d, sizeof(float)*2*gR, cudaMemcpyHostToDevice) );

    // Set block and grid size
    int dimBlock = HEIGHT_TILE;
    int dimGrid = h / HEIGHT_TILE + 1;

    // Run kernel function
    row_gauss_kernel<<<dimGrid, dimBlock>>>(w, h, gR, dev_dst, dev_coef, dev_buf);
    gpuErrchk( cudaPeekAtLastError() );

    // Transfer from device mem to host and Clean
    gpuErrchk( cudaMemcpy(host_dst, dev_dst, sizeof(float)*img_size, cudaMemcpyDeviceToHost) );
    cudaFree(dev_dst); cudaFree(dev_coef); cudaFree(dev_buf);
    delete [] host_buf;

}
