#include "image_utility.h"
#include "common.h"
#include "image.h"
#include "ezsift.h"

#include <list>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

#define HEIGHT_TILE 16
#define WIDTH_TILE 16


// #define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
// inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
// {
//     if (code != cudaSuccess)
//     {
//         fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//         if (abort) exit(code);
//     }
// }
inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n",
                cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
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
        float *dev_src, float *dev_dst, float *dev_coef, float *dev_buf)
{
    int height_now = blockIdx.x * blockDim.x + threadIdx.x;
    if (height_now >= h) return;

    // --- Copy buffer ---
    float *row_buf, *row_start;
    row_buf = dev_buf + height_now * (w + gR * 2);
    row_start = dev_src + height_now * w;
    memcpy(row_buf+gR, row_start, sizeof(float)*w);
    float firstData = *(row_start);
    float lastData = *(row_start + w - 1);
    for (int i = 0; i < gR; i++) {
        row_buf[i] = firstData;
        row_buf[i + w + gR] = lastData;
    }

    // --- Calculate new column ---
    // int dimBlock = WIDTH_TILE;
    // int dimGrid = w / WIDTH_TILE + 1;
    // cal_width_elem<<<dimGrid, dimBlock>>>(w, h, height_now, gR, dev_coef, dev_buf, dev_dst);

    float partialSum;
    float *coef;
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
}


void row_filter_transpose_gpu(
        float *host_src, float *host_dst,
        int w, int h, float *coef1d, int gR)
{
    // Image<float> img_t(h, w);
    int img_size = w * h;
    int in_buf_size = h * (w + gR * 2);
    int out_buf_size = w * (h + gR * 2);
    // Allocate and copy to device memory
    float *dev_src, *dev_dst, *dev_swap, *dev_coef;
    float *dev_in_buf, *dev_out_buf;

    // Image
    checkCuda( cudaMalloc( (void**)&dev_src, sizeof(float)*img_size ) );
    checkCuda( cudaMalloc( (void**)&dev_dst, sizeof(float)*img_size ) );
    checkCuda( cudaMalloc( (void**)&dev_swap, sizeof(float)*img_size ) );
    // Buffer
    checkCuda( cudaMalloc( (void**)&dev_in_buf, sizeof(float)*in_buf_size) );
    checkCuda( cudaMalloc( (void**)&dev_out_buf, sizeof(float)*out_buf_size) );
    // Coef array
    checkCuda( cudaMalloc( (void**)&dev_coef, sizeof(float)*2*gR ) );

    checkCuda( cudaMemcpy(dev_src, host_src, sizeof(float)*img_size, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(dev_coef, coef1d, sizeof(float)*2*gR, cudaMemcpyHostToDevice) );

    // Set block and grid size
    int dimBlock, dimGrid;

    // Run kernel function
    dimBlock = HEIGHT_TILE;
    dimGrid = h / HEIGHT_TILE + 1;
    row_gauss_kernel<<<dimGrid, dimBlock>>>(w, h, gR, dev_src, dev_swap, dev_coef, dev_in_buf);
    checkCuda( cudaPeekAtLastError() );

    dimBlock = WIDTH_TILE;
    dimGrid = w / WIDTH_TILE + 1;
    row_gauss_kernel<<<dimGrid, dimBlock>>>(h, w, gR, dev_swap, dev_dst, dev_coef, dev_out_buf);
    checkCuda( cudaPeekAtLastError() );

    // Transfer from device mem to host and Clean
    checkCuda( cudaMemcpy(host_dst, dev_dst, sizeof(float)*img_size, cudaMemcpyDeviceToHost) );
    cudaFree(dev_src); cudaFree(dev_dst); cudaFree(dev_swap);
    cudaFree(dev_in_buf); cudaFree(dev_out_buf);
    cudaFree(dev_coef);
}
