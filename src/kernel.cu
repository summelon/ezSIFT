#include "image_utility.h"
#include "common.h"
#include "image.h"
#include "ezsift.h"

#include <list>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <limits>

#define MATCH_TILE 96
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


// ------------------------------------------------------------------------------------------------


__device__ float fast_resqrt_f(float x)
{
    // 32-bit version
    union {
        float x;
        int i;
    } u;

    float xhalf = (float)0.5 * x;

    // convert floating point value in RAW integer
    u.x = x;

    // gives initial guess y0
    u.i = 0x5f3759df - (u.i >> 1);

    // two Newton steps
    u.x = u.x * ((float)1.5 - xhalf * u.x * u.x);
    u.x = u.x * ((float)1.5 - xhalf * u.x * u.x);
    return u.x;
}

__device__ float fast_sqrt_f(float x)
{
    return (x < 1e-8) ? 0 : x * fast_resqrt_f(x);
}


__global__ void compare_euc_dis(
        float *dev_descr1, float *dev_descr2, int *dev_match_list,
        int list1_size, int list2_size, float threshold, float float_max)
{
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx1 >= list1_size) return;

    float score1, score2;
    score1 = score2 = float_max;
    float *descr1 = dev_descr1 + idx1 * DEGREE_OF_DESCRIPTORS;

    float *descr2, score, dif;
    int match_idx;
    for (int idx2 = 0; idx2 < list2_size; idx2++)
    {
        score = 0;
        descr2 = dev_descr2 + idx2 * DEGREE_OF_DESCRIPTORS;
        for (int i = 0; i < DEGREE_OF_DESCRIPTORS; i++)
        {
            dif = descr1[i] - descr2[i];
            score += dif * dif;
        }
        if (score < score1)
        {
            score2 = score1;
            score1 = score;
            match_idx = idx2;
        }
        else if (score < score2)
        {
            score2 = score;
        }
    }
    if (fast_sqrt_f(score1 / score2) < threshold)
        dev_match_list[idx1] = match_idx;
    else
        dev_match_list[idx1] = -1;
}


void match_keypoints_gpu(std::list<ezsift::SiftKeypoint> &kpt_list1,
                         std::list<ezsift::SiftKeypoint> &kpt_list2,
                         std::list<ezsift::MatchPair> &match_list)
{
    std::list<ezsift::SiftKeypoint>::iterator kpt1;
    std::list<ezsift::SiftKeypoint>::iterator kpt2;
    int list1_size = kpt_list1.size();
    int list2_size = kpt_list2.size();

    // --- Allocate memory ---
    //  Host
    float *host_descr1, *host_descr2;
    int *host_match_list;
    host_descr1 = new float[list1_size*DEGREE_OF_DESCRIPTORS];
    host_descr2 = new float[list2_size*DEGREE_OF_DESCRIPTORS];
    // Store match list2 descriptor index in list1, else -1
    host_match_list = new int[list1_size];

    //  Device
    float *dev_descr1, *dev_descr2;
    int *dev_match_list;
    checkCuda( cudaMalloc( (void**)&dev_descr1, sizeof(float)*list1_size*DEGREE_OF_DESCRIPTORS ) );
    checkCuda( cudaMalloc( (void**)&dev_descr2, sizeof(float)*list2_size*DEGREE_OF_DESCRIPTORS ) );
    checkCuda( cudaMalloc( (void**)&dev_match_list, sizeof(int)*list1_size ) );

    // --- Data transfer ---
    int des_idx = 0;
    for (kpt1 = kpt_list1.begin(); kpt1 != kpt_list1.end(); kpt1++)
    {
        for (int i = 0; i < DEGREE_OF_DESCRIPTORS; i++)
            host_descr1[des_idx++] = kpt1->descriptors[i];
    }
    des_idx = 0;
    for (kpt2 = kpt_list2.begin(); kpt2 != kpt_list2.end(); kpt2++)
    {
        for (int i = 0; i < DEGREE_OF_DESCRIPTORS; i++)
            host_descr2[des_idx++] = kpt2->descriptors[i];
    }
    checkCuda( cudaMemcpy(dev_descr1, host_descr1,
               sizeof(float)*list1_size*DEGREE_OF_DESCRIPTORS, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(dev_descr2, host_descr2,
               sizeof(float)*list2_size*DEGREE_OF_DESCRIPTORS, cudaMemcpyHostToDevice) );

    // --- Call device kernel ---
    int dimBlock = MATCH_TILE, dimGrid = list1_size / MATCH_TILE + 1;
    compare_euc_dis<<<dimGrid, dimBlock>>>(
            dev_descr1, dev_descr2, dev_match_list,
            list1_size, list2_size, ezsift::SIFT_MATCH_NNDR_THR,
            (std::numeric_limits<float>::max)() );

    // --- Copy back and write in match list ---
    checkCuda( cudaMemcpy(host_match_list, dev_match_list,
               sizeof(int)*list1_size, cudaMemcpyDeviceToHost) );
    for (int i = 0; i < list1_size; i++)
    {
        if (host_match_list[i] != -1)
        {
            kpt1 = kpt_list1.begin();
            kpt2 = kpt_list2.begin();
            std::advance(kpt1, i);
            std::advance(kpt2, host_match_list[i]);

            ezsift::MatchPair mp;
            mp.r1 = (int)kpt1->r;
            mp.c1 = (int)kpt1->c;
            mp.r2 = (int)kpt2->r;
            mp.c2 = (int)kpt2->c;

            match_list.push_back(mp);
        }
    }

#if PRINT_MATCH_KEYPOINTS
    std::list<ezsift::MatchPair>::iterator p;
    int match_idx = 0;
    for (p = match_list.begin(); p != match_list.end(); p++) {
        printf("\tMatch %3d: (%4d, %4d) -> (%4d, %4d)\n", match_idx, p->r1,
               p->c1, p->r2, p->c2);
        match_idx++;
    }
#endif

    // --- Clean ---
    //  Host
    delete [] host_descr1; delete [] host_descr2;
    delete [] host_match_list;
    //  Device
    cudaFree(dev_descr1); cudaFree(dev_descr2);
    cudaFree(dev_match_list);
}
