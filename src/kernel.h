#ifndef KERNEL_H_
#define KERNEL_H_

void row_filter_transpose_gpu(
        float *src, float *dst, int w, int h,
        float *coef1d, int gR);

#endif
