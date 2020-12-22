#ifndef KERNEL_H_
#define KERNEL_H_

#include "image_utility.h"
#include "common.h"
#include "image.h"
#include "ezsift.h"

void row_filter_transpose_gpu(
        float *src, float *dst, int w, int h,
        float *coef1d, int gR);

void row_filter_transpose_gpu_opt(
        float *src, float *dst, int w, int h,
        float *coef1d, int gR);


void match_keypoints_gpu(std::list<ezsift::SiftKeypoint> &kpt_list1,
                         std::list<ezsift::SiftKeypoint> &kpt_list2,
                         std::list<ezsift::MatchPair> &match_list);

#endif
