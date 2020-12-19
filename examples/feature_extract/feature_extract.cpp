/*  Copyright (c) 2013, Robert Wang, email: robertwgh (at) gmail.com
    All rights reserved. https://github.com/robertwgh/ezSIFT

    Description: Detect keypoints and extract descriptors from an input image.

    Revision history:
        September 15th, 2013: initial version.
        July 2nd, 2018: code refactor.
*/

#include "ezsift.h"

#include <iostream>
#include <list>
#include <omp.h>

#define USE_FIX_FILENAME 0
int main(int argc, char *argv[])
{

    int num_threads = omp_get_max_threads();
    printf("--- Max thread is %d ---\n", num_threads);
#if USE_FIX_FILENAME
    char *file1 = "img1.pgm";
    printf("--- Using max threads ---\n");
#else
    if (argc != 3) {
        std::cerr
            << "Please input an input image name.\nUsage: feature_extract img"
            << std::endl;
        return -1;
    }
    char file1[128];
    memcpy(file1, argv[1], sizeof(char) * strlen(argv[1]));
    file1[strlen(argv[1])] = 0;
    // Read thread number
    int run_threads = 1;
    run_threads = atoi(argv[2]);
    if (run_threads < 1)
        run_threads = 1;
    else
        run_threads = (num_threads > run_threads) ? run_threads : num_threads;
    printf("--- Now using %d threads ---\n\n", run_threads);
    omp_set_num_threads(run_threads);
#endif

    ezsift::Image<unsigned char> image;
    if (ezsift::read_pgm(file1, image.data, image.w, image.h) != 0) {
        std::cerr << "Failed to open input image." << std::endl;
        return -1;
    }
    std::cout << "Image size: " << image.w << "x" << image.h << std::endl;

    bool bExtractDescriptor = true;
    std::list<ezsift::SiftKeypoint> kpt_list;

    // Double the original image as the first octive.
    ezsift::double_original_image(true);

    // Perform SIFT computation on CPU.
    std::cout << "Start SIFT detection ..." << std::endl;
    ezsift::sift_cpu(image, kpt_list, bExtractDescriptor);

    // Generate output image with keypoints drawing
    char filename[255];
    sprintf(filename, "%s_sift_output.ppm", file1);
    ezsift::draw_keypoints_to_ppm_file(filename, image, kpt_list);

    // Generate keypoints list
    sprintf(filename, "%s_sift_key.key", file1);
    ezsift::export_kpt_list_to_file(filename, kpt_list, bExtractDescriptor);

    std::cout << "\nTotal keypoints number: \t\t"
              << static_cast<unsigned int>(kpt_list.size()) << std::endl;

    return 0;
}
