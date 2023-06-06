#ifndef CUDA_STDDEV_H
#define CUDA_STDDEV_H

#include <cuda_runtime.h>
#include <math.h>
#include "cudaUtility.h"


/**
 * Compute the standard deviation of a set of frames (5). The standard deviation is computed for each pixel in the dimensions of the frames.
 * The resulting image highlights pixels that have changed significantly between the frames.
 * For example, if the frames are 640x480, then the standard deviation frame will also be 640x480.
 * @param frame0 The first frame in the set of frames. It is assumed to be a grayscale cuda image.
 * @param frame1 The second frame in the set of frames. It is assumed to be a grayscale cuda image.
 * @param frame2 The third frame in the set of frames. It is assumed to be a grayscale cuda image.
 * @param frame3 The fourth frame in the set of frames. It is assumed to be a grayscale cuda image.
 * @param frame4 The fifth frame in the set of frames. It is assumed to be a grayscale cuda image.
 * @param stdFrame The resulting standard deviation frame. It is assumed to be a grayscale cuda image.
 * @param frameWidth The width of the frames.
 * @param frameHeight The height of the frames.
 */
cudaError_t cudaStdDev(void* frame0, void* frame1, void* frame2, void* frame3, void* frame4, void* stdFrame, size_t frameWidth, size_t frameHeight);

#endif // CUDA_STDDEV_H
