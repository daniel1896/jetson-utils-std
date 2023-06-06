// Helper function to compute standard deviation of an array

#include "cudaStdDev.h"

// CUDA kernel to compute standard deviation frame from consecutive image frames
template<typename T>
__global__ void gpuStdDev(const T* frame0, const T* frame1, const T* frame2, const T* frame3, const T* frame4, T* stdFrame, int frameWidth, int frameHeight)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= frameWidth || y >= frameHeight)
        return;

    // Compute current pixel index
    const int pxlIndex = y * frameWidth + x;

    // Define variables
    T sum = 0, mean = 0, variance = 0;

    // Compute mean
    sum = frame0[pxlIndex] + frame1[pxlIndex] + frame2[pxlIndex] + frame3[pxlIndex] + frame4[pxlIndex];
    mean = sum / 5;

    // Compute variance
    T diff = 0;
    diff = frame0[pxlIndex] - mean;
    variance += diff * diff;
    diff = frame1[pxlIndex] - mean;
    variance += diff * diff;
    diff = frame2[pxlIndex] - mean;
    variance += diff * diff;
    diff = frame3[pxlIndex] - mean;
    variance += diff * diff;
    diff = frame4[pxlIndex] - mean;
    variance += diff * diff;
    variance /= 5;

    // Compute standard deviation
    stdFrame[pxlIndex] = pow(sqrt(variance) / 255, 0.6);
}


cudaError_t cudaStdDev(void* frame0, void* frame1, void* frame2, void* frame3, void* frame4, void* stdFrame, size_t frameWidth, size_t frameHeight)
{
    const dim3 blockDim(32, 8);
    const dim3 gridDim(iDivUp(frameWidth, blockDim.x), iDivUp(frameHeight, blockDim.y));

    gpuStdDev<float><<<gridDim, blockDim>>>((float*) frame0, (float*) frame1, (float*) frame2, (float*) frame3, (float*) frame4, (float*) stdFrame, frameWidth, frameHeight);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        return error;
    }

    return cudaSuccess;
}
