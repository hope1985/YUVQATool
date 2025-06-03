#ifndef CUDA_CUH
#define CUDA_CUH


#include "config.h"

#if MODE==USE_CUDA

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <stdio.h>


#define NUM_THREADS  32

__global__ void computeWSPSNR_atomic(
    const int* ref,         // [NUM_FRAMES * W * H]
    const int* dist,        // [NUM_FRAMES * W * H]
    const double* row_weights, // [H]
    double* mse_out,                        // [NUM_FRAMES]
    int width,
    int height
);

__global__ void computeWSPSNRShared(
    const int* ref,
    const int* dist,
    const double* row_weights,
    double* partial_sums,   // [NUM_FRAMES][num_blocks]
    int width,
    int height
);

__global__ void computeWSPSNR(
    const int* ref,         // [NUM_FRAMES * W * H]
    const int* dist,        // [NUM_FRAMES * W * H]
    const double* row_weights, // [H]
    double* sew,                        // [NUM_FRAMES]
    int width,
    int height
);


void intit_device_buffer(int** dev_ref, int** dev_rec, double** dev_weights, double** dev_partial_sum, const  int w, int h, const double* weights, int nf );

cudaError_t wspsnr_process_cuda_shared_mem_2streams(const int* ref, const int* rec, const double* weights, double* wspsnr_frame, const double w_sum, const  int w, int h, int bitDepth, int nf);

cudaError_t wspsnr_process_cuda_atomic(const int* ref, const int* rec, const double* weights, double* wspsnr_frame, const double w_sum, const  int w, int h, int bitDepth, int nf );

cudaError_t wspsnr_process_cuda_shared_mem(const int* ref, const int* rec, int* dev_ref, int* dev_rec, double* dev_weights, double* dev_partial_sum,const double* weights, double* wspsnr_frame, const double w_sum, const  int w, int h, int bitDepth, int nf );


//faster than cuda atomic 
cudaError_t wspsnr_process_cuda_shared_mem(const int* ref, const int* rec, const double* weights, double* wspsnr_frame, const double w_sum, const  int w, int h, int bitDepth, int nf );
cudaError_t wspsnr_process_cuda(const int* ref, const int* rec, int* dev_ref, int* dev_rec, double* dev_weights, double* sqdiff_weighted, const double* weights, double* wspsnr_frame, const double w_sum, const  int w, int h, int bitDepth, int nf );


cudaError_t psnr_process_cuda_shared_mem(const int* ref, const int* rec, int* dev_ref, int* dev_rec, double* dev_partial_sum, double* wspsnr_frame, const  int w, int h, int bitDepth, int nf);


#endif //USE_CUDA

#endif  //CUDA_CUH