

#include "config.h"

#if MODE==USE_CUDA
#include "wspsnr_cuda.cuh"
#include <chrono>

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
__global__ void computeWSPSNR(
    const int* ref,         // [NUM_FRAMES * W * H]
    const int* dist,        // [NUM_FRAMES * W * H]
    const double* row_weights, // [H]
    double* sew,                        // [NUM_FRAMES]
    int width,
    int height
) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    int x = tx + blockIdx.x * blockDim.x;
    int y = ty + blockIdx.y * blockDim.y;
    int f = blockIdx.z;  // Frame index

    int frame_stride = width * height;
    int idx = f * frame_stride + y * width + x;


    if (x < width && y < height) {
        int ref_val = ref[idx];
        int dist_val = dist[idx];
        int diff = ref_val - dist_val;
       
        int sqdiff = diff * diff;
        double w = __ldg(&row_weights[y]);
        sew[idx] = sqdiff*w;  // Read-only cache optimized
      

    }
}





__global__ void computeWSPSNR_atomic(
    const int* ref,         // [NUM_FRAMES * W * H]
    const int* dist,        // [NUM_FRAMES * W * H]
    const double* row_weights, // [H]
    double* sse_out,                        // [NUM_FRAMES]
    int width,
    int height
) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    int x = tx + blockIdx.x * blockDim.x;
    int y = ty + blockIdx.y * blockDim.y;
    int f = blockIdx.z;  // Frame index

    int frame_stride = width * height;
    int idx = f * frame_stride + y * width + x;


    if (x < width && y < height) {
        int ref_val = ref[idx];
        int dist_val = dist[idx];
        int diff = ref_val - dist_val;
        int sse = diff * diff;

        double weight = __ldg(&row_weights[y]);  // Read-only cache optimized
        
        // Atomic add to per-frame MSE output
        atomicAdd(&sse_out[f], (sse * weight));

    }
}


__global__ void computeWSPSNRShared2(
    const int* ref,
    const int* dist,
    const double* row_weights,
    double* partial_sums,   // [NUM_FRAMES][num_blocks]
    int width,
    int height
) {
    __shared__ double smem[NUM_THREADS * NUM_THREADS];  // assuming blockDim.x * blockDim.y <= 256

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    int x = tx + blockIdx.x * blockDim.x;
    int y = ty + blockIdx.y * blockDim.y;
    int f = blockIdx.z;

    int frame_stride = width * height;

        double local_sum = 0.0f;
        if (x < width && y < height) {
            int idx = f * frame_stride + y * width + x;
            int diff = ref[idx] - dist[idx];
            int sse = diff * diff;
            double weight = __ldg(&row_weights[y]); // Read-only cache optimized
            local_sum = sse * weight;
        }

        smem[tid] = local_sum;
        __syncthreads();

        // Block-wide reduction
        for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
            if (tid < s)
                smem[tid] += smem[tid + s];
            __syncthreads();
        }


        // Only thread 0 writes the result
        if (tid == 0) {
            int block_idx = blockIdx.y * gridDim.x + blockIdx.x;
            partial_sums[f * gridDim.x * gridDim.y + block_idx] = smem[0];
        }
   
}




__global__ void computeWSPSNRShared(
    const int* ref,
    const int* dist,
    const double* row_weights,
    double* partial_sums,   // [NUM_FRAMES][num_blocks]
    int width,
    int height
) {
    __shared__ double smem[NUM_THREADS * NUM_THREADS];  // assuming blockDim.x * blockDim.y <= 256

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    int x = tx + blockIdx.x * blockDim.x;
    int y = ty + blockIdx.y * blockDim.y;
    int f = blockIdx.z;

    int frame_stride = width * height;

        double local_sum = 0.0f;
        //for (int t = 0; t < 10000; t++)
        {
        if (x < width && y < height) {
            int idx = f * frame_stride + y * width + x;
            int diff = ref[idx] - dist[idx];
            int sse = diff * diff;
            double weight = __ldg(&row_weights[y]); // Read-only cache optimized
            local_sum = sse * weight;
        }
    }
        smem[tid] = local_sum;
        __syncthreads();

        // Block-wide reduction
        for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
            if (tid < s)
                smem[tid] += smem[tid + s];
            __syncthreads();
        }


        // Only thread 0 writes the result
        if (tid == 0) {
            int block_idx = blockIdx.y * gridDim.x + blockIdx.x;
            partial_sums[f * gridDim.x * gridDim.y + block_idx] = smem[0];
        }
    
}

cudaError_t run_process_cuda_atomic(const int* ref, const int* rec, const double* weights, double* wspsnr_frame, const double w_sum, const  int w, int h, int bitDepth, int nf = 1)
{

    int* dev_ref = 0;
    int* dev_rec = 0;
    //__constant__  double* dev_weights = new double[h];
    double* dev_weights = 0;
    double* dev_sseout = 0;
    cudaError_t cudaStatus;
    double MAX_VALUE = (255.0 * (1 << (bitDepth - 8)));



    double* host_sseout = new double[nf];
    int size = w * h * nf;

    /*sdouble* h_row_weights = new double[h];
    for (int y = 0; y < h; ++y)
    {
        double theta = ((y + 0.5f) / h - 0.5f) * M_PI;
        h_row_weights[y] = cosf(theta);
    }*/
    //cudaMemcpyToSymbol(dev_weights, weights, size * sizeof(double));

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_ref, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_rec, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_sseout, nf * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_weights, h * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_ref, ref, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_rec, rec, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_weights, weights, h * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

   
    dim3 blockDim(NUM_THREADS, NUM_THREADS);
    dim3 gridDim((w + NUM_THREADS - 1) / NUM_THREADS, (h + NUM_THREADS - 1) / NUM_THREADS, nf);

    computeWSPSNR_atomic << < gridDim, blockDim >> > (dev_ref, dev_rec, dev_weights, dev_sseout, w, h);


    cudaStatus = cudaMemcpy(host_sseout, dev_sseout, nf * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    for (int f = 0; f < nf; f++)
        wspsnr_frame[f] = 10 * std::log10((MAX_VALUE * MAX_VALUE * w_sum) / host_sseout[f]);


Error:
    cudaFree(dev_ref);
    cudaFree(dev_rec);
    cudaFree(dev_weights);
    cudaFree(dev_sseout);
    cudaFree(host_sseout);

    return cudaStatus;
}


cudaError_t run_process_cuda_shared_mem(const int* ref, const int* rec, const double* weights, double* wspsnr_frame, const double w_sum, const  int w, int h, int bitDepth, int nf = 1)
{

    int* dev_ref = 0;
    int* dev_rec = 0;
    //__constant__  double* dev_weights = new double[h];
    double* dev_weights = 0;
    double* dev_partial_sum = 0;
    cudaError_t cudaStatus;
    double MAX_VALUE = (255.0 * (1 << (bitDepth - 8)));

    int size = w * h * nf;

    dim3 blockDim(NUM_THREADS, NUM_THREADS);
    dim3 gridDim((w + NUM_THREADS - 1) / NUM_THREADS, (h + NUM_THREADS - 1) / NUM_THREADS, nf);
    
    int num_blocks = gridDim.x * gridDim.y;
    int partial_sum_num = num_blocks * gridDim.z;   //num_frames * num_blocks
    double* h_partial_sum = new double[partial_sum_num];

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_ref, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_rec, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_partial_sum, partial_sum_num * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_weights, h * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_ref, ref, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_rec, rec, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_weights, weights, h * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    auto st = std::chrono::high_resolution_clock::now();
    computeWSPSNRShared << <gridDim, blockDim >> > ( dev_ref, dev_rec, dev_weights, dev_partial_sum,w, h );

    cudaStatus = cudaMemcpy(h_partial_sum, dev_partial_sum, partial_sum_num * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    for (int f = 0; f < nf; ++f) {
        double weighted_mse = 0.0f;
        for (int b = 0; b < num_blocks; ++b) {
            weighted_mse += h_partial_sum[f * num_blocks + b];

            wspsnr_frame[f] = 10 * std::log10((MAX_VALUE * MAX_VALUE * w_sum) / weighted_mse);
        }
    }
    auto et = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count() / 1000.0;
    std::cout << duration << std::endl;


Error:
    cudaFree(dev_ref);
    cudaFree(dev_rec);
    cudaFree(dev_weights);
    free(h_partial_sum);
    cudaFree(dev_partial_sum);

    return cudaStatus;
}


void intit_device_buffer(int** dev_ref, int** dev_rec, double** dev_weights , double** dev_partial_sum, const  int w, int h, const double* weights,int nf=1)
{
    
    cudaError_t cudaStatus;

    int size = w * h * nf;

    dim3 blockDim(NUM_THREADS, NUM_THREADS);
    dim3 gridDim((w + NUM_THREADS - 1) / NUM_THREADS, (h + NUM_THREADS - 1) / NUM_THREADS, nf);

    int num_blocks = gridDim.x * gridDim.y;
    int partial_sum_num = num_blocks * gridDim.z;   //num_frames * num_blocks

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&(*dev_ref), size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        
    }

    cudaStatus = cudaMalloc((void**)&(*dev_rec), size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
       
    }

    cudaStatus = cudaMalloc((void**)&(*dev_partial_sum), size * sizeof(double));
    //cudaStatus = cudaMalloc((void**)&(*dev_partial_sum), partial_sum_num * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
       
    }

    cudaStatus = cudaMalloc((void**)&(*dev_weights), h * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        
    }


    cudaStatus = cudaMemcpy((*dev_weights), weights, h * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
      
    }
}

cudaError_t run_process_cuda_shared_mem(const int* ref, const int* rec, int* dev_ref, int* dev_rec, double* dev_weights, double* dev_partial_sum,const double* weights,  double* wspsnr_frame, const double w_sum, const  int w, int h, int bitDepth, int nf = 1)
{

    /*int* dev_ref = 0;
    int* dev_rec = 0;
    //__constant__  double* dev_weights = new double[h];
    double* dev_weights = 0;
    double* dev_partial_sum = 0;*/

    cudaError_t cudaStatus;
    double MAX_VALUE = (255.0 * (1 << (bitDepth - 8)));

    int size = w * h * nf;

    dim3 blockDim(NUM_THREADS, NUM_THREADS);
    dim3 gridDim((w + NUM_THREADS - 1) / NUM_THREADS, (h + NUM_THREADS - 1) / NUM_THREADS, nf);

    int num_blocks = gridDim.x * gridDim.y;
    int partial_sum_num = num_blocks * gridDim.z;   //num_frames * num_blocks
    
    double* h_partial_sum = new double[partial_sum_num];

    // Allocate GPU buffers for three vectors (two input, one output)    .
    /*cudaStatus = cudaMalloc((void**)&dev_ref, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_rec, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_partial_sum, partial_sum_num * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_weights, h * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }*/

   
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_ref, ref, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_rec, rec, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    /*
    cudaStatus = cudaMemcpy(dev_weights, weights, h * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }*/
    auto st = std::chrono::high_resolution_clock::now();
    computeWSPSNRShared << <gridDim, blockDim >> > (dev_ref, dev_rec, dev_weights, dev_partial_sum, w, h);

    cudaStatus = cudaMemcpy(h_partial_sum, dev_partial_sum, partial_sum_num * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    for (int f = 0; f < nf; ++f) {
        double weighted_mse = 0.0f;
        for (int b = 0; b < num_blocks; ++b) {
            weighted_mse += h_partial_sum[f * num_blocks + b];

            wspsnr_frame[f] = 10 * std::log10((MAX_VALUE * MAX_VALUE * w_sum) / weighted_mse);
        }
    }
    auto et = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count() / 1000.0;
    std::cout << duration << std::endl;


Error:
    //cudaFree(dev_ref);
    //cudaFree(dev_rec);
    //cudaFree(dev_weights);
    //cudaFree(dev_partial_sum);
    free(h_partial_sum);
    
    return cudaStatus;
}


cudaError_t run_process_cuda_shared_mem_2streams(const int* ref, const int* rec, const double* weights, double* wspsnr_frame, const double w_sum, const  int w, int h, int bitDepth, int nf = 1)
{

    cudaError_t cudaStatus;
    double MAX_VALUE = (255.0 * (1 << (bitDepth - 8)));
    int size = w * h ;
    int framesize = (w * h);

    dim3 blockDim(NUM_THREADS, NUM_THREADS);
    dim3 gridDim((w + NUM_THREADS - 1) / NUM_THREADS, (h + NUM_THREADS - 1) / NUM_THREADS, nf/2);

    int num_blocks = gridDim.x * gridDim.y;
    int partial_sum_num = num_blocks * gridDim.z;   //num_frames * num_blocks

    double* h_partial_sum = new double[partial_sum_num* nf];


    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    double* dev_weights = 0;

    cudaStatus = cudaMalloc((void**)&dev_weights, h * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


    cudaStatus = cudaMemcpy(dev_weights, weights, h * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    int* dev_ref = 0;
    int* dev_rec = 0;
    double* dev_partial_sum = 0;

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_ref, framesize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_rec, framesize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_partial_sum, num_blocks * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_ref, ref, framesize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_rec, rec, framesize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    computeWSPSNRShared << <gridDim, blockDim ,0, stream1 >> > (dev_ref, dev_rec, dev_weights, dev_partial_sum, w, h);


    int* dev_ref1 = 0;
    int* dev_rec1 = 0;
    double* dev_partial_sum1 = 0;
    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_ref1, framesize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_rec1, framesize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_partial_sum1, num_blocks * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_ref1, ref+ framesize, framesize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_rec1, rec+ framesize, framesize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    computeWSPSNRShared << <gridDim, blockDim, 0, stream2 >> > (dev_ref1, dev_rec1, dev_weights, dev_partial_sum1, w, h);


    // Optional: wait for both to finish
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    cudaStatus = cudaMemcpy(h_partial_sum, dev_partial_sum, num_blocks * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(h_partial_sum+ num_blocks, dev_partial_sum1, num_blocks * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    for (int f = 0; f < nf; ++f) {
        double weighted_mse = 0.0f;
        for (int b = 0; b < num_blocks; ++b) {
            weighted_mse += h_partial_sum[f * num_blocks + b];

            wspsnr_frame[f] = 10 * std::log10((MAX_VALUE * MAX_VALUE * w_sum) / weighted_mse);
        }
    }
    // Cleanup
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);


Error:
    cudaFree(dev_ref);
    cudaFree(dev_rec);
    cudaFree(dev_ref1);
    cudaFree(dev_rec1);
    cudaFree(dev_weights);
    free(h_partial_sum);


    return cudaStatus;
}



cudaError_t run_process_cuda(const int* ref, const int* rec, int* dev_ref, int* dev_rec, double* dev_weights, double* sqdiff_weighted, const double* weights, double* wspsnr_frame, const double w_sum, const  int w, int h, int bitDepth, int nf = 1)
{


    cudaError_t cudaStatus;
    double MAX_VALUE = (255.0 * (1 << (bitDepth - 8)));

    int size = w * h * nf;

    dim3 blockDim(NUM_THREADS, NUM_THREADS);
    dim3 gridDim((w + NUM_THREADS - 1) / NUM_THREADS, (h + NUM_THREADS - 1) / NUM_THREADS, nf);

    int num_blocks = gridDim.x * gridDim.y;
    //int partial_sum_num = num_blocks * gridDim.z;   //num_frames * num_blocks

    //double* h_partial_sum = new double[partial_sum_num];

    // Allocate GPU buffers for three vectors (two input, one output)    .
    /*cudaStatus = cudaMalloc((void**)&dev_ref, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_rec, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_partial_sum, partial_sum_num * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_weights, h * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }*/

    
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_ref, ref, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_rec, rec, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    /*
    cudaStatus = cudaMemcpy(dev_weights, weights, h * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }*/
    auto st = std::chrono::high_resolution_clock::now();
    computeWSPSNR << <gridDim, blockDim >> > (dev_ref, dev_rec, dev_weights, sqdiff_weighted, w, h);

   /* cudaStatus = cudaMemcpy(h_partial_sum, sqdiff_weighted, partial_sum_num * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }*/


    for (int f = 0; f < nf; ++f) {

        double weighted_sse = thrust::reduce(
            thrust::device_pointer_cast(sqdiff_weighted + (w * h * f)),
            thrust::device_pointer_cast(sqdiff_weighted + (w * h * (f + 1))), 
             0.0f, thrust::plus<double>());

        wspsnr_frame[f] = 10 * std::log10((MAX_VALUE * MAX_VALUE * w_sum) / weighted_sse);
    }
    auto et = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count() / 1000.0;
    std::cout << duration << std::endl;


Error:
    //cudaFree(dev_ref);
    //cudaFree(dev_rec);
    //cudaFree(dev_weights);
    //cudaFree(dev_partial_sum);
    //free(h_partial_sum);

    return cudaStatus;
}







#endif
