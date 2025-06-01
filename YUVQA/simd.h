#ifndef SMID_H
#define SMID_H

#include "config.h"

#if MODE==USE_SIMD  

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <chrono>
#include <algorithm>    // std::transform
#include <numeric>      // std::accumulate
#include <omp.h>
#include <cstdlib>      // for _aligned_malloc and _aligned_free
#include <immintrin.h>  // for SIMD operations
//############## NOTE : GCC compiler need this ##############
#include <cstring>      

using namespace std;

static double simd_sum(double* data, size_t len) {
    __m256d acc = _mm256_setzero_pd();
    size_t i = 0;

    for (; i + 3 < len; i += 4) {
        __m256d v = _mm256_loadu_pd(&data[i]);
        acc = _mm256_add_pd(acc, v);
    }

    // Horizontal add the SIMD accumulator
    double temp[4];
    _mm256_storeu_pd(temp, acc);
    double sum = temp[0] + temp[1] + temp[2] + temp[3];

    // Add remaining elements
    for (; i < len; ++i) {
        sum += data[i];
    }

    return sum;
}

template<class T>
static double simd_sse(const T* refPic, const T* recPic, double* sse_row, int W, int H,int lh,int uh,int lw,int uw)
{

#pragma omp parallel for 
    for (int hi = lh; hi < uh; ++hi)
    {
        // std::cout << "num_threads:" << omp_get_num_threads() << std::endl;

        int hi_roi = hi - lh;
        int start = hi * W;
        int i = lw;

#if COMPUTE_DTYPE_IDX == DTYPE_FLOAT
        // SIMD subtraction using AVX(8 floats per iteration)

        __m256 acc = _mm256_setzero_ps();  // for float

        //__m256 acc3 = _mm256_setzero_ps();   //Fake operation

        for (; i <= uw - 8; i += 8) {

            //Fake operation
            /* __m256 acc2 = _mm256_setzero_ps();
            for (int t = 0; t < 10000; t++)
            {
                __m256 av1 = _mm256_loadu_ps(&refPic[i + start]);
                __m256 bv1 = _mm256_loadu_ps(&recPic[i + start]);
                __m256 diff1 = _mm256_sub_ps(av1, bv1); // diff = a - b
                acc2 = _mm256_fmadd_ps(diff1, diff1, acc2); // acc += diff * diff
            }
            acc3 = acc2;*/

            __m256 av = _mm256_loadu_ps(&refPic[i + start]);
            __m256 bv = _mm256_loadu_ps(&recPic[i + start]);
            __m256 diff = _mm256_sub_ps(av, bv); // diff = a - b
            acc = _mm256_fmadd_ps(diff, diff, acc); // acc += diff * diff
        }

        float temp[8];
        _mm256_storeu_ps(temp, acc);
        sse_row[hi_roi] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
        //_mm256_storeu_ps(temp, acc3); //Fake operation

        // Handle remaining elements
        for (; i < uw; ++i) {
            float diff = refPic[i + start] - recPic[i + start];;
            sse_row[hi_roi] += diff * diff;
        }

#elif COMPUTE_DTYPE_IDX == DTYPE_DOUBLE
        __m256d acc = _mm256_setzero_pd();
        for (; i <= uw - 4; i += 4)
        {
            __m256d av = _mm256_loadu_pd(&refPic[i + start]);
            __m256d bv = _mm256_loadu_pd(&recPic[i + start]);
            __m256d diff = _mm256_sub_pd(av, bv);        // diff = a - b
            acc = _mm256_fmadd_pd(diff, diff, acc);      // equivalent to acc += diff*diff
        }

        double temp[4];
        _mm256_storeu_pd(temp, acc);
        sse_row[hi_roi] = temp[0] + temp[1] + temp[2] + temp[3];

        // Handle remaining elements
        for (; i < uw; ++i) {
            double diff = refPic[i + start] - recPic[i + start];
            sse_row[hi_roi] += diff * diff;
        }

#elif COMPUTE_DTYPE_IDX == DTYPE_INT
        __m256i acc = _mm256_setzero_si256();  // accumulator for sum of squares

        for (; i <= uw - 8; i += 8)
        { // 8*32bit = 256bit

            __m256i av = _mm256_loadu_si256((__m256i const*) & refPic[i + start]);
            __m256i bv = _mm256_loadu_si256((__m256i const*) & recPic[i + start]);
            __m256i diff = _mm256_sub_epi32(av, bv);      // diff = a - b
            __m256i diff_sq = _mm256_mullo_epi32(diff, diff); // diff * diff
            acc = _mm256_add_epi32(acc, diff_sq);            // acc += diff*diff
        }

        int temp[8];
        _mm256_storeu_si256((__m256i*)temp, acc);

        //############## NOTE : GCC does not support this ##############
        //_mm256_storeu_epi32(temp, acc);          

        sse_row[hi_roi] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

        // Handle remaining elements
        for (; i < uw; ++i) {
            int diff = refPic[i + start] - recPic[i + start];
            sse_row[hi_roi] += diff * diff;
        }

#endif


    }

}

#if COMPUTE_DTYPE_IDX == DTYPE_FLOAT
static double wspsnr_openmp_simd(const float* refPic, const float* recPic, int W, int H, int bitDepth, const vector<double> weights, vector<int> *roi=NULL)
#elif COMPUTE_DTYPE_IDX == DTYPE_DOUBLE
static double wspsnr_openmp_simd(const double* refPic, const  double* recPic, int W, int H, int bitDepth, const vector<double> weights, vector<int>* roi = NULL)
#elif COMPUTE_DTYPE_IDX == DTYPE_INT
static double wspsnr_openmp_simd(const int* refPic, const int* recPic, int W, int H, int bitDepth, const vector<double> weights, vector<int>* roi = NULL)
#endif 
{
    // Calculate MAX_VALUE based on bitDepth
    double MAX_VALUE = (255.0 * (1 << (bitDepth - 8)));
    double wspsnr = 0.0;

    int lh = 0;
    int uh = H;
    int lw = 0;
    int uw = W;
    int roiH = H;
    int roiW = W;
    if (roi != NULL)
    {
         lh = (*roi)[0];
         uh = (*roi)[1];
         lw = (*roi)[2];
         uw = (*roi)[3];
         roiH = uh - lh;
         roiW = uw - lw;
    }
    
#ifdef _WIN32
    double* sse_row = (double*)_aligned_malloc(roiH * sizeof(double), 32);
    double* w_sum_row = (double*)_aligned_malloc(roiH * sizeof(double), 32);
#else
    double* sse_row = (double*)aligned_alloc(32, roiH * sizeof(double));
    double* w_sum_row = (double*)aligned_alloc(32, roiH * sizeof(double));
#endif

    //int num_threads = 4;  
    //omp_set_num_threads(num_threads);
    //std::cout << "num_threads:" << omp_get_num_threads() << std::endl;

   /*
#pragma omp parallel for 
    for (int hi = lh; hi < uh; ++hi)
    {
        // std::cout << "num_threads:" << omp_get_num_threads() << std::endl;
        
        int hi_roi = hi - lh;
        int start = hi * W;
        int i = lw;
      

#if COMPUTE_DTYPE_IDX == DTYPE_FLOAT
        // SIMD subtraction using AVX(8 floats per iteration)

        __m256 acc = _mm256_setzero_ps();  // for float

        //__m256 acc3 = _mm256_setzero_ps();   //Fake operation

        for (; i <= uw - 8; i += 8) {

            //Fake operation
            // __m256 acc2 = _mm256_setzero_ps();  
            //for (int t = 0; t < 10000; t++)
            //{
            //    __m256 av1 = _mm256_loadu_ps(&refPic[i + start]);
            //    __m256 bv1 = _mm256_loadu_ps(&recPic[i + start]);
            //    __m256 diff1 = _mm256_sub_ps(av1, bv1); // diff = a - b
            //    acc2 = _mm256_fmadd_ps(diff1, diff1, acc2); // acc += diff * diff
            //}
            //acc3 = acc2;

            __m256 av = _mm256_loadu_ps(&refPic[i + start]);
            __m256 bv = _mm256_loadu_ps(&recPic[i + start]);
            __m256 diff = _mm256_sub_ps(av, bv); // diff = a - b
            acc = _mm256_fmadd_ps(diff, diff, acc); // acc += diff * diff
        }

        float temp[8];
        _mm256_storeu_ps(temp, acc);
        sse_row[hi_roi] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
        //_mm256_storeu_ps(temp, acc3); //Fake operation

        // Handle remaining elements
        for (; i < uw; ++i) {
            float diff = refPic[i + start] - recPic[i + start];;
            sse_row[hi_roi] += diff * diff;
        }

#elif COMPUTE_DTYPE_IDX == DTYPE_DOUBLE
        __m256d acc = _mm256_setzero_pd();
        for (; i <= uw - 4; i += 4) 
        {
            __m256d av = _mm256_loadu_pd(&refPic[i + start]);
            __m256d bv = _mm256_loadu_pd(&recPic[i + start]);
            __m256d diff = _mm256_sub_pd(av, bv);        // diff = a - b
            acc = _mm256_fmadd_pd(diff, diff, acc);      // equivalent to acc += diff*diff
        }

        double temp[4];
        _mm256_storeu_pd(temp, acc);
        sse_row[hi_roi] = temp[0] + temp[1] + temp[2] + temp[3];

        // Handle remaining elements
        for (; i < uw; ++i) {
            double diff = refPic[i + start] - recPic[i + start];
            sse_row[hi_roi] += diff * diff;
        }

#elif COMPUTE_DTYPE_IDX == DTYPE_INT
        __m256i acc = _mm256_setzero_si256();  // accumulator for sum of squares
        
        for (; i <= uw - 8; i += 8)
        { // 8*32bit = 256bit

            __m256i av = _mm256_loadu_si256((__m256i const*) & refPic[i + start]);
            __m256i bv = _mm256_loadu_si256((__m256i const*) & recPic[i + start]);
            __m256i diff = _mm256_sub_epi32(av, bv);      // diff = a - b
            __m256i diff_sq = _mm256_mullo_epi32(diff, diff); // diff * diff
            acc = _mm256_add_epi32(acc, diff_sq);            // acc += diff*diff
        }

        int temp[8];
        _mm256_storeu_si256((__m256i*)temp, acc);
        
        //############## NOTE : GCC does not support this ##############
        //_mm256_storeu_epi32(temp, acc);          

        sse_row[hi_roi] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

        // Handle remaining elements
        for (; i < uw; ++i) {
            int diff = refPic[i + start] - recPic[i + start];
            sse_row[hi_roi] += diff * diff;
        }

#endif

        sse_row[hi_roi] = sse_row[hi_roi] * weights[hi];
        w_sum_row[hi_roi] = (weights[hi] * roiW);

    }
   
   */
    
    simd_sse(refPic, recPic, sse_row, W, H, lh, uh, lw, uw);
#pragma omp parallel for 
    for (int hi = lh; hi < uh; ++hi)
    {
        int hi_roi = hi - lh;
        sse_row[hi_roi] = sse_row[hi_roi] * weights[hi];
        w_sum_row[hi_roi] = (weights[hi] * roiW);
    } 

    
    double sse_sum = simd_sum(sse_row, roiH);
    double w_sum = simd_sum(w_sum_row, roiH);

#ifdef _WIN32
    _aligned_free(sse_row);
    _aligned_free(w_sum_row);
#else
    free(sse_row);
    free(w_sum_row);
#endif

    wspsnr = 10 * std::log10((MAX_VALUE * MAX_VALUE * w_sum) / sse_sum);
    return wspsnr;
}


#if COMPUTE_DTYPE_IDX == DTYPE_FLOAT
static double psnr_openmp_simd(const float* refPic, const float* recPic, int W, int H, int bitDepth, vector<int>* roi = NULL)
#elif COMPUTE_DTYPE_IDX == DTYPE_DOUBLE
static double psnr_openmp_simd(const double* refPic, const  double* recPic, int W, int H, int bitDepth vector<int>* roi = NULL)
#elif COMPUTE_DTYPE_IDX == DTYPE_INT
static double psnr_openmp_simd(const int* refPic, const int* recPic, int W, int H, int bitDepth, vector<int>* roi = NULL)
#endif 
{
    // Calculate MAX_VALUE based on bitDepth
    double MAX_VALUE = (255.0 * (1 << (bitDepth - 8)));
    double psnr = 0.0;

    int lh = 0;
    int uh = H;
    int lw = 0;
    int uw = W;
    int roiH = H;
    int roiW = W;
    if (roi != NULL)
    {
        lh = (*roi)[0];
        uh = (*roi)[1];
        lw = (*roi)[2];
        uw = (*roi)[3];
        roiH = uh - lh;
        roiW = uw - lw;
    }

#ifdef _WIN32
    double* sse_row = (double*)_aligned_malloc(roiH * sizeof(double), 32);
#else
    double* sse_row = (double*)aligned_alloc(32, roiH * sizeof(double));
#endif

    //int num_threads = 4;  
    //omp_set_num_threads(num_threads);
    //std::cout << "num_threads:" << omp_get_num_threads() << std::endl;

/*
#pragma omp parallel for 
    for (int hi = lh; hi < uh; ++hi)
    {
        // std::cout << "num_threads:" << omp_get_num_threads() << std::endl;

        int hi_roi = hi - lh;
        int start = hi * W;
        int i = lw;

#if COMPUTE_DTYPE_IDX == DTYPE_FLOAT
        // SIMD subtraction using AVX(8 floats per iteration)

        __m256 acc = _mm256_setzero_ps();  // for float

        // __m256 acc3 = _mm256_setzero_ps();   //Fake operation

        for (; i <= uw - 8; i += 8) {

            //Fake operation
            // __m256 acc2 = _mm256_setzero_ps();
            //for (int t = 0; t < 10000; t++)
            //{
            //    __m256 av1 = _mm256_loadu_ps(&refPic[i + start]);
            //    __m256 bv1 = _mm256_loadu_ps(&recPic[i + start]);
            //    __m256 diff1 = _mm256_sub_ps(av1, bv1); // diff = a - b
            //    acc2 = _mm256_fmadd_ps(diff1, diff1, acc2); // acc += diff * diff
            //}
            //acc3 = acc2;

            __m256 av = _mm256_loadu_ps(&refPic[i + start]);
            __m256 bv = _mm256_loadu_ps(&recPic[i + start]);
            __m256 diff = _mm256_sub_ps(av, bv); // diff = a - b
            acc = _mm256_fmadd_ps(diff, diff, acc); // acc += diff * diff
        }

        float temp[8];
        _mm256_storeu_ps(temp, acc);
        sse_row[hi_roi] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
        //_mm256_storeu_ps(temp, acc3); //Fake operation

        // Handle remaining elements
        for (; i < uw; ++i) {
            float diff = refPic[i + start] - recPic[i + start];;
            sse_row[hi_roi] += diff * diff;
        }

#elif COMPUTE_DTYPE_IDX == DTYPE_DOUBLE
        __m256d acc = _mm256_setzero_pd();
        for (; i <= uw - 4; i += 4)
        {
            __m256d av = _mm256_loadu_pd(&refPic[i + start]);
            __m256d bv = _mm256_loadu_pd(&recPic[i + start]);
            __m256d diff = _mm256_sub_pd(av, bv);        // diff = a - b
            acc = _mm256_fmadd_pd(diff, diff, acc);      // equivalent to acc += diff*diff
        }

        double temp[4];
        _mm256_storeu_pd(temp, acc);
        sse_row[hi_roi] = temp[0] + temp[1] + temp[2] + temp[3];

        // Handle remaining elements
        for (; i < uw; ++i) {
            double diff = refPic[i + start] - recPic[i + start];
            sse_row[hi_roi] += diff * diff;
        }

#elif COMPUTE_DTYPE_IDX == DTYPE_INT
        __m256i acc = _mm256_setzero_si256();  // accumulator for sum of squares

        for (; i <= uw - 8; i += 8)
        { // 8*32bit = 256bit

            __m256i av = _mm256_loadu_si256((__m256i const*) & refPic[i + start]);
            __m256i bv = _mm256_loadu_si256((__m256i const*) & recPic[i + start]);
            __m256i diff = _mm256_sub_epi32(av, bv);      // diff = a - b
            __m256i diff_sq = _mm256_mullo_epi32(diff, diff); // diff * diff
            acc = _mm256_add_epi32(acc, diff_sq);            // acc += diff*diff
        }

        int temp[8];
        _mm256_storeu_si256((__m256i*)temp, acc);

        //############## NOTE : GCC does not support this ##############
        //_mm256_storeu_epi32(temp, acc);          

        sse_row[hi_roi] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

        // Handle remaining elements
        for (; i < uw; ++i) {
            int diff = refPic[i + start] - recPic[i + start];
            sse_row[hi_roi] += diff * diff;
        }

#endif
    }*/
    simd_sse(refPic, recPic, sse_row, W, H, lh, uh, lw, uw);

    double sse_sum = simd_sum(sse_row, roiH);
    sse_sum = sse_sum / (roiH * roiW);
#ifdef _WIN32
    _aligned_free(sse_row);
   
#else
    free(sse_row);
    free(w_sum_row);
#endif

    psnr = 10 * std::log10((MAX_VALUE * MAX_VALUE) / sse_sum);
    return psnr;
}


#endif //MODE==USE_SIMD  

#endif  //SMID_H