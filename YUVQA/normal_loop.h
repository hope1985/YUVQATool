#ifndef NORMAL_LOOP_H
#define NORMAL_LOOP_H

#include <vector>
#include <algorithm>    // std::transform
#include <numeric>     // std::accumulate
#include <string>
#include <cmath>
//############## NOTE : GCC compiler need this ##############
#include <cstring>   // need for GCC

using namespace std;


template<class T>
static double wspsnr_slow(const vector<T> refPic, const vector<T> recPic, int W, int H, int bitDepth, const vector<double> weights) {
    // Convert to double precision
    vector<double> refPicDouble(refPic.begin(), refPic.end());
    vector<double> recPicDouble(recPic.begin(), recPic.end());

    // Calculate MAX_VALUE based on bitDepth
    double MAX_VALUE = (255.0 * (1 << (bitDepth - 8)));
    double wspsnr = 0.0;
    double w_sum = 0.0;
    double sse_sum = 0.0;

#if MODE==USE_OPENMP
    std::vector<double> sse_sum_row(H);
    std::vector<double> w_sum_row(H);
#pragma omp parallel for
#endif 
    for (int hi = 0; hi < H; ++hi) {

        // Extract row slices
        auto start = hi * W;
        auto end = start + W;
        std::vector<double> ref(refPicDouble.begin() + start, refPicDouble.begin() + end);
        std::vector<double> rec(recPicDouble.begin() + start, recPicDouble.begin() + end);
        std::vector<double> diff(recPicDouble.size());

        // Compute row-wise sum squared error
        std::transform(ref.begin(), ref.end(), rec.begin(), diff.begin(),
            [](double x, double y) { return std::pow(x - y, 2); });
        double ssd_row = std::accumulate(diff.begin(), diff.end(), 0.0) * weights[hi];

        //OR
        /*double ssd_row = std::inner_product(
            ref.begin(), ref.end(), rec.begin(), 0.0,
            std::plus<>(),
            [](double x, double y) { return std::pow(x - y,2); }) * weights[hi];*/

#if MODE==USE_OPENMP
        sse_sum_row[hi] = ssd_row;
        w_sum_row[hi] = (weights[hi] * W);
#elif MODE==USE_NORMAL_LOOP
        sse_sum += ssd_row;
        w_sum += (weights[hi] * W);
#endif

    }
#if MODE==USE_OPENMP
    double sse_sum = std::accumulate(sse_sum_row.begin(), sse_sum_row.end(), 0.0);
    double w_sum = std::accumulate(w_sum_row.begin(), w_sum_row.end(), 0.0);
#endif

    wspsnr = 10 * std::log10((MAX_VALUE * MAX_VALUE * w_sum) / sse_sum);
    return wspsnr;
}

template<class T>
static double wspsnr(const T* refPic, const T* recPic, int W, int H, int bitDepth, const vector<double> weights, vector<int>* roi=NULL ) {

    // Calculate MAX_VALUE based on bitDepth
    double MAX_VALUE = (255.0 * (1 << (bitDepth - 8)));
    double wspsnr = 0.0;
    double w_sum = 0.0;
    double sse_sum = 0.0;


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

#if MODE==USE_OPENMP
    std::vector<double> sse_sum_row(H);
#pragma omp parallel for
#endif
    for (int hi = lh; hi < uh; ++hi) {
        auto row_st_idx = hi * roiW;
        for (int wi = lw; wi < uw; ++wi) {

            int diff = refPic[row_st_idx + wi] - recPic[row_st_idx + wi];

#if MODE==USE_OPENMP
            sse_sum_row[hi] += (diff * diff);
#else
            sse_sum += (diff * diff) * weights[hi];
#endif
        }
    }

    for (int hi = lh; hi < uh; ++hi) {
#if MODE==USE_OPENMP
        sse_sum += sse_sum_row[hi] * weights[hi];
#endif
        w_sum += (weights[hi] * W);
    }


    wspsnr = 10 * std::log10((MAX_VALUE * MAX_VALUE * w_sum) / sse_sum);
    return wspsnr;
}

template<class T>
static double psnr(const T* refPic, const T* recPic, int W, int H, int bitDepth, vector<int>* roi =NULL) {

    // Calculate MAX_VALUE based on bitDepth
    double MAX_VALUE = (255.0 * (1 << (bitDepth - 8)));
    double psnr = 0.0;
    double sse_sum = 0.0;


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


#if MODE==USE_OPENMP
    std::vector<double> sse_sum_row(roiH);
#pragma omp parallel for
#endif
    for (int hi = lh; hi < uh; ++hi) {
        auto row_st_idx = hi * roiW;
        int roi_hi = hi - lh;
        for (int wi = lw; wi < uw; ++wi) {

            int diff = refPic[row_st_idx + wi] - recPic[row_st_idx + wi];

#if MODE==USE_OPENMP
            sse_sum_row[roi_hi] += (diff * diff);
#else
            sse_sum += (diff * diff);
#endif
        }
    }

#if MODE==USE_OPENMP
    for (int hi = lh; hi < uh; ++hi) {
        sse_sum += sse_sum_row[hi - lh];
    }
#endif
    sse_sum = sse_sum / (roiH * roiW);
    psnr = 10 * std::log10((MAX_VALUE * MAX_VALUE) / sse_sum);
    return psnr;
}



#endif //NORMAL_LOOP_H