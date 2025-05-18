#ifndef WSPSNR_H
#define WSPSNR_H
#include <vector>
#include <algorithm>    // std::transform
#include <numeric>      // std::accumulate
#include <string>
#include <cmath>
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
    double mse_sum = 0.0;


    for (int hi = 0; hi < H; ++hi) {
        //double weight = std::cos((hi - (H / 2.0 - 0.5)) * M_PI / H);

        // Extract row slices
        auto start = hi * W;
        auto end = start + W;
        std::vector<double> ref(refPicDouble.begin() + start, refPicDouble.begin() + end);
        std::vector<double> rec(recPicDouble.begin() + start, recPicDouble.begin() + end);

        std::vector<double> diff(recPicDouble.size());

        // Compute row-wise mean squared error
        std::transform(ref.begin(), ref.end(), rec.begin(), diff.begin(),
            [](double x, double y) { return std::pow(x - y, 2); });
        double mse_row = std::accumulate(diff.begin(), diff.end(), 0.0) * weights[hi];

        /*double mse_row = std::inner_product(
            ref.begin(), ref.end(), rec.begin(), 0.0,
            std::plus<>(),
            [](double x, double y) { return std::pow(x - y,2); }) * weights[hi];*/

        mse_sum += mse_row;
        w_sum += (weights[hi] * W);
    }
    //43.598565988796224
    wspsnr = 10 * std::log10((MAX_VALUE * MAX_VALUE * w_sum) / mse_sum);
    return wspsnr;
}

template<class T>
static double wspsnr(const T* refPic, const T* recPic, int W, int H, int bitDepth, const vector<double> weights) {

    // Calculate MAX_VALUE based on bitDepth
    double MAX_VALUE = (255.0 * (1 << (bitDepth - 8)));
    double wspsnr = 0.0;
    double w_sum = 0.0;
    double sse_sum = 0.0;

    for (int hi = 0; hi < H; ++hi) {
        auto row_st_idx = hi * W;
        w_sum += (weights[hi] * W);
        for (int wi = 0; wi < W; ++wi) {

            int diff = refPic[row_st_idx + wi] - recPic[row_st_idx + wi];
            sse_sum += (diff * diff) * weights[hi];
        }
    }
    //43.598565988796224
    wspsnr = 10 * std::log10((MAX_VALUE * MAX_VALUE * w_sum) / sse_sum);
    return wspsnr;
}

template<class T>
static double wspsnr(const vector<T> refPic, const vector<T> recPic, int W, int H, int bitDepth, const vector<double> weights) {

    // Calculate MAX_VALUE based on bitDepth
    double MAX_VALUE = (255.0 * (1 << (bitDepth - 8)));
    double wspsnr = 0.0;
    double w_sum = 0.0;
    double sse_sum = 0.0;

    for (int hi = 0; hi < H; ++hi) {
        auto row_st_idx = hi * W;
        w_sum += (weights[hi] * W);
        for (int wi = 0; wi < W; ++wi) {

            int diff = refPic[row_st_idx + wi] - recPic[row_st_idx + wi];
            sse_sum += (diff * diff) * weights[hi];
        }
    }
    //43.598565988796224
    wspsnr = 10 * std::log10((MAX_VALUE * MAX_VALUE * w_sum) / sse_sum);
    return wspsnr;
}

template<class T>
static double wspsnr_openmp(const vector<T> refPic, const vector<T> recPic, int W, int H, int bitDepth, const vector<double> weights) {

    // Calculate MAX_VALUE based on bitDepth
    double MAX_VALUE = (255.0 * (1 << (bitDepth - 8)));
    double wspsnr = 0.0;
    double w_sum = 0.0;
    std::vector<double> sse_sum_row(H);
    double sse_sum = 0.0;

#pragma omp parallel for
    for (int hi = 0; hi < H; ++hi) {
        auto row_st_idx = hi * W;

        for (int wi = 0; wi < W; ++wi) {

            int diff = refPic[row_st_idx + wi] - recPic[row_st_idx + wi];
            sse_sum_row[hi] += (diff * diff);
        }
    }


    for (int hi = 0; hi < H; ++hi) {
        sse_sum += sse_sum_row[hi] * weights[hi];
        w_sum += (weights[hi] * W);
    }

    //43.598565988796224
    wspsnr = 10 * std::log10((MAX_VALUE * MAX_VALUE * w_sum) / sse_sum);
    return wspsnr;
}

template<class T>
static double wspsnr_openmp(const T* refPic, const T* recPic, int W, int H, int bitDepth, const vector<double> weights) {

    // Calculate MAX_VALUE based on bitDepth
    double MAX_VALUE = (255.0 * (1 << (bitDepth - 8)));
    double wspsnr = 0.0;
    double w_sum = 0.0;
    std::vector<double> sse_sum_row(H);
    double sse_sum = 0.0;

#pragma omp parallel for
    for (int hi = 0; hi < H; ++hi) {
        auto row_st_idx = hi * W;

        for (int wi = 0; wi < W; ++wi) {

            int diff = refPic[row_st_idx + wi] - recPic[row_st_idx + wi];
            sse_sum_row[hi] += (diff * diff);
        }
    }


    for (int hi = 0; hi < H; ++hi) {
        sse_sum += sse_sum_row[hi] * weights[hi];
        w_sum += (weights[hi] * W);
    }

    //43.598565988796224
    wspsnr = 10 * std::log10((MAX_VALUE * MAX_VALUE * w_sum) / sse_sum);
    return wspsnr;
}

template<class T>
static double wspsnr_openmp_slow(const vector<T> refPic, const vector<T> recPic, int W, int H, int bitDepth, const vector<double> weights) {
    // Convert to double precision
    vector<double> refPicDouble(refPic.begin(), refPic.end());
    vector<double> recPicDouble(recPic.begin(), recPic.end());

    // Calculate MAX_VALUE based on bitDepth
    double MAX_VALUE = (255.0 * (1 << (bitDepth - 8)));
    double wspsnr = 0.0;


    std::vector<double> mse_sum_row(H);
    std::vector<double> w_sum_row(H);

    //int num_threads = 4;  
    //omp_set_num_threads(num_threads);
    //std::cout << "num_threads:" << omp_get_num_threads() << std::endl;

#pragma omp parallel for
    for (int hi = 0; hi < H; ++hi) {

        // Extract row slices
        int start = hi * W;
        int end = start + W;
        std::vector<double> ref(refPicDouble.begin() + start, refPicDouble.begin() + end);
        std::vector<double> rec(recPicDouble.begin() + start, recPicDouble.begin() + end);

        double mse_row = std::inner_product(
            ref.begin(), ref.end(), rec.begin(), 0.0,
            std::plus<>(),
            [](double x, double y) { return std::pow(x - y, 2); }) * weights[hi];

        mse_sum_row[hi] = mse_row;
        w_sum_row[hi] = (weights[hi] * W);
    }

    //43.598565988796224
    double mse_sum = std::accumulate(mse_sum_row.begin(), mse_sum_row.end(), 0.0);
    double w_sum = std::accumulate(w_sum_row.begin(), w_sum_row.end(), 0.0);

    wspsnr = 10 * std::log10((MAX_VALUE * MAX_VALUE * w_sum) / mse_sum);
    return wspsnr;
}


#endif WSPSNR_H