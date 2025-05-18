#pragma once

#include "config.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <chrono>
#include <tuple>
#include <algorithm>    // std::transform
#include <numeric>      // std::accumulate
#include <omp.h>
#include <cstdlib>      // for _aligned_malloc and _aligned_free
#include <immintrin.h>  // for SIMD operations
#include <filesystem>
#include <string>

#include <cstring>   // need for GCC

using namespace std;


#define M_PI 3.14159265358979323846


static const vector<double> get_wpsnr_weights(int H)
{
    vector<double> weights(H);


    for (int hi = 0; hi < H; ++hi)
        weights[hi] = std::cos((hi - (H / 2.0 - 0.5)) * M_PI / H);

    return weights;
}

static  double get_sum_weights(vector<double> weights, int W,int H)
{
    double sum = 0;


    for (int hi = 0; hi < H; ++hi)
        sum+= weights[hi]*W;

    return sum;
}


