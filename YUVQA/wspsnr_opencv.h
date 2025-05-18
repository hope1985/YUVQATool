

#ifndef WSPSNR_OPENCV
#define WSPSNR_OPENCV

#include "config.h"

#if MODE==USE_OPENCV

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

static double wpsnr_opencv(const cv::Mat& refPic, const cv::Mat& recPic, int W, int H, int bitDepth, const vector<double> weights,int computation_dtype= CV_32S) {
    
    // Convert to double precision
    cv::Mat refPicDouble, recPicDouble;
    refPic.convertTo(refPicDouble, computation_dtype);
    recPic.convertTo(recPicDouble, computation_dtype);

    // Calculate MAX_VALUE based on bitDepth
    double MAX_VALUE = (255.0 * (1 << (bitDepth - 8)));
    double wspsnr = 0.0;
    double w_sum = 0.0;
    double sse_sum = 0.0;

    for (int hi = 0; hi < H; ++hi) {
        //double weight = std::cos((hi - (H / 2.0 - 0.5)) * M_PI / H);

        // Extract row slices
        cv::Mat ref = refPicDouble.row(hi);
        cv::Mat rec = recPicDouble.row(hi);

        // Compute row-wise mean squared error
        cv::Mat diff;
        //cv::absdiff(ref, rec, diff);
        cv::subtract(ref, rec, diff);
        cv::Mat sqDiff;
        cv::multiply(diff, diff, sqDiff,1.0, computation_dtype);
        double sse_row = cv::sum(sqDiff)[0] * weights[hi];

        sse_sum += sse_row;
        w_sum += (weights[hi] * W);
    }

    wspsnr = 10 * std::log10((MAX_VALUE * MAX_VALUE * w_sum) / sse_sum);
    return wspsnr;
}

#endif
#endif