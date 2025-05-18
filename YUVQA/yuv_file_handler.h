#ifndef YUV_FILE_HANDLER_H
#define YUV_FILE_HANDLER_H

#include "config.h"


#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <string>

#include <cstring>   // need for GCC

using namespace std;
//namespace fs = std::filesystem;

/*vector<string> find_all_files(string folder, string extension) {

    vector<string> filenames;

    try {
        for (const auto& entry : fs::directory_iterator(folder)) {
            if (entry.is_regular_file() && entry.path().extension() == extension) {
                //std::cout << entry.path().string() << std::endl;
                std::cout << entry.path().filename().string() << std::endl;
                filenames.push_back(entry.path().filename().string());
            }
        }
    }
    catch (const fs::filesystem_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return filenames;
}*/

// Function to open YUV420 file
static ifstream open_YUV420_file(string inDir, string filename, int W, int H, int bd, int startFrame = 0) {
    int bpp = 1; // bytes per pixel
    if (bd == 10) {
        bpp = 2;
    }

    int Ybyte = W * H * bpp;
    int UVbyte = Ybyte / 2;

    ifstream yuv_f(inDir + filename + ".yuv", ios::binary);

    for (int k = 0; k < startFrame; ++k) {
        yuv_f.seekg(Ybyte + UVbyte, ios::cur);
    }
    auto s = yuv_f.is_open();
    return yuv_f;
}



#if MODE==USE_OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;
// Function to read YUV420 frame


static void read_YUV420_frame(ifstream& yuv_f, int width, int height, int bit_depth, Mat& Y_img, Mat& U_img, Mat& V_img) {
    int bpp = 1;
    if (bit_depth == 10) {
        bpp = 2;
    }
    int Ybytes = width * height * bpp;
    int UVbytes = Ybytes / 4;
    int UV_width = width / 2;
    int UV_height = height / 2;

    vector<char> Ybuff(Ybytes);
    vector<char> Ubuff(UVbytes);
    vector<char> Vbuff(UVbytes);

    yuv_f.read(reinterpret_cast<char*>(Ybuff.data()), Ybytes);
    yuv_f.read(reinterpret_cast<char*>(Ubuff.data()), UVbytes);
    yuv_f.read(reinterpret_cast<char*>(Vbuff.data()), UVbytes);


    Y_img = Mat(height, width, bit_depth == 10 ? CV_16UC1 : CV_8UC1, Ybuff.data()).clone();
    U_img = Mat(UV_height, UV_width, bit_depth == 10 ? CV_16UC1 : CV_8UC1, Ubuff.data()).clone();
    V_img = Mat(UV_height, UV_width, bit_depth == 10 ? CV_16UC1 : CV_8UC1, Vbuff.data()).clone();

}

#else
template<class T>
static  std::tuple<vector<T>, vector<T>, vector<T>> read_YUV420_frame(ifstream& yuv_f, int width, int height, int bit_depth) {
    int bpp = 1;
    if (bit_depth == 10) {
        bpp = 2;
    }

    int Ypixels = width * height;
    int Ybytes = Ypixels * bpp;

    int UVpixels = Ypixels / 4;
    int UVbytes = Ybytes / 4;

    vector<T> Ybuff(Ypixels);
    vector<T> Ubuff(UVpixels);
    vector<T> Vbuff(UVpixels);

    yuv_f.read(reinterpret_cast<char*>(Ybuff.data()), Ybytes);
    yuv_f.read(reinterpret_cast<char*>(Ubuff.data()), UVbytes);
    yuv_f.read(reinterpret_cast<char*>(Vbuff.data()), UVbytes);

    return std::make_tuple(std::move(Ybuff), std::move(Ubuff), std::move(Vbuff));

}

template<class Tin, class Tout>
static  std::tuple<vector<Tout>, vector<Tout>, vector<Tout>> read_YUV420_frame(ifstream& yuv_f, int width, int height, int bit_depth) {
    int bpp = 1;
    if (bit_depth == 10) {
        bpp = 2;
    }

    int Ypixels = width * height;
    int Ybytes = Ypixels * bpp;

    int UVpixels = Ypixels / 4;
    int UVbytes = Ybytes / 4;


    vector<Tin> buff(Ypixels + 2 * UVpixels);
    yuv_f.read(reinterpret_cast<char*>(buff.data()), Ybytes + 2 * UVbytes);

    vector<Tout> Ybuff(buff.data(), buff.data() + Ypixels);
    vector<Tout> Ubuff(buff.data() + Ypixels, buff.data() + Ypixels + UVpixels);
    vector<Tout> Vbuff(buff.data() + Ypixels + UVpixels, buff.data() + Ypixels + 2 * UVpixels);

    //return std::make_tuple(Ybuff, Ubuff, Vbuff);
    return std::make_tuple(std::move(Ybuff), std::move(Ubuff), std::move(Vbuff));
}

template<class Tin, class Tout>
static  void read_YUV420_frame(ifstream& yuv_f, Tout* Ybuffd, Tout* Ubuffd, Tout* Vbuffd, int width, int height, int bit_depth) {
    int bpp = 1;
    if (bit_depth == 10) {
        bpp = 2;
    }

    int Ypixels = width * height;
    int Ybytes = Ypixels * bpp;

    int UVpixels = Ypixels / 4;
    int UVbytes = Ybytes / 4;



    vector<Tin> buff(Ypixels + 2 * UVpixels);
    yuv_f.read(reinterpret_cast<char*>(buff.data()), Ybytes + 2 * UVbytes);

    vector<Tout> Ybuff(buff.data(), buff.data() + Ypixels);
    vector<Tout> Ubuff(buff.data() + Ypixels, buff.data() + Ypixels + UVpixels);
    vector<Tout> Vbuff(buff.data() + Ypixels + UVpixels, buff.data() + Ypixels + 2 * UVpixels);

    //Tout* Ybuffd = (Tout*)_aligned_malloc (Ypixels * sizeof(Tout), 32 );
    //Tout* Ubuffd = (Tout*)_aligned_malloc( UVpixels * sizeof(Tout), 32);
    //Tout* Vbuffd = (Tout*)_aligned_malloc( UVpixels * sizeof(Tout), 32);

    memcpy(Ybuffd, Ybuff.data(), Ypixels * sizeof(Tout));
    memcpy(Ubuffd, Ubuff.data(), UVpixels * sizeof(Tout));
    std::memcpy(Vbuffd, Vbuff.data(), UVpixels * sizeof(Tout));


    //return std::make_tuple(Ybuffd, Ubuffd, Vbuffd);
}


#endif

#endif YUV_FILE_HANDLER_H

