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

    //Convert Tin to Tout
    vector<Tout> Ybuff(buff.data(), buff.data() + Ypixels);
    vector<Tout> Ubuff(buff.data() + Ypixels, buff.data() + Ypixels + UVpixels);
    vector<Tout> Vbuff(buff.data() + Ypixels + UVpixels, buff.data() + Ypixels + 2 * UVpixels);

    //return std::make_tuple(Ybuff, Ubuff, Vbuff);
    return std::make_tuple(std::move(Ybuff), std::move(Ubuff), std::move(Vbuff));
}

template<class Tin, class Tout>
static  void read_YUV420_frame(ifstream& yuv_f, Tout* Ybuffout, Tout* Ubuffout, Tout* Vbuffout, int width, int height, int bit_depth) {
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

	//Convert Tin to Tout
    vector<Tout> Ybuff(buff.data(), buff.data() + Ypixels);
    vector<Tout> Ubuff(buff.data() + Ypixels, buff.data() + Ypixels + UVpixels);
    vector<Tout> Vbuff(buff.data() + Ypixels + UVpixels, buff.data() + Ypixels + 2 * UVpixels);

    memcpy(Ybuffout, Ybuff.data(), Ypixels * sizeof(Tout));
    memcpy(Ubuffout, Ubuff.data(), UVpixels * sizeof(Tout));
    memcpy(Vbuffout, Vbuff.data(), UVpixels * sizeof(Tout));
}


#endif YUV_FILE_HANDLER_H

