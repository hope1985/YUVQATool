
#include "cpu_info.h"
#include "config.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <stdio.h>
#include "commons.h"
#include "yuv_file_handler.h"
#include <filesystem>

namespace fs = std::filesystem;
using namespace std;

#if (MODE== USE_NORMAL_LOOP) || (MODE== USE_OPENMP)
#include "normal_loop.h"
#elif MODE== USE_SIMD
#include "simd.h"
#elif MODE== USE_CUDA
#include "cuda.cuh"
#endif

static string inDir = "";
static string ref_inDir = "";
static int startFrame = 0;
static int frames = 30;
static int threads = 0;
static int metric = PSNR_METRIC;
static string metric_name = "PSNR";
static std::vector<int>* roiY = NULL;
static std::vector<int>* roiUV = NULL;

void compute_wspsnr_file(string filename, string ref_filename, int W, int H, int startFrame, int frames, int bd) {

	double wpsnr_avg[3] = { 0 };
	double total_time = 0;

	auto yuv_f = open_YUV420_file(inDir, filename, W, H, bd);

	auto ref_yuv_f = open_YUV420_file(ref_inDir, ref_filename, W, H, bd);

	auto wspsnr_weightsY = get_wpsnr_weights(H);
	auto wspsnr_weightsUV = get_wpsnr_weights(H / 2);

	double duration = 0;

	int pixel_cnt_y = W * H;
	int pixel_cnt_uv = (W * H) / 4;
	int batch_size = 1;

	if (metric == WSPSNR_ERP_METRIC)
		std::cout << "filename,fn,WSPSNR-Y(dB),WSPSNR-U(dB),WSPSNR-V(dB),Time(Sec)" << std::endl;
#if MODE == USE_CUDA

	//batch_size = 1;
	auto w_sumY = get_sum_weights(wspsnr_weightsY, W, H);
	auto w_sumUV = get_sum_weights(wspsnr_weightsUV, W / 2, H / 2);

	int* dev_refY = 0;
	int* dev_recY = 0;
	double* dev_weightsY = 0;
	double* dev_partial_sumY = 0;
	intit_device_buffer(&dev_refY, &dev_recY, &dev_weightsY, &dev_partial_sumY, W, H, wspsnr_weightsY.data(), batch_size);
	int* dev_refU = 0;
	int* dev_recU = 0;
	double* dev_weightsU = 0;
	double* dev_partial_sumU = 0;
	intit_device_buffer(&dev_refU, &dev_recU, &dev_weightsU, &dev_partial_sumU, W / 2, H / 2, wspsnr_weightsUV.data(), batch_size);

	int* dev_refV = 0;
	int* dev_recV = 0;
	double* dev_weightsV = 0;
	double* dev_partial_sumV = 0;
	intit_device_buffer(&dev_refV, &dev_recV, &dev_weightsV, &dev_partial_sumV, W / 2, H / 2, wspsnr_weightsUV.data(), batch_size);

	cudaError_t cudaStatus;

#elif MODE==USE_SIMD ||  MODE==USE_OPENMP  || MODE==USE_NORMAL_LOOP 

#ifdef _WIN32
	static COMPUTE_DTYPE* ref_Y_img = (COMPUTE_DTYPE*)_aligned_malloc(W * H * sizeof(COMPUTE_DTYPE), 32);
	static COMPUTE_DTYPE* ref_U_img = (COMPUTE_DTYPE*)_aligned_malloc(W / 2 * H / 2 * sizeof(COMPUTE_DTYPE), 32);
	static COMPUTE_DTYPE* ref_V_img = (COMPUTE_DTYPE*)_aligned_malloc(W / 2 * H / 2 * sizeof(COMPUTE_DTYPE), 32);
	static COMPUTE_DTYPE* Y_img = (COMPUTE_DTYPE*)_aligned_malloc(W * H * sizeof(COMPUTE_DTYPE), 32);
	static COMPUTE_DTYPE* U_img = (COMPUTE_DTYPE*)_aligned_malloc(W / 2 * H / 2 * sizeof(COMPUTE_DTYPE), 32);
	static COMPUTE_DTYPE* V_img = (COMPUTE_DTYPE*)_aligned_malloc(W / 2 * H / 2 * sizeof(COMPUTE_DTYPE), 32);

#else
	static COMPUTE_DTYPE* ref_Y_img = (COMPUTE_DTYPE*)aligned_alloc(32, W * H * sizeof(COMPUTE_DTYPE));
	static  COMPUTE_DTYPE* ref_U_img = (COMPUTE_DTYPE*)aligned_alloc(32, W / 2 * H / 2 * sizeof(COMPUTE_DTYPE));
	static COMPUTE_DTYPE* ref_V_img = (COMPUTE_DTYPE*)aligned_alloc(32, W / 2 * H / 2 * sizeof(COMPUTE_DTYPE));

	static COMPUTE_DTYPE* Y_img = (COMPUTE_DTYPE*)aligned_alloc(32, W * H * sizeof(COMPUTE_DTYPE));
	static  COMPUTE_DTYPE* U_img = (COMPUTE_DTYPE*)aligned_alloc(32, W / 2 * H / 2 * sizeof(COMPUTE_DTYPE));
	static COMPUTE_DTYPE* V_img = (COMPUTE_DTYPE*)aligned_alloc(32, W / 2 * H / 2 * sizeof(COMPUTE_DTYPE));

#endif       
#endif

	for (int k = startFrame; k < frames; k = k + batch_size)
	{




#if (MODE== USE_NORMAL_LOOP) || (MODE== USE_OPENMP)

		double wspsnr_frame[3] = { 0 };
		double duration = 0;
		if (bd == 8)
		{
			read_YUV420_frame<unsigned char, COMPUTE_DTYPE>(ref_yuv_f, ref_Y_img, ref_U_img, ref_V_img, W, H, bd);
			read_YUV420_frame<unsigned char, COMPUTE_DTYPE>(yuv_f, Y_img, U_img, V_img, W, H, bd);

			auto st = std::chrono::high_resolution_clock::now();
			// Calculate WPSNR
			wspsnr_frame[0] = wspsnr(ref_Y_img, Y_img, W, H, bd, wspsnr_weightsY, roiY);
			wspsnr_frame[1] = wspsnr(ref_U_img, U_img, W / 2, H / 2, bd, wspsnr_weightsUV, roiUV);
			wspsnr_frame[2] = wspsnr(ref_V_img, V_img, W / 2, H / 2, bd, wspsnr_weightsUV, roiUV);
			auto et = std::chrono::high_resolution_clock::now();
			duration = chrono::duration_cast<chrono::milliseconds>(et - st).count() / 1000.0;
		}
		else
		{
			read_YUV420_frame<unsigned short, COMPUTE_DTYPE>(ref_yuv_f, ref_Y_img, ref_U_img, ref_V_img, W, H, bd);
			read_YUV420_frame<unsigned short, COMPUTE_DTYPE>(yuv_f, Y_img, U_img, V_img, W, H, bd);

			auto st = std::chrono::high_resolution_clock::now();
			// Calculate WPSNR
			wspsnr_frame[0] = wspsnr(ref_Y_img, Y_img, W, H, bd, wspsnr_weightsY, roiY);
			wspsnr_frame[1] = wspsnr(ref_U_img, U_img, W / 2, H / 2, bd, wspsnr_weightsUV, roiUV);
			wspsnr_frame[2] = wspsnr(ref_V_img, V_img, W / 2, H / 2, bd, wspsnr_weightsUV, roiUV);
			auto et = std::chrono::high_resolution_clock::now();
			duration = chrono::duration_cast<chrono::milliseconds>(et - st).count() / 1000.0;

		}

#elif MODE== USE_SIMD

		double wspsnr_frame[3] = { 0 };
		double duration = 0;
		if (bd == 8)
		{

			//vector<unsigned char> ref_Y_img, ref_U_img, ref_V_img;
			//std::tie(ref_Y_img, ref_U_img, ref_V_img) = read_YUV420_frame<unsigned char, unsigned char>(ref_yuv_f, W, H, bd);

			//vector<unsigned char> Y_img, U_img, V_img;
			//std::tie(Y_img, U_img, V_img) = read_YUV420_frame<unsigned char, unsigned char>(yuv_f, W, H, bd);

			read_YUV420_frame<unsigned char, COMPUTE_DTYPE>(ref_yuv_f, ref_Y_img, ref_U_img, ref_V_img, W, H, bd);
			read_YUV420_frame<unsigned char, COMPUTE_DTYPE>(yuv_f, Y_img, U_img, V_img, W, H, bd);

			auto st = std::chrono::high_resolution_clock::now();
			// Calculate WPSNR
			wspsnr_frame[0] = wspsnr_openmp_simd(ref_Y_img, Y_img, W, H, bd, wspsnr_weightsY, roiY);
			wspsnr_frame[1] = wspsnr_openmp_simd(ref_U_img, U_img, W / 2, H / 2, bd, wspsnr_weightsUV, roiUV);
			wspsnr_frame[2] = wspsnr_openmp_simd(ref_V_img, V_img, W / 2, H / 2, bd, wspsnr_weightsUV, roiUV);
			auto et = std::chrono::high_resolution_clock::now();
			duration = chrono::duration_cast<chrono::milliseconds>(et - st).count() / 1000.0;


		}
		else
		{

			//vector<unsigned short> ref_Y_img, ref_U_img, ref_V_img;
			//std::tie(ref_Y_img, ref_U_img, ref_V_img) = read_YUV420_frame<unsigned short, unsigned short>(ref_yuv_f, W, H, bd);

			//vector<unsigned short> Y_img, U_img, V_img;
			//std::tie(Y_img, U_img, V_img) = read_YUV420_frame<unsigned short, unsigned short>(yuv_f, W, H, bd);

			read_YUV420_frame<unsigned short, COMPUTE_DTYPE>(ref_yuv_f, ref_Y_img, ref_U_img, ref_V_img, W, H, bd);
			read_YUV420_frame<unsigned short, COMPUTE_DTYPE>(yuv_f, Y_img, U_img, V_img, W, H, bd);


			auto st = std::chrono::high_resolution_clock::now();
			// Calculate WPSNR
			wspsnr_frame[0] = wspsnr_openmp_simd(ref_Y_img, Y_img, W, H, bd, wspsnr_weightsY, roiY);
			wspsnr_frame[1] = wspsnr_openmp_simd(ref_U_img, U_img, W / 2, H / 2, bd, wspsnr_weightsUV, roiUV);
			wspsnr_frame[2] = wspsnr_openmp_simd(ref_V_img, V_img, W / 2, H / 2, bd, wspsnr_weightsUV, roiUV);



			auto et = std::chrono::high_resolution_clock::now();
			duration = chrono::duration_cast<chrono::milliseconds>(et - st).count() / 1000.0;

		}


#elif MODE== USE_CUDA


		static vector<int> refYbatch(pixel_cnt_y * batch_size);
		static vector<int> recYbatch(pixel_cnt_y * batch_size);
		static vector<int> refUbatch(pixel_cnt_uv * batch_size);
		static vector<int> recUbatch(pixel_cnt_uv * batch_size);
		static vector<int> refVbatch(pixel_cnt_uv * batch_size);
		static vector<int> recVbatch(pixel_cnt_uv * batch_size);


		double* wspsnr_framesY = new double[batch_size];
		double* wspsnr_framesU = new double[batch_size];
		double* wspsnr_framesV = new double[batch_size];


		//Read frames as many as batch_size
		if (bd == 8)
		{



			for (int z = 0; z < batch_size; z++)
			{
				vector<int> ref_Y_img, ref_U_img, ref_V_img;
				std::tie(ref_Y_img, ref_U_img, ref_V_img) = read_YUV420_frame<unsigned char, int>(ref_yuv_f, W, H, bd);

				vector<int> Y_img, U_img, V_img;
				std::tie(Y_img, U_img, V_img) = read_YUV420_frame<unsigned char, int>(yuv_f, W, H, bd);

				std::copy(ref_Y_img.begin(), ref_Y_img.end(), refYbatch.begin() + z * pixel_cnt_y);
				std::copy(Y_img.begin(), Y_img.end(), recYbatch.begin() + z * pixel_cnt_y);

				std::copy(ref_U_img.begin(), ref_U_img.end(), refUbatch.begin() + z * pixel_cnt_uv);
				std::copy(U_img.begin(), U_img.end(), recUbatch.begin() + z * pixel_cnt_uv);

				std::copy(ref_V_img.begin(), ref_V_img.end(), refVbatch.begin() + z * pixel_cnt_uv);
				std::copy(V_img.begin(), V_img.end(), recVbatch.begin() + z * pixel_cnt_uv);
			}

			auto st = std::chrono::high_resolution_clock::now();


			//Convert unsigned char/unsigned short to double
			/*vector<double> refPicY(refYbatch.begin(), refYbatch.end());
			vector<double> recPicY(recYbatch.begin(), recYbatch.end());

			vector<double> refPicU(refUbatch.begin(), refUbatch.end());
			vector<double> recPicU(recUbatch.begin(), recUbatch.end());

			vector<double> refPicV(refVbatch.begin(), refVbatch.end());
			vector<double> recPicV(recVbatch.begin(), recVbatch.end());*/

			vector<int> refPicY = refYbatch;
			vector<int> recPicY = recYbatch;
			vector<int> refPicU = refUbatch;
			vector<int> recPicU = recUbatch;
			vector<int> refPicV = refVbatch;
			vector<int> recPicV = recVbatch;


			//Works onyly with batch_size=2 and even number of frames
			//wspsnr_process_cuda_shared_mem_2streams(refPicY.data(), recPicY.data(), wspsnr_weightsY.data(), wspsnr_framesY, w_sumY, W, H, bd, batch_size);
			//wspsnr_process_cuda_shared_mem_2streams(refPicU.data(), recPicU.data(), wspsnr_weightsUV.data(), wspsnr_framesU, w_sumUV, W / 2, H / 2, bd, batch_size);
			//wspsnr_process_cuda_shared_mem_2streams(refPicV.data(), recPicV.data(), wspsnr_weightsUV.data(), wspsnr_framesV, w_sumUV, W / 2, H / 2, bd, batch_size);


			// Add vectors in parallel.
			/*cudaError_t cudaStatus = run_process_cuda(refPicY.data(), recPicY.data(), wspsnr_weightsY.data(), wspsnr_framesY, w_sumY, W, H, bd, batch_size);

			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "run_process_cuda for Y channel failed!");
				return 1;
			}*/


			wspsnr_process_cuda_shared_mem(refPicY.data(), recPicY.data(), dev_refY, dev_recY, dev_weightsY, dev_partial_sumY, wspsnr_weightsY.data(), wspsnr_framesY, w_sumY, W, H, bd, batch_size);
			wspsnr_process_cuda_shared_mem(refPicU.data(), recPicU.data(), dev_refU, dev_recU, dev_weightsU, dev_partial_sumU, wspsnr_weightsUV.data(), wspsnr_framesU, w_sumUV, W / 2, H / 2, bd, batch_size);
			wspsnr_process_cuda_shared_mem(refPicV.data(), recPicV.data(), dev_refV, dev_recV, dev_weightsV, dev_partial_sumV, wspsnr_weightsUV.data(), wspsnr_framesV, w_sumUV, W / 2, H / 2, bd, batch_size);


			/*
			wspsnr_process_cuda(refPicY.data(), recPicY.data(), wspsnr_weightsY.data(), wspsnr_framesY, w_sumY, W, H, bd, batch_size);
			wspsnr_process_cuda(refPicU.data(), recPicU.data(), wspsnr_weightsUV.data(), wspsnr_framesU, w_sumUV, W / 2, H / 2, bd, batch_size);
			wspsnr_process_cuda(refPicV.data(), recPicV.data(), wspsnr_weightsUV.data(), wspsnr_framesV, w_sumUV, W / 2, H / 2, bd, batch_size);
			*/

			auto et = std::chrono::high_resolution_clock::now();
			duration = chrono::duration_cast<chrono::milliseconds>(et - st).count() / 1000.0;
		}
		else
		{

			for (int z = 0; z < batch_size; z++)
			{
				vector<int> ref_Y_img, ref_U_img, ref_V_img;
				std::tie(ref_Y_img, ref_U_img, ref_V_img) = read_YUV420_frame<unsigned short, int>(ref_yuv_f, W, H, bd);

				vector<int> Y_img, U_img, V_img;
				std::tie(Y_img, U_img, V_img) = read_YUV420_frame<unsigned short, int>(yuv_f, W, H, bd);

				std::copy(ref_Y_img.begin(), ref_Y_img.end(), refYbatch.begin() + z * pixel_cnt_y);
				std::copy(Y_img.begin(), Y_img.end(), recYbatch.begin() + z * pixel_cnt_y);

				std::copy(ref_U_img.begin(), ref_U_img.end(), refUbatch.begin() + z * pixel_cnt_uv);
				std::copy(U_img.begin(), U_img.end(), recUbatch.begin() + z * pixel_cnt_uv);

				std::copy(ref_V_img.begin(), ref_V_img.end(), refVbatch.begin() + z * pixel_cnt_uv);
				std::copy(V_img.begin(), V_img.end(), recVbatch.begin() + z * pixel_cnt_uv);
			}

			auto st = std::chrono::high_resolution_clock::now();

			//Convert unsigned char/unsigned short to double
			/*vector<double> refPicY(refYbatch.begin(), refYbatch.end());
			vector<double> recPicY(recYbatch.begin(), recYbatch.end());

			vector<double> refPicU(refUbatch.begin(), refUbatch.end());
			vector<double> recPicU(recUbatch.begin(), recUbatch.end());

			vector<double> refPicV(refVbatch.begin(), refVbatch.end());
			vector<double> recPicV(recVbatch.begin(), recVbatch.end());*/

			vector<int> refPicY = refYbatch;
			vector<int> recPicY = recYbatch;
			vector<int> refPicU = refUbatch;
			vector<int> recPicU = recUbatch;
			vector<int> refPicV = refVbatch;
			vector<int> recPicV = recVbatch;

			//Works onyly with batch_size=2 and even number of framees
			//wspsnr_process_cuda_shared_mem_2streams(refPicY.data(), recPicY.data(), wspsnr_weightsY.data(), wspsnr_framesY, w_sumY, W, H, bd, batch_size);
			//wspsnr_process_cuda_shared_mem_2streams(refPicU.data(), recPicU.data(), wspsnr_weightsUV.data(), wspsnr_framesU, w_sumUV, W / 2, H / 2, bd, batch_size);
			//wspsnr_process_cuda_shared_mem_2streams(refPicV.data(), recPicV.data(), wspsnr_weightsUV.data(), wspsnr_framesV, w_sumUV, W / 2, H / 2, bd, batch_size);

			wspsnr_process_cuda_shared_mem(refPicY.data(), recPicY.data(), dev_refY, dev_recY, dev_weightsY, dev_partial_sumY, wspsnr_weightsY.data(), wspsnr_framesY, w_sumY, W, H, bd, batch_size);
			wspsnr_process_cuda_shared_mem(refPicU.data(), recPicU.data(), dev_refU, dev_recU, dev_weightsU, dev_partial_sumU, wspsnr_weightsUV.data(), wspsnr_framesU, w_sumUV, W / 2, H / 2, bd, batch_size);
			wspsnr_process_cuda_shared_mem(refPicV.data(), recPicV.data(), dev_refV, dev_recV, dev_weightsV, dev_partial_sumV, wspsnr_weightsUV.data(), wspsnr_framesV, w_sumUV, W / 2, H / 2, bd, batch_size);


			auto et = std::chrono::high_resolution_clock::now();
			duration = chrono::duration_cast<chrono::milliseconds>(et - st).count() / 1000.0;
		}

#endif

#if MODE!= USE_CUDA  

		wpsnr_avg[0] += wspsnr_frame[0];
		wpsnr_avg[1] += wspsnr_frame[1];
		wpsnr_avg[2] += wspsnr_frame[2];
		total_time += duration;

		std::cout << filename << ",  " << std::setw(3) << std::left << k << ",  "
			<< std::setw(7) << std::left << std::fixed << std::setprecision(3) << wspsnr_frame[0] << ",  " << wspsnr_frame[1] << ",  " << wspsnr_frame[2] << ",  "
			<< std::setw(8) << std::left << std::fixed << std::setprecision(4) << duration << std::endl;

#else

		total_time += duration;
		for (int i = 0; i < batch_size; i++)
		{
			wpsnr_avg[0] += wspsnr_framesY[i];
			wpsnr_avg[1] += wspsnr_framesU[i];
			wpsnr_avg[2] += wspsnr_framesV[i];
			std::cout << filename << ",  " << std::setw(3) << std::left << k << ",  "
				<< std::setw(7) << std::left << std::fixed << std::setprecision(3) << wspsnr_framesY[i] << ",  " << wspsnr_framesU[i] << ",  " << wspsnr_framesV[i] << ",  "
				<< std::setw(8) << std::left << std::fixed << std::setprecision(4) << duration / batch_size << std::endl;
		}

#endif
	}

#if MODE == USE_SIMD || MODE ==USE_OPENMP || MODE==  USE_NORMAL_LOOP 
#ifdef _WIN32
	_aligned_free(ref_Y_img);
	_aligned_free(ref_U_img);
	_aligned_free(ref_V_img);
	_aligned_free(Y_img);
	_aligned_free(U_img);
	_aligned_free(V_img);
	/*free(ref_Y_img);
	free(ref_U_img);
	free(ref_V_img);
	free(Y_img);
	free(U_img);
	free(V_img);*/

#else
	free(ref_Y_img);
	free(ref_U_img);
	free(ref_V_img);
	free(Y_img);
	free(U_img);
	free(V_img);
#endif

#elif MODE == USE_CUDA 

	cudaFree(dev_refY);
	cudaFree(dev_recY);
	cudaFree(dev_weightsY);
	cudaFree(dev_partial_sumY);
	cudaFree(dev_refU);
	cudaFree(dev_recU);
	cudaFree(dev_weightsU);
	cudaFree(dev_partial_sumU);
	cudaFree(dev_refV);
	cudaFree(dev_recV);
	cudaFree(dev_weightsV);
	cudaFree(dev_partial_sumV);
#endif


	std::cout << "filename,AVG_WSPSNR-Y(dB),AVG_WSPSNR-U(dB),AVG_WSPSNR-V(dB),AVG_FRAME_TIME(Sec),TOTAL_TIME(Sec)" << std::endl;
	std::cout << filename << ",  "
		<< std::setw(7) << std::left << std::fixed << std::setprecision(3) << wpsnr_avg[0] / frames << ",  " << wpsnr_avg[1] / frames << ",  " << wpsnr_avg[2] / frames << ",  "
		<< std::setw(7) << std::left << std::fixed << std::setprecision(4) << total_time / frames << ",  "
		<< std::setw(8) << std::left << std::fixed << std::setprecision(4) << total_time << std::endl;

	yuv_f.close();
	ref_yuv_f.close();

}

void compute_psnr_file(string filename, string ref_filename, int W, int H, int startFrame, int frames, int bd) {

	double psnr_avg[3] = { 0 };
	double total_time = 0;

	auto yuv_f = open_YUV420_file(inDir, filename, W, H, bd);

	auto ref_yuv_f = open_YUV420_file(ref_inDir, ref_filename, W, H, bd);

	double duration = 0;

	int pixel_cnt_y = W * H;
	int pixel_cnt_uv = (W * H) / 4;
	int batch_size = 1;

	std::cout << "filename,fn,PSNR-Y(dB),PSNR-U(dB),PSNR-V(dB),Time(Sec)" << std::endl;
#if MODE == USE_CUDA

	int* dev_refY = 0;
	int* dev_recY = 0;
	double* dev_partial_sumY = 0;
	intit_device_buffer(&dev_refY, &dev_recY, NULL, &dev_partial_sumY, W, H, NULL, batch_size);
	int* dev_refU = 0;
	int* dev_recU = 0;
	double* dev_partial_sumU = 0;
	intit_device_buffer(&dev_refU, &dev_recU, NULL, &dev_partial_sumU, W / 2, H / 2, NULL, batch_size);

	int* dev_refV = 0;
	int* dev_recV = 0;
	double* dev_partial_sumV = 0;
	intit_device_buffer(&dev_refV, &dev_recV, NULL, &dev_partial_sumV, W / 2, H / 2, NULL, batch_size);

	cudaError_t cudaStatus;

#elif MODE==USE_SIMD ||  MODE==USE_OPENMP  || MODE==USE_NORMAL_LOOP 

#ifdef _WIN32
	static COMPUTE_DTYPE* ref_Y_img = (COMPUTE_DTYPE*)_aligned_malloc(W * H * sizeof(COMPUTE_DTYPE), 32);
	static COMPUTE_DTYPE* ref_U_img = (COMPUTE_DTYPE*)_aligned_malloc(W / 2 * H / 2 * sizeof(COMPUTE_DTYPE), 32);
	static COMPUTE_DTYPE* ref_V_img = (COMPUTE_DTYPE*)_aligned_malloc(W / 2 * H / 2 * sizeof(COMPUTE_DTYPE), 32);
	static COMPUTE_DTYPE* Y_img = (COMPUTE_DTYPE*)_aligned_malloc(W * H * sizeof(COMPUTE_DTYPE), 32);
	static COMPUTE_DTYPE* U_img = (COMPUTE_DTYPE*)_aligned_malloc(W / 2 * H / 2 * sizeof(COMPUTE_DTYPE), 32);
	static COMPUTE_DTYPE* V_img = (COMPUTE_DTYPE*)_aligned_malloc(W / 2 * H / 2 * sizeof(COMPUTE_DTYPE), 32);

#else
	static COMPUTE_DTYPE* ref_Y_img = (COMPUTE_DTYPE*)aligned_alloc(32, W * H * sizeof(COMPUTE_DTYPE));
	static  COMPUTE_DTYPE* ref_U_img = (COMPUTE_DTYPE*)aligned_alloc(32, W / 2 * H / 2 * sizeof(COMPUTE_DTYPE));
	static COMPUTE_DTYPE* ref_V_img = (COMPUTE_DTYPE*)aligned_alloc(32, W / 2 * H / 2 * sizeof(COMPUTE_DTYPE));

	static COMPUTE_DTYPE* Y_img = (COMPUTE_DTYPE*)aligned_alloc(32, W * H * sizeof(COMPUTE_DTYPE));
	static  COMPUTE_DTYPE* U_img = (COMPUTE_DTYPE*)aligned_alloc(32, W / 2 * H / 2 * sizeof(COMPUTE_DTYPE));
	static COMPUTE_DTYPE* V_img = (COMPUTE_DTYPE*)aligned_alloc(32, W / 2 * H / 2 * sizeof(COMPUTE_DTYPE));

#endif       
#endif

	for (int k = startFrame; k < frames; k = k + batch_size)
	{


#if (MODE== USE_NORMAL_LOOP) || (MODE== USE_OPENMP)

		double psnr_frame[3] = { 0 };
		double duration = 0;
		if (bd == 8)
		{
			read_YUV420_frame<unsigned char, COMPUTE_DTYPE>(ref_yuv_f, ref_Y_img, ref_U_img, ref_V_img, W, H, bd);
			read_YUV420_frame<unsigned char, COMPUTE_DTYPE>(yuv_f, Y_img, U_img, V_img, W, H, bd);

			auto st = std::chrono::high_resolution_clock::now();
			// Calculate WPSNR
			psnr_frame[0] = psnr(ref_Y_img, Y_img, W, H, bd, roiY);
			psnr_frame[1] = psnr(ref_U_img, U_img, W / 2, H / 2, bd, roiUV);
			psnr_frame[2] = psnr(ref_V_img, V_img, W / 2, H / 2, bd, roiUV);
			auto et = std::chrono::high_resolution_clock::now();
			duration = chrono::duration_cast<chrono::milliseconds>(et - st).count() / 1000.0;
		}
		else
		{
			read_YUV420_frame<unsigned short, COMPUTE_DTYPE>(ref_yuv_f, ref_Y_img, ref_U_img, ref_V_img, W, H, bd);
			read_YUV420_frame<unsigned short, COMPUTE_DTYPE>(yuv_f, Y_img, U_img, V_img, W, H, bd);

			auto st = std::chrono::high_resolution_clock::now();
			// Calculate WPSNR
			psnr_frame[0] = psnr(ref_Y_img, Y_img, W, H, bd, roiY);
			psnr_frame[1] = psnr(ref_U_img, U_img, W / 2, H / 2, bd, roiUV);
			psnr_frame[2] = psnr(ref_V_img, V_img, W / 2, H / 2, bd, roiUV);
			auto et = std::chrono::high_resolution_clock::now();
			duration = chrono::duration_cast<chrono::milliseconds>(et - st).count() / 1000.0;

		}

#elif MODE== USE_SIMD

		double psnr_frame[3] = { 0 };
		double duration = 0;
		if (bd == 8)
		{
			read_YUV420_frame<unsigned char, COMPUTE_DTYPE>(ref_yuv_f, ref_Y_img, ref_U_img, ref_V_img, W, H, bd);
			read_YUV420_frame<unsigned char, COMPUTE_DTYPE>(yuv_f, Y_img, U_img, V_img, W, H, bd);

			auto st = std::chrono::high_resolution_clock::now();
			// Calculate WPSNR
			psnr_frame[0] = psnr_openmp_simd(ref_Y_img, Y_img, W, H, bd, roiY);
			psnr_frame[1] = psnr_openmp_simd(ref_U_img, U_img, W / 2, H / 2, bd, roiUV);
			psnr_frame[2] = psnr_openmp_simd(ref_V_img, V_img, W / 2, H / 2, bd, roiUV);
			auto et = std::chrono::high_resolution_clock::now();
			duration = chrono::duration_cast<chrono::milliseconds>(et - st).count() / 1000.0;
		}
		else
		{

			read_YUV420_frame<unsigned short, COMPUTE_DTYPE>(ref_yuv_f, ref_Y_img, ref_U_img, ref_V_img, W, H, bd);
			read_YUV420_frame<unsigned short, COMPUTE_DTYPE>(yuv_f, Y_img, U_img, V_img, W, H, bd);

			auto st = std::chrono::high_resolution_clock::now();
			// Calculate WPSNR
			psnr_frame[0] = psnr_openmp_simd(ref_Y_img, Y_img, W, H, bd, roiY);
			psnr_frame[1] = psnr_openmp_simd(ref_U_img, U_img, W / 2, H / 2, bd, roiUV);
			psnr_frame[2] = psnr_openmp_simd(ref_V_img, V_img, W / 2, H / 2, bd, roiUV);

			auto et = std::chrono::high_resolution_clock::now();
			duration = chrono::duration_cast<chrono::milliseconds>(et - st).count() / 1000.0;

		}


#elif MODE== USE_CUDA

		static vector<int> refYbatch(pixel_cnt_y * batch_size);
		static vector<int> recYbatch(pixel_cnt_y * batch_size);
		static vector<int> refUbatch(pixel_cnt_uv * batch_size);
		static vector<int> recUbatch(pixel_cnt_uv * batch_size);
		static vector<int> refVbatch(pixel_cnt_uv * batch_size);
		static vector<int> recVbatch(pixel_cnt_uv * batch_size);

		double* psnr_framesY = new double[batch_size];
		double* psnr_framesU = new double[batch_size];
		double* psnr_framesV = new double[batch_size];

		//Read frames as many as batch_size
		if (bd == 8)
		{
			for (int z = 0; z < batch_size; z++)
			{
				vector<int> ref_Y_img, ref_U_img, ref_V_img;
				std::tie(ref_Y_img, ref_U_img, ref_V_img) = read_YUV420_frame<unsigned char, int>(ref_yuv_f, W, H, bd);

				vector<int> Y_img, U_img, V_img;
				std::tie(Y_img, U_img, V_img) = read_YUV420_frame<unsigned char, int>(yuv_f, W, H, bd);

				std::copy(ref_Y_img.begin(), ref_Y_img.end(), refYbatch.begin() + z * pixel_cnt_y);
				std::copy(Y_img.begin(), Y_img.end(), recYbatch.begin() + z * pixel_cnt_y);

				std::copy(ref_U_img.begin(), ref_U_img.end(), refUbatch.begin() + z * pixel_cnt_uv);
				std::copy(U_img.begin(), U_img.end(), recUbatch.begin() + z * pixel_cnt_uv);

				std::copy(ref_V_img.begin(), ref_V_img.end(), refVbatch.begin() + z * pixel_cnt_uv);
				std::copy(V_img.begin(), V_img.end(), recVbatch.begin() + z * pixel_cnt_uv);
			}

			auto st = std::chrono::high_resolution_clock::now();

			vector<int> refPicY = refYbatch;
			vector<int> recPicY = recYbatch;
			vector<int> refPicU = refUbatch;
			vector<int> recPicU = recUbatch;
			vector<int> refPicV = refVbatch;
			vector<int> recPicV = recVbatch;

			psnr_process_cuda_shared_mem(refPicY.data(), recPicY.data(), dev_refY, dev_recY, dev_partial_sumY, psnr_framesY, W, H, bd, batch_size);
			psnr_process_cuda_shared_mem(refPicU.data(), recPicU.data(), dev_refU, dev_recU, dev_partial_sumU, psnr_framesU, W / 2, H / 2, bd, batch_size);
			psnr_process_cuda_shared_mem(refPicV.data(), recPicV.data(), dev_refV, dev_recV, dev_partial_sumV, psnr_framesV, W / 2, H / 2, bd, batch_size);

			auto et = std::chrono::high_resolution_clock::now();
			duration = chrono::duration_cast<chrono::milliseconds>(et - st).count() / 1000.0;
		}
		else
		{

			for (int z = 0; z < batch_size; z++)
			{
				vector<int> ref_Y_img, ref_U_img, ref_V_img;
				std::tie(ref_Y_img, ref_U_img, ref_V_img) = read_YUV420_frame<unsigned short, int>(ref_yuv_f, W, H, bd);

				vector<int> Y_img, U_img, V_img;
				std::tie(Y_img, U_img, V_img) = read_YUV420_frame<unsigned short, int>(yuv_f, W, H, bd);

				std::copy(ref_Y_img.begin(), ref_Y_img.end(), refYbatch.begin() + z * pixel_cnt_y);
				std::copy(Y_img.begin(), Y_img.end(), recYbatch.begin() + z * pixel_cnt_y);

				std::copy(ref_U_img.begin(), ref_U_img.end(), refUbatch.begin() + z * pixel_cnt_uv);
				std::copy(U_img.begin(), U_img.end(), recUbatch.begin() + z * pixel_cnt_uv);

				std::copy(ref_V_img.begin(), ref_V_img.end(), refVbatch.begin() + z * pixel_cnt_uv);
				std::copy(V_img.begin(), V_img.end(), recVbatch.begin() + z * pixel_cnt_uv);
			}

			auto st = std::chrono::high_resolution_clock::now();
			vector<int> refPicY = refYbatch;
			vector<int> recPicY = recYbatch;
			vector<int> refPicU = refUbatch;
			vector<int> recPicU = recUbatch;
			vector<int> refPicV = refVbatch;
			vector<int> recPicV = recVbatch;

			psnr_process_cuda_shared_mem(refPicY.data(), recPicY.data(), dev_refY, dev_recY, dev_partial_sumY, psnr_framesY, W, H, bd, batch_size);
			psnr_process_cuda_shared_mem(refPicU.data(), recPicU.data(), dev_refU, dev_recU, dev_partial_sumU, psnr_framesU, W / 2, H / 2, bd, batch_size);
			psnr_process_cuda_shared_mem(refPicV.data(), recPicV.data(), dev_refV, dev_recV, dev_partial_sumV, psnr_framesV, W / 2, H / 2, bd, batch_size);

			auto et = std::chrono::high_resolution_clock::now();
			duration = chrono::duration_cast<chrono::milliseconds>(et - st).count() / 1000.0;
		}

#endif

#if MODE!= USE_CUDA  

		psnr_avg[0] += psnr_frame[0];
		psnr_avg[1] += psnr_frame[1];
		psnr_avg[2] += psnr_frame[2];
		total_time += duration;

		std::cout << filename << ",  " << std::setw(3) << std::left << k << ",  "
			<< std::setw(7) << std::left << std::fixed << std::setprecision(3) << psnr_frame[0] << ",  " << psnr_frame[1] << ",  " << psnr_frame[2] << ",  "
			<< std::setw(8) << std::left << std::fixed << std::setprecision(4) << duration << std::endl;
#else

		total_time += duration;
		for (int i = 0; i < batch_size; i++)
		{
			psnr_avg[0] += psnr_framesY[i];
			psnr_avg[1] += psnr_framesU[i];
			psnr_avg[2] += psnr_framesV[i];
			std::cout << filename << ",  " << std::setw(3) << std::left << k << ",  "
				<< std::setw(7) << std::left << std::fixed << std::setprecision(3) << psnr_framesY[i] << ",  " << psnr_framesU[i] << ",  " << psnr_framesV[i] << ",  "
				<< std::setw(8) << std::left << std::fixed << std::setprecision(4) << duration / batch_size << std::endl;
		}

#endif
	}

#if MODE == USE_SIMD || MODE ==USE_OPENMP || MODE==  USE_NORMAL_LOOP 
#ifdef _WIN32
	_aligned_free(ref_Y_img);
	_aligned_free(ref_U_img);
	_aligned_free(ref_V_img);
	_aligned_free(Y_img);
	_aligned_free(U_img);
	_aligned_free(V_img);
	/*free(ref_Y_img);
	free(ref_U_img);
	free(ref_V_img);
	free(Y_img);
	free(U_img);
	free(V_img);*/

#else
	free(ref_Y_img);
	free(ref_U_img);
	free(ref_V_img);
	free(Y_img);
	free(U_img);
	free(V_img);
#endif

#elif MODE == USE_CUDA 

	cudaFree(dev_refY);
	cudaFree(dev_recY);
	cudaFree(dev_partial_sumY);
	cudaFree(dev_refU);
	cudaFree(dev_recU);
	cudaFree(dev_partial_sumU);
	cudaFree(dev_refV);
	cudaFree(dev_recV);
	cudaFree(dev_partial_sumV);
#endif


	std::cout << "filename,AVG_PSNR-Y(dB),AVG_PSNR-U(dB),AVG_PSNR-V(dB),AVG_FRAME_TIME(Sec),TOTAL_TIME(Sec)" << std::endl;
	std::cout << filename << ",  "
		<< std::setw(7) << std::left << std::fixed << std::setprecision(3) << psnr_avg[0] / frames << ",  " << psnr_avg[1] / frames << ",  " << psnr_avg[2] / frames << ",  "
		<< std::setw(7) << std::left << std::fixed << std::setprecision(4) << total_time / frames << ",  "
		<< std::setw(8) << std::left << std::fixed << std::setprecision(4) << total_time << std::endl;

	yuv_f.close();
	ref_yuv_f.close();

}

#if MODE == USE_SIMD || MODE == USE_OPENMP

void set_openmp_threads(int num_threads)
{
	//Set number of threads for openMP
	int phy_cores = get_physical_cores();
	//using threads more than the numner of physical cores is not efficient
	if (threads > phy_cores || threads < 1)
	{
		threads = get_physical_cores();
	}
}
#endif

int main(int argc, char* argv[])
{

	std::string filename;
	std::string ref_filename;

	int w;
	int h;
	int bd;

	if (argc > 1)
	{
		std::string arg = argv[1];
		if (arg == "-help")
		{

			cout << "Usage: " << "-argument" << " [value]" << endl;
			cout << "Argument     |" << "Description/Options" << endl;
			cout << "-i     | " << "Filepath of the input YUV file" << endl;
			cout << "-r     | " << "Filepath of the reference YUV file" << endl;
			//cout << "-id    | " << "Root directory of the input YUV file" << endl;
			//cout << "-rd    | " << "Root directory of the reference YUV file" << endl;
			cout << "-w     | " << "Width of the YUV file" << endl;
			cout << "-h     | " << "Height of the YUV file" << endl;
			cout << "-bd    | " << "Bit-depth of the YUV file (8 or 10)" << endl;
			cout << "-sf    | " << "Start frame index to begin computing the quality metric" << endl;
			cout << "-nf    | " << "Number of frames to compute quality metric" << endl;
			cout << "-qm    | " << "Quality metric type ('0' = PSNR, '1' = WSPSNR for ERP format)" << endl;
			cout << "-roi   | " << "Region of interest for quality computation as [Top, Bottom, Left, Right]" << endl;
#if (MODE == USE_OPENMP || MODE == USE_SIMD)
			cout << "-nt    | " << "Number of threads (used for USE_OPENMP/USE_SIMD modes); set '<= 0` or '>= core count' to use all physical cores)" << endl;
#endif

			return 0;
		}
	}

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];

		if (arg == "-i" && i + 1 < argc) {
			fs::path p(argv[++i]);
			inDir = p.parent_path().string();
			filename = p.stem().string();

			//std::cout << "Filename: " << inDir << "\n";
			//std::cout << "Input Directory: " << filename << "\n";
			//std::cout << "Extension: " << p.extension() << "\n";
		}
		else if (arg == "-r" && i + 1 < argc) {
			fs::path p(argv[++i]);
			ref_inDir = p.parent_path().string();
			ref_filename = p.stem().string();
		}
		//else if (arg == "-id" && i + 1 < argc) {
		//    inDir = argv[++i];
		//}
		//else if (arg == "-rd" && i + 1 < argc) {
		//    ref_inDir = argv[++i];
		//}
		else if (arg == "-w" && i + 1 < argc) {
			w = std::stoi(argv[++i]);
		}
		else if (arg == "-h" && i + 1 < argc) {
			h = std::stoi(argv[++i]);
		}
		else if (arg == "-bd" && i + 1 < argc) {
			bd = std::stoi(argv[++i]);
		}
		else if (arg == "-sf" && i + 1 < argc) {
			startFrame = std::stoi(argv[++i]);
		}
		else if (arg == "-nf" && i + 1 < argc) {
			frames = std::stoi(argv[++i]);
		}
#if (MODE == USE_OPENMP || MODE == USE_SIMD)
		else if (arg == "-nt" && i + 1 < argc) {
			threads = std::stoi(argv[++i]);
		}
#endif
		else if (arg == "-qm" && i + 1 < argc) {
			metric = std::stoi(argv[++i]);
			if (metric == PSNR_METRIC) {
				metric_name = "PSNR";
			}
			else if (metric == WSPSNR_ERP_METRIC) {
				metric_name = "WSPSNR-ERP";
			}
			else {
				std::cerr << "Unknown metric type. Use 0 for PSNR or 1 for WSPSNR-ERP." << std::endl;
				return 1;
			}
		}
		else if (arg == "-roi" && i + 1 < argc) {

			roiY = new vector<int>();
			roiUV = new vector<int>();

			string roi_str(argv[++i]);
			// Remove brackets
			if (roi_str.front() == '[') roi_str.erase(0, 1);
			if (roi_str.back() == ']') roi_str.pop_back();

			std::stringstream ss(roi_str);
			std::string token;


			// Split and convert to int
			while (std::getline(ss, token, ',')) {
				size_t start = token.find_first_not_of(" \t");
				size_t end = token.find_last_not_of(" \t");
				if (start != std::string::npos)
					token = token.substr(start, end - start + 1);
				int val = std::stoi(token);
				roiY->push_back(val);
				roiUV->push_back(val / 2);
			}

		}
	}


#ifdef _WIN32
	if (inDir.empty()) {
		inDir = "";
	}
	else {
		inDir += "\\";
	}
	if (ref_inDir.empty()) {
		ref_inDir = "";
	}
	else {
		ref_inDir += "\\";
	}
#else
	if (inDir.empty()) {
		inDir = "";
	}
	else {
		inDir += "/";
	}
	if (ref_inDir.empty()) {
		ref_inDir = "";
	}
	else {
		ref_inDir += "/";
	}
#endif
	std::cout << "=============================================" << std::endl;
#if MODE==USE_SIMD
	std::cout << "USE_SIMD" << std::endl;
#elif MODE==USE_OPENMP
	std::cout << "USE_OPENMP" << std::endl;
#elif MODE==USE_NORMAL_LOOP
	std::cout << "NORMAL_LOOP" << std::endl;
#elif MODE==USE_CUDA
	std::cout << "USE_CUDA" << std::endl;
#endif
	std::cout << "COMPUTE DTYPE SIZE=" << sizeof(COMPUTE_DTYPE) << std::endl;
	std::cout << "=============================================" << std::endl;
	std::cout << "INPUT FILENAME=" << filename << std::endl;
	std::cout << "REF FILENAME=" << ref_filename << std::endl;
	std::cout << "INPUT DIR=" << inDir << std::endl;
	std::cout << "BIT DEPTH=" << bd << std::endl;
	std::cout << "VIDEO SIZE=[W,H]=[" << w << "," << h << "]" << std::endl;
	std::cout << "NUM FRAMES=" << frames << std::endl;
	std::cout << "START FRAME=" << startFrame << std::endl;
	std::cout << "METRIC=" << metric_name << std::endl;
	cout << "NUM THREADS=" << threads << endl;
	if (roiY != NULL)
	{
		std::cout << "ROI=[T,B,L,R]=[" << (*roiY)[0] << "," << (*roiY)[1] << "," << (*roiY)[2] << "," << (*roiY)[3] << "]" << std::endl;
	}
	std::cout << "=============================================" << std::endl;

	auto st = std::chrono::high_resolution_clock::now();

#if MODE==USE_SIMD || MODE==USE_OPENMP

	set_openmp_threads(threads);
#endif

	if (metric == PSNR_METRIC)
		compute_psnr_file(filename, ref_filename, w, h, startFrame, frames, bd);
	else if (metric == WSPSNR_ERP_METRIC)
		compute_wspsnr_file(filename, ref_filename, w, h, startFrame, frames, bd);
	else
	{
		std::cout << "UNKNOWN QUALITY METRIC!" << std::endl;
		return 1;
	}
#if MODE==USE_CUDA

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
#endif

	auto et = std::chrono::high_resolution_clock::now();
	auto  duration = chrono::duration_cast<chrono::milliseconds>(et - st).count() / 1000.0;
	std::cout << "TOTAL PRCOCESS TIME :" << std::setw(3) << std::left << duration << std::endl;

	return 0;
}

