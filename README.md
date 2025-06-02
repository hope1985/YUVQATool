# YUVQATool

**YUVQATool** is a lightweight and high-performance quality assessment tool for YUV videos. It is purly implemented in **C++** and supports acceleration via **SIMD (AVX2)** and **CUDA**.

## Features

- ⚙️ Written entirely in C++
- 🚀 SIMD (AVX2) acceleration using 
- 🚀 OPENMP parallel computation
- ⚡ GPU acceleration with CUDA
- 📊 Supports computation of:
  - **PSNR** (Peak Signal-to-Noise Ratio)
  - **WS-PSNR** (Weighted-to-Spherically-uniform PSNR) for equirectangular projection (ERP) format

## Supported Formats

- Raw YUV 4:2:0 video
- Bit-depth: 8-bit (support for 10-bit can be added upon request)

## Dependencies
- C++17 compatible compiler
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (CUDA ≥ 12.8, compute_61, sm_61)

## Build Instructions
Change the value of **#define MODE** macro in the **config.h** to use OPENMP, SIMD, or CUDA:
```sh
- #define MODE  USE_NORMAL_LOOP   
- #define MODE  USE_OPENMP     
- #define MODE  USE_SIMD
- #define MODE  USE_CUDA
```

### Window
The project is originally written in Visual Studio 2022. The solution file of the project exists in the git.
### Linux
The code can also be written in Linux using the GCC compiler. The build command will be added soon... 

### Command-line Options

| Option  | Description/Options                                                                                     |
|---------|---------------------------------------------------------------------------------------------------------|
| `-i`    | Filepath of the input YUV file                                                                          |
| `-r`    | Filepath of the reference YUV file                                                                      |
| `-w`    | Width of the YUV file                                                                                   |
| `-h`    | Height of the YUV file                                                                                  |
| `-bd`   | Bit-depth of the YUV file (8 or 10)                                                                     |
| `-sf`   | Start frame index to begin computing the quality metric                                                 |
| `-nf`   | Number of frames to compute quality metric                                               |
| `-qm`   | Quality metric type ('0' = PSNR, '1' = WSPSNR for ERP format)                                           |
| `-roi`  | Region of interest for quality computation as `[Top, Bottom, Left, Right]`                              |
| `-nt`   | Number of threads (used in USE_OPENMP/USE_SIMD modes); set '<= 0` or '>= core count' to use all physical cores)     |


### Usage














