# YUVQATool

**YUVQATool** is a lightweight and high-performance quality assessment tool for YUV videos. It is purly implemented in **C++** and supports acceleration via **OPENMP**, **SIMD (AVX2)**, and **CUDA**.

## Features
- 📊 Supports **region of interest (ROI)** or **full frame** computation of:
  - **PSNR** 
  - **WS-PSNR** for equirectangular projection (ERP) format

## Supported Formats
- Raw YUV 4:2:0 video
- Bit-depth: 8-bit/10-bit

## Dependencies
- C++17 compatible compiler
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (CUDA ≥ 12.8, arch ≥ compute_61, code ≥ sm_61)

## Build Instructions
Change the value of **#define MODE** macro in the **config.h** to use OPENMP, SIMD, or CUDA:
```sh
- #define MODE  USE_NORMAL_LOOP   
- #define MODE  USE_OPENMP     
- #define MODE  USE_SIMD
- #define MODE  USE_CUDA
```

### Windows
The project is originally written in Visual Studio 2022. The solution file for the Visual Studio project exists in Git.
### Linux
The code can also be executed in Linux using the GCC compiler. The build command will be added soon... 

### Command-line Options

| Option  | Description                                                                                     |
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














