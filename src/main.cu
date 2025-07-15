#include <cuda_runtime.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <nvtx3/nvtx3.hpp>
#include <string>

#include "./acc_helper_functions_impl.h"
#include "./acc_projector.h"
#include "./acc_projector_impl.h"
#include "./acc_projectorkernel_impl.h"
#include "./coarse_kernel.cuh"
#include "./command_line_parser.h"
#include "./cuvector.cuh"
#include "./diff2.cuh"
#include "./kernel_test_result.h"

/**
 * ----------
 * This tool runs performance tests on various “coarse” kernels featuring different optimizations, 
 * across multiple data iterations. Results are aggregated and saved as a CSV file.
 *
 * Usage:
 *   ./build/test_coarse -i <input_dir> -o <output.csv> [-d <data_per_iter>] [-t <test_times>]
 *
 * Options:
 *   -i input_dir    Path to input data files (required)
 *   -o output.csv   Path to CSV output file (required)
 *   -d data_num     Number of data files per iteration (default: 1000)
 *  
 */

struct CoarseParam {
  // projector kernel
  int xdim, ydim, zdim, rmax;

  // kernel parameters
  int translation_num;
  int orientation_num;
  int image_size;
  // 9 * orientation_num
  CuVector<XFLOAT> eulers;

  // 2 * translation_num trans_x = trans_xyz,
  // trans_y = trans_xyz + translation_num
  CuVector<XFLOAT> trans_xyz;

  // 2 * image_size  real = fimg, imag = fimg + image_size
  CuVector<XFLOAT> fimg;

  // image_size
  CuVector<XFLOAT> corr;

  // orientation_num * translation_num
  CuVector<XFLOAT> diff2s;

  int filtered_image_size;
  // 2 * filtered_image_size
  CuVector<XFLOAT> fimg_filtered;

  // 2 * filtered_image_size
  CuVector<XFLOAT> coor_filtered;

  // filtered_image_size
  CuVector<XFLOAT> corr_filtered;

  // projector
  std::shared_ptr<AccProjector> projector;
  AccProjectorKernel projector_kernel;

  void PrintSnippet() {
    printf("=======================  CoarseParam  =======================\n");
    if (projector == nullptr) {
      printf("CoarseParam is not initialized!\n");
      return;
    }
    printf("Img dimensions     : %6d x %6d x %6d\n", xdim, ydim, zdim);
    printf("Maximal radius     : %6d\n", rmax);
    printf("Translation number : %6d\n", translation_num);
    printf("Orientation number : %6d\n", orientation_num);
    printf("Image size         : %6d\n", image_size);
    printf("Filtered image size: %6d\n", filtered_image_size);
  }
};

void FilterCoarseParamImage(CoarseParam& param, cudaStream_t stream) {
  param.filtered_image_size = 0;
  std::vector<XFLOAT> fimg_filtered_real, fimg_filtered_imag, coor_filtered_x,
      coor_filtered_y, corr_filtered;
  fimg_filtered_real.resize(param.image_size);
  fimg_filtered_imag.resize(param.image_size);
  coor_filtered_x.resize(param.image_size);
  coor_filtered_y.resize(param.image_size);
  corr_filtered.resize(param.image_size);

  int ptr_start = 0;
  int ptr_end = param.image_size - 1;
  for (int i = 0; i < param.image_size; i++) {
    int x, y;
    pixel_index2coor(i, param.projector_kernel.imgX,
                     param.projector_kernel.imgY, param.projector_kernel.maxR,
                     x, y);
    float r2 = (x * x + y * y) * param.projector_kernel.padding_factor *
               param.projector_kernel.padding_factor;

    if (r2 <= param.projector_kernel.maxR2_padded) {
      fimg_filtered_real[ptr_start] = param.fimg[i];
      fimg_filtered_imag[ptr_start] = param.fimg[i + param.image_size];
      coor_filtered_x[ptr_start] = x;
      coor_filtered_y[ptr_start] = y;
      corr_filtered[ptr_start] = param.corr[i];

      ptr_start++;
      param.filtered_image_size++;
    } else {
      fimg_filtered_real[ptr_end] = param.fimg[i];
      fimg_filtered_imag[ptr_end] = param.fimg[i + param.image_size];
      coor_filtered_x[ptr_end] = x;
      coor_filtered_y[ptr_end] = y;
      corr_filtered[ptr_end] = param.corr[i];
      ptr_end--;
    }
  }
  assert(ptr_start == ptr_end + 1);
  if (ptr_start != ptr_end + 1) {
    printf("ptr_start != ptr_end + 1: %d %d\n", ptr_start, ptr_end);
  }

  if (param.filtered_image_size == 0) {
    printf("No filtered image found! maxR2_padded : %12f\n",
           (float)param.projector_kernel.maxR2_padded);
  }

  param.fimg_filtered = CuVector<XFLOAT>(2 * param.image_size, stream);
  param.corr_filtered = CuVector<XFLOAT>(param.image_size, stream);
  param.coor_filtered = CuVector<XFLOAT>(2 * param.image_size, stream);
  for (int i = 0; i < param.image_size; i++) {
    param.fimg_filtered[i] = fimg_filtered_real[i];
    param.fimg_filtered[i + param.image_size] = fimg_filtered_imag[i];
    param.corr_filtered[i] = corr_filtered[i];
    param.coor_filtered[2 * i] = coor_filtered_x[i];
    param.coor_filtered[2 * i + 1] = coor_filtered_y[i];
  }
  param.coor_filtered.copy_to_device();
  param.fimg_filtered.copy_to_device();
  param.corr_filtered.copy_to_device();

  HANDLE_ERROR(cudaGetLastError());
}

CoarseParam CreateCoarseParamFromFile(const std::string& filename,
                                      cudaStream_t stream) {
  CoarseParam param;
  std::ifstream infile(filename, std::ios::binary);
  if (!infile) {
    std::cerr << "File not found: " << filename << std::endl;
    exit(EXIT_FAILURE);
  }

  // Read the header
  infile.read(reinterpret_cast<char*>(&param.xdim), sizeof(int));
  infile.read(reinterpret_cast<char*>(&param.ydim), sizeof(int));
  infile.read(reinterpret_cast<char*>(&param.zdim), sizeof(int));
  infile.read(reinterpret_cast<char*>(&param.rmax), sizeof(int));
  infile.read(reinterpret_cast<char*>(&param.translation_num), sizeof(int));
  infile.read(reinterpret_cast<char*>(&param.orientation_num), sizeof(int));
  infile.read(reinterpret_cast<char*>(&param.image_size), sizeof(int));

  // Read the data
  param.eulers = CuVector<float>(9 * param.orientation_num, stream);
  infile.read(reinterpret_cast<char*>(param.eulers.host_ptr()),
              sizeof(float) * 9 * param.orientation_num);

  param.trans_xyz = CuVector<float>(2 * param.translation_num, stream);
  infile.read(reinterpret_cast<char*>(param.trans_xyz.host_ptr()),
              sizeof(float) * 2 * param.translation_num);

  param.fimg = CuVector<float>(2 * param.image_size, stream);
  infile.read(reinterpret_cast<char*>(param.fimg.host_ptr()),
              sizeof(float) * 2 * param.image_size);

  param.corr = CuVector<float>(param.image_size, stream);
  infile.read(reinterpret_cast<char*>(param.corr.host_ptr()),
              sizeof(float) * param.image_size);

  param.diff2s =
      CuVector<float>(param.orientation_num * param.translation_num, stream);
  infile.read(reinterpret_cast<char*>(param.diff2s.host_ptr()),
              sizeof(float) * param.orientation_num * param.translation_num);

  // Perform the copy to device
  param.eulers.copy_to_device();
  param.trans_xyz.copy_to_device();
  param.fimg.copy_to_device();
  param.corr.copy_to_device();
  param.diff2s.copy_to_device();

  HANDLE_ERROR(cudaGetLastError());
  infile.close();
  return param;
}

std::shared_ptr<AccProjector> CreateProjectorFromFile(
    const std::string& filename) {
  auto proj = std::make_shared<AccProjector>();
  HANDLE_ERROR(cudaGetLastError());
  proj->constructFromFile(filename);
  HANDLE_ERROR(cudaGetLastError());
  return proj;
}

KernelTestResult TestCoarseKernel(CoarseParam& param, int test_times = 100,
                                  std::string data_name = "default",
                                  cudaStream_t stream = 0) {
  // =====================================================================
  // 1. Basic initialization and settings
  // =====================================================================
  KernelTestResult kernel_test_result;
  kernel_test_result.data_name_ = data_name;
  kernel_test_result.translation_num_ = param.translation_num;
  kernel_test_result.orientation_num_ = param.orientation_num;
  kernel_test_result.image_size_ = param.image_size;

  param.projector_kernel = AccProjectorKernel::makeKernel(
      *param.projector, param.xdim, param.ydim, param.zdim, param.rmax);
  HANDLE_ERROR(cudaGetLastError());
  FilterCoarseParamImage(param, stream);
  HANDLE_ERROR(cudaGetLastError());
  param.PrintSnippet();

  float* eulers = param.eulers.device_ptr();
  float* trans_x = param.eulers.device_ptr();
  float* trans_y = param.eulers.device_ptr() + param.translation_num;
  float* fimg_real = param.fimg.device_ptr();
  float* fimg_imag = param.fimg.device_ptr() + param.image_size;
  float* corr = param.corr.device_ptr();
  float* fimg_filtered_real = param.fimg_filtered.device_ptr();
  float* fimg_filtered_imag =
      param.fimg_filtered.device_ptr() + param.image_size;
  float* corr_filtered = param.corr_filtered.device_ptr();
  float* coor_filtered = param.coor_filtered.device_ptr();

  HANDLE_ERROR(cudaStreamSynchronize(stream));
  param.corr.copy_to_host();

  // =====================================================================
  // 2. Generate reference result: first use double precision kernel to
  //    calculate the reference result
  // =====================================================================
  CuVector<float> diff2s_ref = param.diff2s.deepcopy();
  HANDLE_ERROR(cudaStreamSynchronize(stream));
  auto diff2s_d = diff2s_ref.convert_to<double>();
  double* diff2s_d_ptr = diff2s_d.device_ptr();

  runDiff2KernelCoarseDouble(param.projector_kernel, trans_x, trans_y, nullptr,
                             corr, fimg_real, fimg_imag, eulers, diff2s_d_ptr,
                             1.,  // local_sqrtXi2
                             param.orientation_num, param.translation_num,
                             param.image_size, stream,
                             false,  // do_CC
                             false,
                             false);  // dataIs3D
  diff2s_d.copy_to_host();
  cudaStreamSynchronize(stream);
  HANDLE_ERROR(cudaGetLastError());

  CuVector<float> diff2s_ref_filter = param.diff2s.deepcopy();
  HANDLE_ERROR(cudaStreamSynchronize(stream));
  auto diff2s_d_filter = diff2s_ref_filter.convert_to<double>();
  double* diff2s_d_filter_ptr = diff2s_d_filter.device_ptr();

  runDiff2KernelCoarseDouble(
      param.projector_kernel, trans_x, trans_y, nullptr, corr, fimg_real,
      fimg_imag, eulers, diff2s_d_filter_ptr,
      1.,  // local_sqrtXi2
      param.orientation_num, param.translation_num, param.image_size, stream,
      false,  // do_CC
      false,
      true);  // dataIs3D
  diff2s_d_filter.copy_to_host();
  cudaStreamSynchronize(stream);
  HANDLE_ERROR(cudaGetLastError());

  // =====================================================================
  // 3. Use the original kernel to get the baseline result
  // =====================================================================
  CuVector<float> diff2s_ori = param.diff2s.deepcopy();
  float* diff2s_ori_ptr = diff2s_ori.device_ptr();

  CuVector<float> diff2s_ori_buf = param.diff2s.deepcopy();
  float* diff2s_ori_buf_ptr = diff2s_ori_buf.device_ptr();

  // run for correctness check
  {
    nvtx3::scoped_range r{"kernel"};
    runDiff2KernelCoarse(param.projector_kernel, trans_x, trans_y, nullptr,
                         corr, fimg_real, fimg_imag, eulers, diff2s_ori_ptr,
                         1.,  // local_sqrtXi2
                         param.orientation_num, param.translation_num,
                         param.image_size, stream,
                         false,   // do_CC
                         false);  // dataIs3D
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream);
  for (int i = 0; i < test_times; i++) {
    runDiff2KernelCoarse(param.projector_kernel, trans_x, trans_y, nullptr,
                         corr, fimg_real, fimg_imag, eulers, diff2s_ori_buf_ptr,
                         1.,  // local_sqrtXi2
                         param.orientation_num, param.translation_num,
                         param.image_size, stream,
                         false,   // do_CC
                         false);  // dataIs3D
  }
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float baseline_ms = 0;
  cudaEventElapsedTime(&baseline_ms, start, stop);
  float time_us = baseline_ms * 1000 / test_times;
  kernel_test_result.original_time_us_ = time_us;  // Update to use time_us
  HANDLE_ERROR(cudaGetLastError());
  auto error_comparator =
      check_array(diff2s_ori_ptr, diff2s_d_ptr,
                  param.translation_num * param.orientation_num, stream);
  kernel_test_result.AddOptimizationResult("original", time_us,
                                           error_comparator);

  auto TimeKernelRun = [&](auto&& kernel,
                           const std::string& description) -> void {
    CuVector<float> diff2s_opt = param.diff2s.deepcopy();
    float* diff2s_opt_ptr = diff2s_opt.device_ptr();

    CuVector<float> diff2s_opt_buf = param.diff2s.deepcopy();
    float* diff2s_opt_buf_ptr = diff2s_opt_buf.device_ptr();

    size_t workspace_size = kernel.get_workspace_size_bytes();
    CuVector<uint32_t> diff2s_workspace(workspace_size, stream);
    diff2s_workspace.copy_to_device();
    cudaMemset(diff2s_workspace.device_ptr(), 0xff, workspace_size);

    // run for correctness check
    {
      nvtx3::scoped_range r{"kernel"};
      kernel.run(eulers, trans_x, trans_y, fimg_filtered_real,
                 fimg_filtered_imag, param.projector_kernel, corr_filtered,
                 diff2s_opt_ptr, diff2s_opt_ptr, stream, coor_filtered,
                 diff2s_workspace.device_ptr());
    }

    std::vector<CuVector<uint32_t>> diff2s_workspaces;
    diff2s_workspaces.reserve(test_times);
    for (int i = 0; i < test_times; i++) {
      diff2s_workspaces.emplace_back(workspace_size, stream);
      diff2s_workspaces.back().copy_to_device();
      cudaMemset(diff2s_workspaces.back().device_ptr(), 0xff, workspace_size);
    }

    // run for performance test
    cudaEvent_t t_start, t_stop;
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_stop);
    cudaStreamSynchronize(stream);
    cudaEventRecord(t_start, stream);
    {
      for (int i = 0; i < test_times; i++) {
        // Only timing this line
        kernel.run(eulers, trans_x, trans_y, fimg_filtered_real,
                   fimg_filtered_imag, param.projector_kernel, corr_filtered,
                   diff2s_opt_buf_ptr, diff2s_opt_buf_ptr, stream,
                   coor_filtered, diff2s_workspaces[i].device_ptr());
      }
    }
    cudaEventRecord(t_stop, stream);
    cudaEventSynchronize(t_stop);
    float t_ms = 0;
    cudaEventElapsedTime(&t_ms, t_start, t_stop);
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_stop);
    float time_us = t_ms * 1000 / test_times;

    diff2s_workspace.copy_to_host();
    auto error_comparator =
        check_array(diff2s_opt_ptr, diff2s_d_ptr,
                    param.translation_num * param.orientation_num, stream);

    kernel_test_result.AddOptimizationResult(description, time_us,
                                             error_comparator);
  };

  int sm_num_ = 0;
  cudaDeviceGetAttribute(&sm_num_, cudaDevAttrMultiProcessorCount, /*device=*/0);

  TimeKernelRun(
      CoarseMatrixKernelCudaCoreIm2colSplitImgNfuse(
          param.translation_num, param.orientation_num, param.image_size, sm_num_),
      std::string("Im2col Multi-Level Blocking + CUDA Core"));

  TimeKernelRun(
      CoarseMatrixKernelIm2colSplitImgNfuse(
          param.translation_num, param.orientation_num, param.image_size, sm_num_),
      std::string("Im2col Multi-Level Blocking + Tensor Core"));

  TimeKernelRun(
      CoarseMatrixKernelIm2colSplitImgBCFNfuse(
          param.translation_num, param.orientation_num, param.image_size, sm_num_),
      std::string("+ Conflict Removal"));

  TimeKernelRun(CoarseMatrixKernelIm2colSplitImgBCFProjOverlapNfuse(
                    param.translation_num, param.orientation_num,
                    param.filtered_image_size, param.image_size, sm_num_),
                std::string("+ Register-Based Texture Fetch Masking"));

  TimeKernelRun(CoarseMatrixKernelIm2colSplitImgBCFProjOverlapTransDBNfuse(
                    param.translation_num, param.orientation_num,
                    param.filtered_image_size, param.image_size, sm_num_),
                std::string("+ Collaborative Thread-Block-Level Data Reuse"));

  kernel_test_result.Print();

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  HANDLE_ERROR(cudaGetLastError());
  return kernel_test_result;
}

int main(int argc, char* argv[]) {
  CommandLineParser cmd_parser;

  cmd_parser.AddOption("data_num_per_iter", "d", "Number of data per iteration",
                       false);
  cmd_parser.AddOption("test_times", "t", "Number of test times per kernel",
                       false);
  cmd_parser.AddOption("result_file", "o", "File path for result output", true);
  cmd_parser.AddOption("input_dir", "i", "Input directory for the data", true);

  int total_iters = 19;
  int data_num_per_iter = 1000;
  int test_times = 100;
  std::string result_file_path;
  std::string input_dir;

  try {
    cmd_parser.Parse(argc, argv);

    data_num_per_iter =
        std::stoi(cmd_parser.GetOptionValue("data_num_per_iter"));
    test_times = std::stoi(cmd_parser.GetOptionValue("test_times"));
    result_file_path = cmd_parser.GetOptionValue("result_file");
    input_dir = cmd_parser.GetOptionValue("input_dir");

    std::cout << "Total Data per Iteration: " << data_num_per_iter << std::endl;
    std::cout << "Total Test Times per Kernel: " << test_times << std::endl;
    std::cout << "Result File Path: " << result_file_path << std::endl;
    std::cout << "Input Directory: " << input_dir << std::endl;

    std::ofstream result_file(result_file_path);
    if (!result_file) {
      std::cerr << "Failed to open result file: " << result_file_path
                << std::endl;
      return 1;
    }
    result_file.close();
  } catch (const CommandLineParserException& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    cmd_parser.PrintHelp(argv[0]);
    return 1;
  }

  std::vector<KernelTestResult> kernel_test_results;
  for (int iter = 1; iter <= total_iters; iter++) {
    std::string projectorFile = input_dir + "/projector/" +
                                "projector_proc_1_" + std::to_string(iter) +
                                ".dat";
    std::shared_ptr<AccProjector> proj_ptr =
        CreateProjectorFromFile(projectorFile);

    for (int data = 1; data <= data_num_per_iter; data++) {
      cudaStream_t stream;
      cudaStreamCreate(&stream);

      HANDLE_ERROR(cudaGetLastError());

      std::string dataFile = input_dir + "/" + std::to_string(iter) + "/coarse_data" +
                             std::to_string(data) + "00_1" + ".dat";
      CoarseParam param = CreateCoarseParamFromFile(dataFile, stream);
      param.projector = proj_ptr;

      HANDLE_ERROR(cudaGetLastError());

      KernelTestResult kernel_test_result = TestCoarseKernel(
          param, test_times, std::to_string(iter) + "_" + std::to_string(data),
          stream);
      HANDLE_ERROR(cudaGetLastError());

      kernel_test_results.push_back(kernel_test_result);
      std::cout << "Finished processing iter: " << iter << ", data: " << data
                << std::endl;
      cudaStreamDestroy(stream);
    }
    std::cout << "Finished processing iter: " << iter << std::endl;
  }

  auto kernel_test_result_csv = KernelTestResultsToCSV(kernel_test_results);
  std::ofstream csv_file(result_file_path);
  if (csv_file.is_open()) {
    csv_file << kernel_test_result_csv;
    csv_file.close();
    std::cout << "CSV file saved successfully." << std::endl;
  } else {
    std::cerr << "Unable to open file for writing." << std::endl;
  }
  return 0;
}