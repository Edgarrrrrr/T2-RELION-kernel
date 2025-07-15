#ifndef CHECK_CUH
#define CHECK_CUH

#include <cuda_runtime.h>
#include <stdint.h>

#include <cmath>
#include <iostream>
#include <vector>

struct ErrorComparator {
  double sum_abs_error;  // absolute error sum
  double max_abs_error;  // max absolute error
  double avg_abs_error;  // average absolute error
  double sum_rel_error;  // relative error sum
  double max_rel_error;  // max relative error
  double avg_rel_error;  // average relative error
  int count;      // count of elements involved in absolute error statistics
  int count_rel;  // count of elements involved in relative error statistics

  __host__ __device__ ErrorComparator()
      : sum_abs_error(0.0),
        max_abs_error(0.0),
        avg_abs_error(0.0),
        sum_rel_error(0.0),
        max_rel_error(0.0),
        avg_rel_error(0.0),
        count(0),
        count_rel(0) {}

  __host__ void Print() const {
    if (max_rel_error > 1e-3 || isnan(avg_rel_error)) {
      printf("\033[0;31m");
    }

    printf("Absolute Error Statistics:\n");
    printf("  - Mean Absolute Error (MAE): %12.4e\n", avg_abs_error);
    printf("  - Max  Absolute Error:       %12.4e\n", max_abs_error);
    printf("  - Sum  Absolute Error:       %12.4e\n", sum_abs_error);
    printf("Relative Error Statistics:\n");
    printf("  - Mean Relative Error      : %12.4e\n", avg_rel_error);
    printf("  - Max  Relative Error      : %12.4e\n", max_rel_error);
    printf("  - Sum  Relative Error      : %12.4e\n", sum_rel_error);

    if (max_rel_error > 1e-3 || isnan(avg_rel_error)) {
      printf("\033[0m");
    }
  }

  __host__ static std::string CSVHeader() {
    // return "sum abs error,max abs error,avg abs error,sum rel error,max rel
    // error,avg rel error,count,count rel";
    return "avg rel error";
  }
  __host__ std::string CSVRow() const {
    std::ostringstream oss;
    oss << std::scientific << avg_rel_error;
    // oss << std::scientific
    //     << sum_abs_error << ","
    //     << max_abs_error << ","
    //     << avg_abs_error << ","
    //     << sum_rel_error << ","
    //     << max_rel_error << ","
    //     << avg_rel_error << ","
    //     << count << ","
    //     << count_rel;
    return oss.str();
  }
};

__device__ double atomicMax_double(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    double old_val = __longlong_as_double(assumed);
    double new_val = max(val, old_val);
    unsigned long long int new_val_ll = __double_as_longlong(new_val);
    old = atomicCAS(address_as_ull, assumed, new_val_ll);
  } while (assumed != old);
  return __longlong_as_double(old);
}

template <typename Ta, typename Tb>
__global__ void check_array_equal(Ta *a, Tb *b, size_t size,
                                  ErrorComparator *globalStats) {
  double local_sum_abs = 0.0;
  double local_max_abs = 0.0;
  double local_sum_rel = 0.0;
  double local_max_rel = 0.0;
  int local_count = 0;
  int local_count_rel = 0;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  for (size_t i = idx; i < size; i += stride) {
    double abs_err = abs((double)a[i] - (double)b[i]);
    local_sum_abs += abs_err;
    local_count++;
    if (abs_err > local_max_abs) {
      local_max_abs = abs_err;
    }

    if (abs((double)b[i]) > 1e-12) {
      double rel_err = abs_err / abs((double)b[i]);
      local_sum_rel += rel_err;
      local_count_rel++;
      if (rel_err > local_max_rel) {
        local_max_rel = rel_err;
      }
    }
  }

  if (local_count > 0) {
    atomicAdd(&(globalStats->sum_abs_error), local_sum_abs);
    atomicAdd(&(globalStats->count), local_count);
    atomicMax_double(&(globalStats->max_abs_error), local_max_abs);
  }
  if (local_count_rel > 0) {
    atomicAdd(&(globalStats->sum_rel_error), local_sum_rel);
    atomicAdd(&(globalStats->count_rel), local_count_rel);
    atomicMax_double(&(globalStats->max_rel_error), local_max_rel);
  }
}

template <typename Ta, typename Tb>
ErrorComparator check_array(Ta *d_a, Tb *d_b, size_t size,
                            cudaStream_t stream) {
  ErrorComparator *d_stats;
  ErrorComparator h_stats;
  cudaMallocAsync(&d_stats, sizeof(ErrorComparator), stream);
  cudaMemsetAsync(d_stats, 0, sizeof(ErrorComparator), stream);
  check_array_equal<Ta, Tb><<<208, 256, 0, stream>>>(d_a, d_b, size, d_stats);
  cudaMemcpyAsync(&h_stats, d_stats, sizeof(ErrorComparator),
                  cudaMemcpyDeviceToHost, stream);
  cudaFreeAsync(d_stats, stream);
  cudaStreamSynchronize(stream);
  HANDLE_ERROR(cudaGetLastError());

  if (h_stats.count > 0) {
    h_stats.avg_abs_error = h_stats.sum_abs_error / h_stats.count;
  }
  if (h_stats.count_rel > 0) {
    h_stats.avg_rel_error = h_stats.sum_rel_error / h_stats.count_rel;
  }

  return h_stats;
}

#endif