#ifndef REG_BITMAP_CUH
#define REG_BITMAP_CUH

#include <assert.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#include "./copy_traits.cuh"

template <size_t N>
struct RegBitmap {
  // Using 32-bit unsigned integer to store N bits, data is expected to store in
  // registers
  uint32_t data[(N + 31) / 32];

  static constexpr size_t reg_num = (N + 31) / 32;

  __device__ __host__ __forceinline__ RegBitmap() {
// Initialize all bits to 0
#pragma unroll
    for (size_t i = 0; i < reg_num; i++) {
      data[i] = 0;
    }
  }

  // Function to set a bit; index is the position of the bit, value is the value
  // to set (0 or 1)
  __device__ __host__ __forceinline__ void set(size_t index,
                                               bool value = true) {
    // Check if the index is within the range
    assert(index < N);
    // Find the corresponding uint32_t array index
    size_t arrayIndex = index / 32;
    // Find the corresponding bit index
    size_t bitIndex = index % 32;
    if (value) {
      data[arrayIndex] |= (1 << bitIndex);  // Set to 1
    } else {
      data[arrayIndex] &= ~(1 << bitIndex);  // Set to 0
    }
  }

  // Function to get a bit; index is the position of the bit
  __device__ __host__ __forceinline__ bool get(size_t index) const {
    assert(index < N);
    size_t arrayIndex = index / 32;
    size_t bitIndex = index % 32;
    return (data[arrayIndex] & (1 << bitIndex)) != 0;
  }

  __device__ __host__ __forceinline__ void clear_all() {
#pragma unroll
    for (size_t i = 0; i < reg_num; i++) {
      data[i] = 0;
    }
  }

  __device__ __host__ __forceinline__ size_t count_set_bits() const {
    size_t count = 0;
#pragma unroll
    for (size_t i = 0; i < reg_num; i++) {
      count += __popc(data[i]);
    }
    return count;
  }

  __device__ __forceinline__ void print_bits() const {
    for (size_t i = 0; i < reg_num; i++) {
      for (size_t j = 0; j < 32; j++) {
        printf("%d", (data[i] & (1 << j)) != 0);
      }
      printf(" ");
    }
    printf("\n");
  }
};

struct DeviceBitmap {
  // Using 32-bit unsigned integer to store bits, data is expected to store in
  // global memory (N + 31) / 32 32-bit unsigned integers are used to store N
  // bits
  uint32_t *data_ptr_;
  size_t bit_num_;
  __device__ __forceinline__ DeviceBitmap(uint32_t *data_ptr, size_t bit_num)
      : data_ptr_(data_ptr), bit_num_(bit_num) {}

  // Function to set a bit; index is the position of the bit, value is the value
  // to set (0 or 1)
  __device__ __forceinline__ void set(size_t index, bool value = true) {
    assert(index < bit_num_);
    // size_t arrayIndex = index / 32;
    // size_t bitIndex = index % 32;
    // if (value) {
    //   atomicOr(&data_ptr_[arrayIndex], (1 << bitIndex));  // Set to 1
    // } else {
    //   atomicAnd(&data_ptr_[arrayIndex], ~(1 << bitIndex));  // Set to 0
    // }

    // 32-bit bitmap
    volatile uint32_t *volatile_data_ptr =
        reinterpret_cast<volatile uint32_t *>(data_ptr_);

    if (value)
      // magic value to indicate set, not 0 because array is always initialized to 0
      volatile_data_ptr[index] = 0xfefefefe;
    else
      volatile_data_ptr[index] = 0x0;
  }

  __device__ __forceinline__ bool get(size_t index) const {
    assert(index < bit_num_);
    // size_t arrayIndex = index / 32;
    // size_t bitIndex = index % 32;

    // volatile uint32_t *volatile_data_ptr =
    //     reinterpret_cast<volatile uint32_t *>(data_ptr_);
    // uint32_t val = volatile_data_ptr[arrayIndex];
    // return (val & (1 << bitIndex)) != 0;
    return __ldcg(&data_ptr_[index]) == 0xfefefefe;
    // return __ldcg(&data_ptr_[index]) == 0;
  }

  __device__ __forceinline__ void get_async(size_t index,
                                            uint32_t &buffer) const {
    assert(index < bit_num_);
    // size_t arrayIndex = index / 32;

    // volatile uint32_t *volatile_data_ptr =
    //     reinterpret_cast<volatile uint32_t *>(data_ptr_);
    // buffer = volatile_data_ptr[arrayIndex];

    // buffer = __ldcg(&data_ptr_[index]);

    copy_traits::copy_async<uint32_t, uint32_t, 128,
                            copy_traits::CacheOperator::kCacheAtAllLevel>(
        buffer, data_ptr_[index]);
  }

  __device__ __forceinline__ bool get_dump(size_t index,
                                           uint32_t &buffer) const {
    assert(index < bit_num_);
    // size_t bitIndex = index % 32;
    // return (buffer & (1 << bitIndex)) != 0;
    return buffer == 0xfefefefe;
  }
};

#endif  // REG_BITMAP_CUH