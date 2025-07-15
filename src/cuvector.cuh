#ifndef CUVECTOR_H
#define CUVECTOR_H

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

template <typename T>
class CuVector {
 public:
  CuVector(size_t size, cudaStream_t stream = cudaStreamPerThread)
      : size_(size), stream_(stream), device_ptr_(nullptr), host_ptr_(size) {
    host_ptr_.resize(size);
    cudaMalloc(&device_ptr_, size * sizeof(T));
  }

  CuVector() : size_(0), stream_(0), device_ptr_(nullptr) {}

  ~CuVector() {
    if (device_ptr_ != nullptr) {
      cudaFree(device_ptr_);
    }
  }

  T* host_ptr() { return host_ptr_.data(); }
  T* data() { return host_ptr_.data(); }

  T* device_ptr() { return device_ptr_; }

  size_t size() const { return size_; }

  void copy_to_device() {
    cudaMemcpyAsync(device_ptr_, host_ptr_.data(), size_ * sizeof(T),
                    cudaMemcpyHostToDevice, stream_);
    cudaStreamSynchronize(stream_);
  }

  void copy_to_host() {
    cudaMemcpyAsync(host_ptr_.data(), device_ptr_, size_ * sizeof(T),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
  }

  void copy_to_device_async() {
    cudaMemcpyAsync(device_ptr_, host_ptr_.data(), size_ * sizeof(T),
                    cudaMemcpyHostToDevice, stream_);
  }

  void copy_to_host_async() {
    cudaMemcpyAsync(host_ptr_.data(), device_ptr_, size_ * sizeof(T),
                    cudaMemcpyDeviceToHost, stream_);
  }

  void sync() { cudaStreamSynchronize(stream_); }

  void set_stream(cudaStream_t stream) { stream_ = stream; }

  void resize(size_t size) {
    size_ = size;
    host_ptr_.resize(size);
    cudaFree(device_ptr_);
    cudaMalloc(&device_ptr_, size_ * sizeof(T));
    copy_to_device();
  }

  CuVector<T> deepcopy() const {
    CuVector<T> new_vec(size_, stream_);
    new_vec.host_ptr_ = host_ptr_;

    cudaMemcpyAsync(new_vec.device_ptr_, device_ptr_, size_ * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream_);
    cudaStreamSynchronize(stream_);

    return new_vec;
  }

  void print_snippet() {
    std::cout << "CuVector snippet: type : " << typeid(T).name()
              << " size : " << size_ << std::endl;
    std::cout << "    host_ptr_ : ";
    for (int i = 0; i < std::min(size_, (size_t)50); i++) {
      std::cout << host_ptr_[i] << " ";
    }
    std::cout << std::endl;
  }

  T& operator[](size_t index) { return host_ptr_[index]; }

  CuVector(const CuVector<T>& other) {
    size_ = other.size_;
    stream_ = other.stream_;
    host_ptr_ = other.host_ptr_;
    cudaMalloc(&device_ptr_, size_ * sizeof(T));
    cudaMemcpyAsync(device_ptr_, other.device_ptr_, size_ * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream_);
    cudaStreamSynchronize(stream_);
  }

  CuVector<T>& operator=(const CuVector<T>& other) {
    if (this == &other) {
      return *this;
    }
    size_ = other.size_;
    stream_ = other.stream_;
    host_ptr_ = other.host_ptr_;
    cudaMalloc(&device_ptr_, size_ * sizeof(T));
    cudaMemcpyAsync(device_ptr_, other.device_ptr_, size_ * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream_);
    cudaStreamSynchronize(stream_);
    return *this;
  }

  CuVector(CuVector<T>&& other) {
    size_ = other.size_;
    stream_ = other.stream_;
    host_ptr_ = std::move(other.host_ptr_);
    device_ptr_ = other.device_ptr_;
    other.device_ptr_ = nullptr;
  }

  CuVector<T>& operator=(CuVector<T>&& other) {
    if (this == &other) {
      return *this;
    }
    size_ = other.size_;
    stream_ = other.stream_;
    host_ptr_ = std::move(other.host_ptr_);
    device_ptr_ = other.device_ptr_;
    other.device_ptr_ = nullptr;
    return *this;
  }

  template <typename U>
  CuVector<U> convert_to() {
    copy_to_host();
    CuVector<U> result(size_, stream_);
    for (size_t i = 0; i < size_; ++i) {
      result[i] = static_cast<U>(host_ptr_[i]);
    }
    result.copy_to_device();
    return result;
  }

 private:
  size_t size_;              // vector size
  cudaStream_t stream_;      // CUDA stream
  T* device_ptr_;            // device raw pointer
  std::vector<T> host_ptr_;  // host data
};

#endif  // CUVECTOR_H