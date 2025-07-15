#ifndef KERNEL_TEST_RESULT_H
#define KERNEL_TEST_RESULT_H

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "./check.cuh"

struct KernelTestResult {
  std::string data_name_;
  int translation_num_;
  int orientation_num_;
  int image_size_;
  double original_time_us_;
  std::vector<double> optimized_times_us_;
  std::vector<ErrorComparator> error_comparators_;
  std::vector<std::string> optimized_description_;

  KernelTestResult() = default;

  KernelTestResult(const std::string& data_name, int translation_num,
                   int orientation_num, int image_size, double original_time_us)
      : data_name_(data_name),
        translation_num_(translation_num),
        orientation_num_(orientation_num),
        image_size_(image_size),
        original_time_us_(original_time_us) {}

  // void AddOptimizedTime(const std::string& description, double time_us) {
  //   optimized_description_.push_back(description);
  //   optimized_times_us_.push_back(time_us);
  // }

  void AddOptimizationResult(
      const std::string& description, double time_us,
      const ErrorComparator& error_comparator) {
    optimized_description_.push_back(description);
    optimized_times_us_.push_back(time_us);
    error_comparators_.push_back(error_comparator);
  }

  void Print() const {
    std::cout << "Data Name: " << data_name_ << std::endl;
    std::cout << "Translation Num: " << translation_num_ << std::endl;
    std::cout << "Orientation Num: " << orientation_num_ << std::endl;
    std::cout << "Image Size: " << image_size_ << std::endl;
    std::cout << "Original Time (us): " << original_time_us_ << std::endl;
    for (size_t i = 0; i < optimized_times_us_.size(); ++i) {
      std::cout << "--------------------------------" << std::endl;
      std::cout << optimized_description_[i]
                << "   Time (us): " << optimized_times_us_[i] << std::endl;
      error_comparators_[i].Print();
    }
  }

  std::string CSVHeader() const {
    std::string header = "data name,translation num,orientation num,image size,original time (us)";
    std::string optimized_header_section =
        "optimized description,optimized times (us)," + ErrorComparator::CSVHeader();
    for (size_t i = 0; i < optimized_times_us_.size(); ++i) {
      header += "," + optimized_header_section;
    }
    return header;
  }

  std::string CSVRow() const {
    std::string row = data_name_ + "," + std::to_string(translation_num_) + "," +
                      std::to_string(orientation_num_) + "," +
                      std::to_string(image_size_) + "," +
                      std::to_string(original_time_us_);
    for (size_t i = 0; i < optimized_times_us_.size(); ++i) {
      row += "," + optimized_description_[i] + "," +
             std::to_string(optimized_times_us_[i]) + "," +
             error_comparators_[i].CSVRow();
    }
    return row;
  }
};

std::string KernelTestResultsToCSV(const std::vector<KernelTestResult>& results) {
  // assert all results have the same optimization descriptions and the same
  // number of optimized times
  if (results.empty()) {
    return "";
  }
  std::string csv = results[0].CSVHeader() + "\n";
  for (const auto& result : results) {
    csv += result.CSVRow() + "\n";
  }
  return csv;
}

#endif  // KERNEL_TEST_RESULT_H