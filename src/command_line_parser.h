#ifndef COMMAND_LINE_PARSER_H_
#define COMMAND_LINE_PARSER_H_

#include <algorithm>
#include <exception>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// Exception class for command line parser errors
class CommandLineParserException : public std::exception {
 public:
  explicit CommandLineParserException(const std::string& message)
      : message_(message) {}

  const char* what() const noexcept override { return message_.c_str(); }

 private:
  std::string message_;
};

// CommandLineParser handles parsing of command line arguments
class CommandLineParser {
 public:
  CommandLineParser() = default;

  // Adds an option with long and short names, description, and required flag
  void AddOption(const std::string& long_option,
                 const std::string& short_option,
                 const std::string& description, bool required = false) {
    Option option;
    option.long_option = long_option;
    option.short_option = short_option;
    option.description = description;
    option.required = required;
    option.is_set = false;
    option.value = "";

    // Insert into long option map
    options_[long_option] = option;

    // Map short option to long option
    if (!short_option.empty()) {
      short_to_long_[short_option] = long_option;
    }
  }

  // Parses the command line arguments
  void Parse(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];

      if (arg.rfind("--", 0) == 0) {  // Long option
        std::string opt = arg.substr(2);
        auto it = options_.find(opt);
        if (it == options_.end()) {
          throw CommandLineParserException("Unknown option: --" + opt);
        }

        // Check if next argument exists and is not another option
        if (i + 1 < argc && argv[i + 1][0] != '-') {
          it->second.value = argv[++i];
        } else {
          throw CommandLineParserException("Option requires a value: --" + opt);
        }

        it->second.is_set = true;
      } else if (arg.rfind("-", 0) == 0 && arg.length() > 1) {  // Short option
        std::string opt = arg.substr(1);
        auto it_short = short_to_long_.find(opt);
        if (it_short == short_to_long_.end()) {
          throw CommandLineParserException("Unknown short option: -" + opt);
        }

        std::string long_opt = it_short->second;
        auto it = options_.find(long_opt);
        if (it == options_.end()) {
          throw CommandLineParserException("Unknown option: --" + long_opt);
        }

        // Check if next argument exists and is not another option
        if (i + 1 < argc && argv[i + 1][0] != '-') {
          it->second.value = argv[++i];
        } else {
          throw CommandLineParserException("Option requires a value: --" +
                                           long_opt);
        }

        it->second.is_set = true;
      } else {  // Positional or unknown argument
        throw CommandLineParserException("Unknown argument: " + arg);
      }
    }

    // Verify all required options are set
    for (const auto& pair : options_) {
      const Option& opt = pair.second;
      if (opt.required && !opt.is_set) {
        throw CommandLineParserException("Missing required option: --" +
                                         opt.long_option);
      }
    }
  }

  // Retrieves the value of a given long option
  std::string GetOptionValue(const std::string& long_option) const {
    auto it = options_.find(long_option);
    if (it != options_.end() && it->second.is_set) {
      return it->second.value;
    }
    throw CommandLineParserException("Option not set or does not exist: --" +
                                     long_option);
  }

  // Checks if a given long option was set
  bool HasOption(const std::string& long_option) const {
    auto it = options_.find(long_option);
    return (it != options_.end()) && (it->second.is_set);
  }

  // Prints help message with all available options
  void PrintHelp(const std::string& program_name) const {
    std::cout << "Usage: " << program_name << " [options]\n\n";
    std::cout << "Options:\n";

    for (const auto& pair : options_) {
      const Option& opt = pair.second;
      std::cout << "  ";
      if (!opt.short_option.empty()) {
        std::cout << "-" << opt.short_option << ", ";
      } else {
        std::cout << "    ";
      }
      std::cout << "--" << opt.long_option;
      std::cout << "\t" << opt.description;
      if (opt.required) {
        std::cout << " (required)";
      }
      std::cout << "\n";
    }

    std::cout << "\n";
  }

 private:
  struct Option {
    std::string long_option;
    std::string short_option;
    std::string description;
    bool required;
    bool is_set;
    std::string value;
  };

  std::map<std::string, Option> options_;
  std::map<std::string, std::string> short_to_long_;
};

#endif  // COMMAND_LINE_PARSER_H_
