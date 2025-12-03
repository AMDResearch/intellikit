/****************************************************************************
MIT License

Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
****************************************************************************/

#pragma once

#include <cstdio>

#include <fmt/core.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

extern "C" {
extern char** environ;
}

namespace intelliperf {

namespace detail {

inline void print_env_variables() {
  char** env = environ;
  while (*env) {
    std::cout << *env << std::endl;
    ++env;
  }
}

enum struct LogLevel {
  NONE,
  INFO,
  ERROR,
  DETAIL,
};
constexpr auto operator+(LogLevel logLevel) noexcept {
  return static_cast<std::underlying_type_t<LogLevel>>(logLevel);
}

constexpr auto log_level_to_string(const LogLevel level) {
  if (level == LogLevel::INFO) {
    return "INFO";
  } else if (level == LogLevel::ERROR) {
    return "ERROR";
  } else if (level == LogLevel::DETAIL) {
    return "DETAIL";
  }
  return "";
}
template <typename... Args>
inline void log_message(const LogLevel level,
                        const char* file,
                        int line,
                        const char* msg,
                        Args... args) {
  static const char* log_env = std::getenv("ACCORDO_LOG_LEVEL");
  if (log_env) {
    static auto log_level_env = std::atoi(log_env);

    if (log_level_env >= +level) {
      const char* color_reset = "\033[0m";
      const char* color_info = "\033[37m";
      const char* color_error = "\033[31m";

      const char* color = level == LogLevel::ERROR ? color_error : color_info;

      std::string formatted_message;
      if constexpr (sizeof...(args) > 0) {
        formatted_message = fmt::vformat(msg, fmt::make_format_args(args...));
      } else {
        formatted_message = msg;
      }

      std::printf("%s[%s]: [%s:%d] %s%s\n",
                  color,
                  log_level_to_string(level),
                  file,
                  line,
                  formatted_message.c_str(),
                  color_reset);

      static const char* log_file = std::getenv("ACCORDO_LOG_FILE");
      if (log_file) {
        static std::ofstream log_stream(log_file, std::ios::app);
        if (log_stream) {
          std::ostringstream oss;
          oss << log_level_to_string(level) << ": [" << file << ":" << line << "] "
              << formatted_message << "\n";
          log_stream << oss.str();
        }
      }
    }
  }
}
}  // namespace detail
}  // namespace intelliperf

#define LOG_DETAIL(msg, ...)    \
  intelliperf::detail::log_message( \
      intelliperf::detail::LogLevel::DETAIL, __FILE__, __LINE__, msg, ##__VA_ARGS__)
#define LOG_INFO(msg, ...)      \
  intelliperf::detail::log_message( \
      intelliperf::detail::LogLevel::INFO, __FILE__, __LINE__, msg, ##__VA_ARGS__)
#define LOG_ERROR(msg, ...)     \
  intelliperf::detail::log_message( \
      intelliperf::detail::LogLevel::ERROR, __FILE__, __LINE__, msg, ##__VA_ARGS__)
