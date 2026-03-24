cmake_minimum_required(VERSION 3.22)
project(kerncap VERSION 0.1.0 LANGUAGES CXX)

# Enable HIP language support
enable_language(HIP)

add_subdirectory(src)
