cmake_minimum_required(VERSION 3.21)
project(omniprobe_cmake_example LANGUAGES CXX HIP)

# A two-file HIP application, built the ordinary way. Nothing here mentions
# Omniprobe: instrumentation is added from the outside, at configure time:
#
#   cmake -S . -B build-omniprobe -DCMAKE_BUILD_TYPE=RelWithDebInfo \
#         -DCMAKE_HIP_FLAGS="$(omniprobe-ik flags memory)"
#
# Use a fresh build directory, so every file is recompiled with the plugin, and an
# optimised build type with debug info: at -O0 (CMake's default) Omniprobe sees no
# accesses, and without -g it cannot name source lines.
add_executable(app src/main.hip src/kernels.hip)
