/**
 * kerncap_log.hpp — Minimal fprintf-based logging for the HSA capture tool.
 *
 * Thread-safe on POSIX (fprintf to stderr is atomic for reasonable messages).
 * Zero dependencies, zero static constructors — safe for LD_PRELOAD injection.
 */
#pragma once

#include <cstdarg>
#include <cstdio>

#define KERNCAP_INFO(fmt, ...)  fprintf(stderr, "[kerncap INFO] "  fmt "\n", ##__VA_ARGS__)
#define KERNCAP_WARN(fmt, ...)  fprintf(stderr, "[kerncap WARN] "  fmt "\n", ##__VA_ARGS__)
#define KERNCAP_ERROR(fmt, ...) fprintf(stderr, "[kerncap ERROR] " fmt "\n", ##__VA_ARGS__)
