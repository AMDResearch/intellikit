# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Makefile for metrix validation microbenchmarks

HIPCC ?= hipcc
CFLAGS = -O2 -fno-inline -std=c++17
KERNEL_DIR = kernels
BIN_DIR = bin

# Auto-detect GPU architecture if possible
GPU_ARCH ?= $(shell rocminfo 2>/dev/null | grep -m1 'Name:.*gfx' | awk '{print $$NF}' || echo "gfx942")
ARCH_FLAG = --offload-arch=$(GPU_ARCH)

SOURCES = $(wildcard $(KERNEL_DIR)/*.hip)
BINARIES = $(patsubst $(KERNEL_DIR)/%.hip,$(BIN_DIR)/%,$(SOURCES))

.PHONY: all clean

all: $(BIN_DIR) $(BINARIES)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/%: $(KERNEL_DIR)/%.hip $(KERNEL_DIR)/common.h | $(BIN_DIR)
	$(HIPCC) $(CFLAGS) $(ARCH_FLAG) -I$(KERNEL_DIR) $< -o $@

clean:
	rm -rf $(BIN_DIR)

# Individual targets for convenience
.PHONY: list
list:
	@echo "Benchmarks:"
	@for f in $(SOURCES); do echo "  $$(basename $$f .hip)"; done
