HIPCC ?= hipcc
CXXFLAGS := -O2 -fno-inline -Wall

KERNEL_DIR := kernels
BIN_DIR := bin

TARGETS := bw_sequential_read bw_sequential_write bw_copy bw_strided_access \
           cache_l2_resident cache_l1_resident lds_bank_conflicts \
           compute_valu_fma compute_mfma atomic_contention mixed_compute_memory

BINARIES := $(addprefix $(BIN_DIR)/,$(TARGETS))

all: $(BIN_DIR) $(BINARIES)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/%: $(KERNEL_DIR)/%.hip $(KERNEL_DIR)/common.h | $(BIN_DIR)
	$(HIPCC) $(CXXFLAGS) -I$(KERNEL_DIR) $< -o $@

clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean
