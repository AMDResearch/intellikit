HIPCC = hipcc
HIPCC_FLAGS = -O3

TARGETS = high_hit low_hit

all: $(TARGETS)

high_hit: high_hit.hip
	$(HIPCC) $(HIPCC_FLAGS) -o high_hit high_hit.hip

low_hit: low_hit.hip
	$(HIPCC) $(HIPCC_FLAGS) -o low_hit low_hit.hip

clean:
	rm -f $(TARGETS)

.PHONY: all clean

