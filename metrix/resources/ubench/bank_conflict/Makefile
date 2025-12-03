HIPCC = hipcc
HIPCC_FLAGS = -O3

TARGETS = conflict no_conflict

all: $(TARGETS)

conflict: conflict.hip
	$(HIPCC) $(HIPCC_FLAGS) -o conflict conflict.hip

no_conflict: no_conflict.hip
	$(HIPCC) $(HIPCC_FLAGS) -o no_conflict no_conflict.hip

clean:
	rm -f $(TARGETS)

.PHONY: all clean

