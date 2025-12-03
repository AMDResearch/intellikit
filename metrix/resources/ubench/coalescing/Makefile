HIPCC = hipcc
HIPCC_FLAGS = -O3

TARGETS = coalesced uncoalesced

all: $(TARGETS)

coalesced: coalesced.hip
	$(HIPCC) $(HIPCC_FLAGS) -o coalesced coalesced.hip

uncoalesced: uncoalesced.hip
	$(HIPCC) $(HIPCC_FLAGS) -o uncoalesced uncoalesced.hip

clean:
	rm -f $(TARGETS)

.PHONY: all clean

