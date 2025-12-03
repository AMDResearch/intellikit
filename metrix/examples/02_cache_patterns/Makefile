HIP_PATH ?= $(shell dirname $(shell dirname $(shell which hipcc)))
HIPCC = $(shell which hipcc)

TARGET = cache_patterns
SOURCE = kernel.hip

CXXFLAGS = -O3

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(HIPCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run

