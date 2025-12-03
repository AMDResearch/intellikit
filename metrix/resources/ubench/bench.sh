#!/bin/bash

# Function to handle errors without exiting the shell
handle_error() {
  echo "Error: $1"
}

# Check for input argument
if [ $# -ne 1 ]; then
  handle_error "Usage: $0 <hip_or_python_file>"
  exit
fi

# Extract filename without extension
FILE="$1"

EXTENSION="${FILE##*.}"

if [ "$EXTENSION" == "py" ]; then
  BASENAME=$(basename "$FILE" .py)
else
  BASENAME=$(basename "$FILE" .hip)
fi

WORKLOAD_NAME="${BASENAME}_profile"
WORKLOAD_PATH="workloads/${WORKLOAD_NAME}"
DEVICE=0
LOG_FILE="${BASENAME}_analysis.log"
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
PROFILER=$ROCM_PATH/bin/rocprof-compute
UNIT="ms"
if [ "$EXTENSION" == "hip" ]; then
  EXECUTABLE="${BASENAME}.out"
else
  EXECUTABLE="${BASENAME}.py"
fi

if ! command -v "$PROFILER" &> /dev/null; then
  echo "Error: $PROFILER is not installed or not in PATH."
  exit 1
fi


# Compile the HIP file
if [ "$EXTENSION" == "hip" ]; then
  echo "Compiling $FILE..."
  hipcc "$FILE" -o "$EXECUTABLE"
  if [ $? -ne 0 ]; then
    handle_error "Compilation failed for $HIP_FILE."
    exit 1
  fi

  echo "Compilation successful. Executable: $EXECUTABLE"
  
  # Extract kernel name from source file (look for __global__ functions)
  KERNEL_NAME=$(grep -oP '__global__\s+void\s+\K\w+' "$FILE" | head -n 1)
  if [ -n "$KERNEL_NAME" ]; then
    echo "Found kernel: $KERNEL_NAME"
    KERNEL_FILTER="-k $KERNEL_NAME"
  else
    KERNEL_FILTER=""
  fi
fi

# Profile the executable using Omniperf with a workload name
echo "Profiling $EXECUTABLE with Omniperf..."
rm -rf "$WORKLOAD_PATH"
$PROFILER profile --name "$WORKLOAD_NAME" --device $DEVICE $KERNEL_FILTER -- "./$EXECUTABLE"
if [ $? -ne 0 ]; then
  handle_error "Profiling failed for $EXECUTABLE."
  exit 1
fi

echo "Profiling completed successfully."

# Analyze the profiling results and log output
echo "Analyzing the profiling data..."

GPU_WORKLOAD_PATH=$(find "$WORKLOAD_PATH" -mindepth 1 -maxdepth 1 -type d | head -n 1)

$PROFILER analyze -p "$GPU_WORKLOAD_PATH" --time-unit $UNIT --save-dfs "$GPU_WORKLOAD_PATH/analyze" | tee "$LOG_FILE"
if [ $? -ne 0 ]; then
  handle_error "Analysis failed for workload at $WORKLOAD_PATH."
  exit 1
fi

echo "Analysis completed successfully. Results saved to $LOG_FILE."