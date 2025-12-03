# Introduction

Welcome to the Guided Tuning Documentation!

This project, **Guided Tuning** (GT), is designed to provide a streamlined and efficient approach to optimize performance of HPC and ML apps. By leveraging a tuned set of high-level performance counters and guided workflows, this project aims to simplify the iterative nature of performance profiling. GT currently supports the following architectures:

- **AMD Instinct**: MI200, MI300

## Key Features

- **User-Friendly Interface**: Easy to use CLI tool with an emphasis on minimal user input.
- **Customizable Tuning**: Baked in configuration files to can be extended to fit additional metrics and future architechtures.
- **Performance Optimization**: Analysis mode suggests optimizations based on known performance bottlenecks.
- **Dynamic DB Backend**: Workload data is automatically backed up onto a locally hosted SQL DB that you own. Easily share performance data with colleagues.

## Purpose

The primary goal of this project is maximize the performance insights we can gather from a small set of high-level performance counters. We aim to provide a tool that can be used by both HPC and ML developers to quickly identify performance bottlenecks and suggest optimizations.

## Getting Started

GT's command line tool uses modes to help you quickly get started. The modes are designed to be intuitive and require minimal user input. The tool is built to be flexible and can be easily extended to support additional architectures and performance metrics.

```console
$ gt --help
usage: gt [mode] [options]

Command line interface guided tool

Modes:
  {profile,db,analyze}  Select GT mode:
    profile             Profile the target application and load into DB
    db                  Query the db
    analyze             Analyze database results

Help:
  -h, --help            show this help message and exit

General Options:
  -v, --verbose         Increase output verbosity (use multiple times for higher levels)
```

To begin, explore the documentation to understand the setup process, different modes, and best practices for using GT. Happy tuning!