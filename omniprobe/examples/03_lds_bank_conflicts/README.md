# LDS bank conflicts

`analyze_lds.py` builds two versions of a kernel that stages data through an LDS
tile and reads it column-wise (the access pattern of a matrix transpose):

1. Writes the kernel to a temporary directory.
2. Compiles it with `hipcc -O2 -g` plus `instrumentation_flags("memory")`.
3. Runs `analyze_memory` and prints the ranked LDS sites from `summarize_memory`.

Expected on gfx90a: the 32-float-wide tile reports **62 extra cycles per wave
execution** at the column-read line (all 32 lanes of each half-wave in one bank:
31 extra cycles per half), and the tile padded to 33 floats reports none.

```bash
python analyze_lds.py
```

Requires the same setup as the other examples (see [`../README.md`](../README.md)).
