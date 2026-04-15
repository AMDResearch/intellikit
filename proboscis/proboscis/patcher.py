# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
ELF surgery for injecting hidden probe arguments into AMDGPU code objects.

Ported from omniprobe's tools/codeobj/ pipeline:
  - common.py         → ELF parsing, naming conventions
  - plan_hidden_abi.py → compute hidden arg offsets
  - emit_hidden_abi_metadata.py → rewrite MessagePack metadata
  - rewrite_metadata_note.py → in-place .note section rewriting

The pipeline:
  1. Parse the code object ELF
  2. Plan hidden ABI changes (compute where to insert the probe context pointer)
  3. Rewrite the AMDGPU metadata note with the new hidden argument
  4. Output a patched code object with the expanded kernarg layout
"""

from __future__ import annotations

import struct
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .msgpack_codec import packb, unpackb

# ─── Constants ────────────────────────────────────────────────────────────────

PROBOSCIS_PREFIX = "__proboscis_"
PROBOSCIS_HIDDEN_ARG = "hidden_proboscis_ctx"

ELF_MAGIC = b"\x7fELF"
ELF64_HEADER_FORMAT = "<16sHHIQQQIHHHHHH"
ELF64_SECTION_HEADER_FORMAT = "<IIQQQQIIQQ"
ELF64_PROGRAM_HEADER_FORMAT = "<IIQQQQQQ"
NOTE_HEADER_FORMAT = "<III"

SHT_NOTE = 7
SHF_ALLOC = 0x2
PT_NOTE = 4


# ─── ELF Parsing ─────────────────────────────────────────────────────────────

def align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


def read_c_string(blob: bytes, start: int) -> str:
    end = blob.find(b"\x00", start)
    if end == -1:
        end = len(blob)
    return blob[start:end].decode("utf-8")


def load_elf_sections(
    data: bytearray,
) -> Tuple[List[dict], Dict[str, dict]]:
    """Parse ELF64 section headers. Returns (sections_list, section_map_by_name)."""
    if not data[:4] == ELF_MAGIC:
        raise ValueError("Not an ELF file")

    header = struct.unpack_from(ELF64_HEADER_FORMAT, data, 0)
    section_offset = header[6]
    section_entry_size = header[11]
    section_count = header[12]
    shstr_index = header[13]

    sections: List[dict] = []
    for index in range(section_count):
        offset = section_offset + index * section_entry_size
        fields = struct.unpack_from(ELF64_SECTION_HEADER_FORMAT, data, offset)
        sections.append({
            "index": index,
            "header_offset": offset,
            "name_offset": fields[0],
            "type": fields[1],
            "flags": fields[2],
            "addr": fields[3],
            "offset": fields[4],
            "size": fields[5],
            "link": fields[6],
            "info": fields[7],
            "addralign": fields[8],
            "entsize": fields[9],
        })

    shstr = sections[shstr_index]
    shstr_data = data[shstr["offset"]: shstr["offset"] + shstr["size"]]
    section_map: Dict[str, dict] = {}
    for section in sections:
        name = read_c_string(shstr_data, section["name_offset"]) if shstr_data else ""
        section["name"] = name
        section_map[name] = section

    return sections, section_map


def load_program_headers(data: bytearray) -> List[dict]:
    header = list(struct.unpack_from(ELF64_HEADER_FORMAT, data, 0))
    ph_offset = header[5]
    ph_entry_size = header[9]
    ph_count = header[10]
    programs: List[dict] = []
    for index in range(ph_count):
        offset = ph_offset + index * ph_entry_size
        fields = struct.unpack_from(ELF64_PROGRAM_HEADER_FORMAT, data, offset)
        programs.append({
            "index": index,
            "header_offset": offset,
            "type": fields[0],
            "flags": fields[1],
            "offset": fields[2],
            "vaddr": fields[3],
            "paddr": fields[4],
            "filesz": fields[5],
            "memsz": fields[6],
            "align": fields[7],
        })
    return programs


# ─── AMDGPU Metadata Note ────────────────────────────────────────────────────

def find_amdgpu_note(section_bytes: bytes) -> Tuple[int, int, int, bytes, bytes]:
    """Find the NT_AMDGPU_METADATA note in a .note section."""
    cursor = 0
    header_size = struct.calcsize(NOTE_HEADER_FORMAT)
    while cursor + header_size <= len(section_bytes):
        namesz, descsz, note_type = struct.unpack_from(NOTE_HEADER_FORMAT, section_bytes, cursor)
        cursor += header_size
        name_start = cursor
        name_end = name_start + namesz
        name_bytes = section_bytes[name_start:name_end]
        cursor = align_up(name_end, 4)
        desc_start = cursor
        desc_end = desc_start + descsz
        desc_bytes = section_bytes[desc_start:desc_end]
        cursor = align_up(desc_end, 4)

        owner = name_bytes.rstrip(b"\x00").decode("utf-8", errors="ignore")
        if owner == "AMDGPU":
            return (
                name_start - header_size,
                cursor,
                note_type,
                name_bytes,
                desc_bytes,
            )
    raise ValueError("NT_AMDGPU_METADATA note not found")


def rebuild_note(note_type: int, name_bytes: bytes, metadata_bytes: bytes) -> bytes:
    """Reconstruct a note entry with new metadata payload."""
    header = struct.pack(NOTE_HEADER_FORMAT, len(name_bytes), len(metadata_bytes), note_type)
    payload = bytearray()
    payload.extend(header)
    payload.extend(name_bytes)
    payload.extend(b"\x00" * (align_up(len(name_bytes), 4) - len(name_bytes)))
    payload.extend(metadata_bytes)
    payload.extend(b"\x00" * (align_up(len(metadata_bytes), 4) - len(metadata_bytes)))
    return bytes(payload)


def parse_amdgpu_metadata(desc_bytes: bytes) -> dict:
    """Parse AMDGPU metadata from the note descriptor (MessagePack format)."""
    return unpackb(desc_bytes)


# ─── Hidden ABI Planning ─────────────────────────────────────────────────────

@dataclass
class HiddenAbiPlan:
    """Plan for adding the hidden probe context pointer to a kernel."""
    kernel_name: str
    symbol: Optional[str]
    source_kernarg_size: int
    explicit_args_length: int
    insertion_offset: int
    new_kernarg_size: int
    pointer_size: int = 8


def compute_explicit_args_length(args: List[dict]) -> int:
    """Compute the byte range covered by explicit (non-hidden) arguments."""
    explicit_end = 0
    for arg in args:
        value_kind = str(arg.get(".value_kind", arg.get("value_kind", "")))
        if value_kind.startswith("hidden_") or value_kind == PROBOSCIS_HIDDEN_ARG:
            continue
        offset = int(arg.get(".offset", arg.get("offset", 0)))
        size = int(arg.get(".size", arg.get("size", 0)))
        explicit_end = max(explicit_end, offset + size)
    return explicit_end


def plan_hidden_abi(
    kernel_obj: dict,
    pointer_size: int = 8,
    alignment: int = 8,
) -> HiddenAbiPlan:
    """
    Compute where to insert the hidden proboscis context pointer.

    The pointer goes after all existing arguments (explicit + hidden),
    aligned to the specified boundary.
    """
    args = kernel_obj.get(".args", kernel_obj.get("args", []))
    kernarg_size = int(
        kernel_obj.get(".kernarg_segment_size", kernel_obj.get("kernarg_segment_size", 0)) or 0
    )
    explicit_length = compute_explicit_args_length(args)

    # Find the end of all existing args (including hidden ones)
    all_args_end = 0
    for arg in args:
        offset = int(arg.get(".offset", arg.get("offset", 0)))
        size = int(arg.get(".size", arg.get("size", 0)))
        all_args_end = max(all_args_end, offset + size)

    insertion_offset = align_up(max(kernarg_size, all_args_end), alignment)
    new_kernarg_size = insertion_offset + pointer_size

    kernel_name = kernel_obj.get(".name", kernel_obj.get("name", ""))
    symbol = kernel_obj.get(".symbol", kernel_obj.get("symbol"))

    return HiddenAbiPlan(
        kernel_name=kernel_name,
        symbol=symbol,
        source_kernarg_size=kernarg_size,
        explicit_args_length=explicit_length,
        insertion_offset=insertion_offset,
        new_kernarg_size=new_kernarg_size,
        pointer_size=pointer_size,
    )


# ─── Metadata Mutation ───────────────────────────────────────────────────────

def mutate_kernel_metadata(kernel_obj: dict, plan: HiddenAbiPlan) -> dict:
    """
    Add the hidden_proboscis_ctx argument to a kernel's metadata entry.

    Modifies kernarg_segment_size and appends the hidden arg to .args.
    Does not rename the kernel — this is an in-place mutation.
    """
    mutated = deepcopy(kernel_obj)

    # Use dotted keys if the source uses them
    uses_dots = ".kernarg_segment_size" in kernel_obj
    kss_key = ".kernarg_segment_size" if uses_dots else "kernarg_segment_size"
    args_key = ".args" if uses_dots else "args"
    offset_key = ".offset" if uses_dots else "offset"
    size_key = ".size" if uses_dots else "size"
    name_key = ".name" if uses_dots else "name"
    vk_key = ".value_kind" if uses_dots else "value_kind"

    mutated[kss_key] = plan.new_kernarg_size

    args = mutated.get(args_key, [])
    # Don't add if already present
    if not any(
        arg.get(name_key) == PROBOSCIS_HIDDEN_ARG
        or arg.get(vk_key) == PROBOSCIS_HIDDEN_ARG
        for arg in args
    ):
        args.append({
            name_key: PROBOSCIS_HIDDEN_ARG,
            offset_key: plan.insertion_offset,
            size_key: plan.pointer_size,
            vk_key: "global_buffer",
        })
        args.sort(key=lambda a: int(a.get(offset_key, 0)))
    mutated[args_key] = args

    return mutated


# ─── Code Object Patcher ─────────────────────────────────────────────────────

@dataclass
class PatchConfig:
    """Configuration for patching a code object."""
    target_kernels: Optional[List[str]] = None  # None = patch all
    pointer_size: int = 8
    alignment: int = 8


def patch_code_object(
    data: bytes,
    config: PatchConfig,
) -> Tuple[bytes, List[HiddenAbiPlan]]:
    """
    Patch an AMDGPU code object to add hidden probe context pointers.

    Steps:
    1. Parse ELF and find .note section with AMDGPU metadata
    2. Decode MessagePack metadata
    3. For each target kernel, compute insertion offset and mutate metadata
    4. Re-encode metadata and rewrite the .note section

    Returns:
        (patched_bytes, list_of_plans) — the patched ELF and plans for each kernel
    """
    elf_data = bytearray(data)
    sections, section_map = load_elf_sections(elf_data)

    note_section = section_map.get(".note")
    if not note_section or note_section["type"] != SHT_NOTE:
        raise ValueError(".note section not found in code object")

    section_offset = note_section["offset"]
    section_size = note_section["size"]
    section_bytes = bytes(elf_data[section_offset: section_offset + section_size])

    note_start, note_end, note_type, name_bytes, desc_bytes = find_amdgpu_note(section_bytes)

    # Parse metadata (MessagePack)
    metadata = parse_amdgpu_metadata(desc_bytes)
    kernels = metadata.get("amdhsa.kernels", [])

    plans: List[HiddenAbiPlan] = []
    new_kernels = []

    for kernel_obj in kernels:
        kernel_name = kernel_obj.get(".name", "")
        should_patch = (
            config.target_kernels is None
            or any(t in kernel_name for t in config.target_kernels)
        )

        if should_patch:
            plan = plan_hidden_abi(kernel_obj, config.pointer_size, config.alignment)
            plans.append(plan)
            new_kernels.append(mutate_kernel_metadata(kernel_obj, plan))
        else:
            new_kernels.append(deepcopy(kernel_obj))

    metadata["amdhsa.kernels"] = new_kernels

    # Re-encode metadata
    new_metadata_bytes = packb(metadata)

    # Rebuild note
    new_note = rebuild_note(note_type, name_bytes, new_metadata_bytes)
    replacement_section = section_bytes[:note_start] + new_note + section_bytes[note_end:]

    if len(replacement_section) <= section_size:
        # Fits in place — pad with zeros
        padded = bytearray(replacement_section)
        padded.extend(b"\x00" * (section_size - len(padded)))
        elf_data[section_offset: section_offset + section_size] = padded
    else:
        # Need to grow — use the full rewrite logic
        elf_data = _grow_note_section(
            elf_data, sections, section_map, replacement_section, note_section
        )

    return bytes(elf_data), plans


def _grow_note_section(
    data: bytearray,
    sections: List[dict],
    section_map: Dict[str, dict],
    replacement: bytes,
    note_section: dict,
) -> bytearray:
    """
    Grow the .note section to accommodate larger metadata.

    Shifts all subsequent sections and updates all ELF headers, program headers,
    and symbol tables accordingly.
    """
    section_offset = note_section["offset"]
    section_size = note_section["size"]
    growth = len(replacement) - section_size

    # Compute shift amount aligned to max section alignment
    insert_offset = section_offset + section_size
    max_align = max(
        (int(s.get("addralign", 1) or 1) for s in sections if s["offset"] >= insert_offset),
        default=1,
    )
    shift = align_up(growth, max_align)

    # Insert zero bytes
    data[insert_offset:insert_offset] = b"\x00" * shift

    # Write new note content
    data[section_offset: section_offset + len(replacement)] = replacement

    # Update ELF header
    header = list(struct.unpack_from(ELF64_HEADER_FORMAT, data, 0))
    if header[5] >= insert_offset:  # e_phoff
        header[5] += shift
    if header[6] >= insert_offset:  # e_shoff
        header[6] += shift
    struct.pack_into(ELF64_HEADER_FORMAT, data, 0, *header)

    # Update program headers
    programs = load_program_headers(data)
    note_addr = note_section.get("addr", 0)
    for prog in programs:
        file_end = prog["offset"] + prog["filesz"]
        if prog["header_offset"] >= insert_offset:
            prog["header_offset"] += shift
        if prog["type"] == PT_NOTE and prog["offset"] <= section_offset < file_end:
            prog["filesz"] += growth
            prog["memsz"] += growth
        elif prog["offset"] > insert_offset:
            prog["offset"] += shift
            prog["vaddr"] += shift
            prog["paddr"] += shift
        elif prog["offset"] <= insert_offset < file_end:
            prog["filesz"] += shift
            prog["memsz"] += shift
        struct.pack_into(
            ELF64_PROGRAM_HEADER_FORMAT, data, prog["header_offset"],
            prog["type"], prog["flags"], prog["offset"],
            prog["vaddr"], prog["paddr"], prog["filesz"],
            prog["memsz"], prog["align"],
        )

    # Update section headers
    for section in sections:
        if section["header_offset"] >= insert_offset:
            section["header_offset"] += shift
        if section["index"] == note_section["index"]:
            section["size"] = len(replacement)
        elif section["offset"] >= insert_offset:
            section["offset"] += shift
        struct.pack_into(
            ELF64_SECTION_HEADER_FORMAT, data, section["header_offset"],
            section["name_offset"], section["type"], section["flags"],
            section["addr"], section["offset"], section["size"],
            section["link"], section["info"], section["addralign"],
            section["entsize"],
        )

    return data


# ─── CLI Entry Point ────────────────────────────────────────────────────────
# Called by the C++ interceptor via subprocess:
#   python3 -m proboscis.patcher <code_object_path> <plan_output_path> [--target kernel]
#
# Patches the code object in-place and writes a JSON plan to plan_output_path.

def main():
    import json
    import sys

    args = sys.argv[1:]
    plan_only = "--plan-only" in args
    if plan_only:
        args.remove("--plan-only")

    if len(args) < 2:
        print("Usage: python3 -m proboscis.patcher [--plan-only] <code_object> <plan_output> [--target kernel]",
              file=sys.stderr)
        sys.exit(1)

    co_path = args[0]
    plan_path = args[1]
    target_kernel = None
    if "--target" in args:
        idx = args.index("--target")
        if idx + 1 < len(args):
            target_kernel = args[idx + 1]

    try:
        data = Path(co_path).read_bytes()
    except Exception as e:
        print(f"Failed to read code object: {e}", file=sys.stderr)
        sys.exit(1)

    target_kernels = [target_kernel] if target_kernel else None
    config = PatchConfig(target_kernels=target_kernels)

    try:
        patched_data, plans = patch_code_object(data, config)
    except Exception as e:
        print(f"Patching failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Write patched code object back (unless plan-only)
    if not plan_only:
        Path(co_path).write_bytes(patched_data)

    # Write plan JSON
    plan_json = []
    for plan in plans:
        plan_json.append({
            "kernel": plan.kernel_name,
            "symbol": plan.symbol or "",
            "orig_size": plan.source_kernarg_size,
            "new_size": plan.new_kernarg_size,
            "probe_ctx_offset": plan.insertion_offset,
            "explicit_len": plan.explicit_args_length,
        })

    Path(plan_path).write_text(json.dumps(plan_json, indent=2))
    mode = "Planned" if plan_only else "Patched"
    print(f"{mode} {len(plans)} kernel(s)", file=sys.stderr)


if __name__ == "__main__":
    main()
