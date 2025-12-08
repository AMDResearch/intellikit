"""
Common helpers for backend implementations.

These utilities are intentionally hardware-agnostic. Any architecture-specific
details (e.g., block limits, counter naming) should live in the gfxXXXX
backend modules and be passed into these helpers.
"""

from collections import defaultdict
from typing import Callable, Dict, List, Optional


def split_counters_into_passes(
    counters: List[str],
    *,
    block_limits: Optional[Dict[str, int]] = None,
    get_counter_block: Optional[Callable[[str], str]] = None,
    max_per_pass: int = 14,
    default_block_limit: int = 4,
    logger=None,
) -> List[List[str]]:
    """
    Split counters into multiple profiling passes based on per-block limits.

    This is a generic bin-packing helper used by architecture-specific
    backends. Hardware-specific information (block names and limits) must
    be provided by the caller.

    Args:
        counters:
            List of hardware counter names to collect.
        block_limits:
            Optional mapping block_name -> max_counters_per_pass.
            If omitted or empty, falls back to simple chunking by `max_per_pass`.
        get_counter_block:
            Optional function mapping counter_name -> block_name.
            Required when `block_limits` is provided.
        max_per_pass:
            Maximum counters per pass for the simple chunking fallback.
        default_block_limit:
            Fallback per-block limit used when a block name is missing
            from `block_limits`.
        logger:
            Optional logger with .info / .debug methods.

    Returns:
        List of counter lists, one per profiling pass.
    """
    # Handle empty counters (timing-only mode) - return single pass with no counters
    if not counters:
        return [[]]

    # If no block limits defined, fall back to simple chunking
    if not block_limits:
        if len(counters) <= max_per_pass:
            return [counters]

        passes: List[List[str]] = []
        for i in range(0, len(counters), max_per_pass):
            passes.append(counters[i : i + max_per_pass])

        if logger is not None:
            logger.info(
                f"Splitting {len(counters)} counters into {len(passes)} simple passes"
            )
        return passes

    if get_counter_block is None:
        raise ValueError(
            "get_counter_block must be provided when block_limits are specified"
        )

    # Organize counters by hardware block
    counters_by_block: Dict[str, List[str]] = defaultdict(list)
    for counter in counters:
        block = get_counter_block(counter)
        counters_by_block[block].append(counter)

    if logger is not None:
        logger.debug(f"Counters by block: {dict(counters_by_block)}")

    # Greedy bin-packing algorithm:
    # For each pass, take as many counters from each block as the limit allows
    passes: List[List[str]] = []
    remaining: Dict[str, List[str]] = {
        block: list(cntrs) for block, cntrs in counters_by_block.items()
    }

    while any(remaining.values()):
        current_pass: List[str] = []
        pass_block_count: Dict[str, int] = defaultdict(int)

        # Try to add counters from each block to current pass
        for block_name in sorted(remaining.keys()):  # Sort for deterministic ordering
            block_counters = remaining[block_name]
            if not block_counters:
                continue

            # Get limit for this block (default to default_block_limit if unknown)
            limit = block_limits.get(block_name, default_block_limit)
            available_slots = limit - pass_block_count[block_name]
            if available_slots <= 0:
                continue

            # Add as many counters from this block as possible
            to_add = block_counters[:available_slots]
            current_pass.extend(to_add)
            pass_block_count[block_name] += len(to_add)

            # Update remaining counters for this block
            remaining[block_name] = block_counters[available_slots:]

        if current_pass:
            passes.append(current_pass)
            if logger is not None:
                logger.debug(
                    f"Pass {len(passes)}: {len(current_pass)} counters, "
                    f"blocks: {dict(pass_block_count)}"
                )

        # Remove blocks with no remaining counters
        remaining = {k: v for k, v in remaining.items() if v}

    if logger is not None:
        logger.info(
            f"Packed {len(counters)} counters into {len(passes)} block-aware passes"
        )

    return passes

