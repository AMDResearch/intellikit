"""
List command implementation
"""

from ..metrics import METRIC_CATALOG, METRIC_PROFILES
from ..metrics.catalog import get_metrics_by_category


def list_command(args):
    """Execute list command"""

    if args.item_type == "metrics":
        list_metrics(args.category)
    elif args.item_type == "profiles":
        list_profiles()
    elif args.item_type == "counters":
        print("⚠️  Counter listing not yet implemented")
    elif args.item_type == "devices":
        list_devices()

    return 0


def list_metrics(category=None):
    """List available metrics"""

    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                     AVAILABLE METRICS                               ║")
    print("╚════════════════════════════════════════════════════════════════════╝\n")

    if category:
        metrics = get_metrics_by_category(category)
        print(f"Category: {category}\n")
    else:
        metrics = list(METRIC_CATALOG.keys())

    # Group by category
    from collections import defaultdict

    by_category = defaultdict(list)

    for metric_name in metrics:
        metric_def = METRIC_CATALOG[metric_name]
        cat = metric_def["category"].value
        by_category[cat].append((metric_name, metric_def))

    # Print grouped
    for cat, metrics_list in sorted(by_category.items()):
        print(f"─── {cat.upper().replace('_', ' ')} {'─' * (60 - len(cat))}")
        for metric_name, metric_def in metrics_list:
            print(f"  • {metric_name}")
            print(f"    {metric_def['description']}")
            print(f"    Unit: {metric_def['unit']}\n")

    print(f"Total: {len(metrics)} metrics")


def list_profiles():
    """List available profiles"""

    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                    AVAILABLE PROFILES                               ║")
    print("╚════════════════════════════════════════════════════════════════════╝\n")

    for profile_name, profile_def in METRIC_PROFILES.items():
        print(f"┌─ {profile_name.upper()} {'─' * (65 - len(profile_name))}")
        print(f"│  {profile_def['description']}")
        print(f"│  Metrics: {len(profile_def['metrics'])}")
        print(f"│  Estimated passes: {profile_def['estimated_passes']}")
        print(f"└{'─' * 68}\n")

    print("Usage: metrix profile --profile <name> ./app")


def list_devices():
    """List supported devices.

    Discovers backend modules on disk under ``metrix/backends/gfx*.py`` so
    the output stays in sync with what's actually installed (instead of a
    hardcoded list that drifts as new backends are added).
    """

    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║                   SUPPORTED DEVICES                                 ║")
    print("╚════════════════════════════════════════════════════════════════════╝\n")

    # Friendly metadata for known archs.  Backends not in this map are still
    # listed (with empty marketing name) so new backends are visible the
    # moment their gfx*.py file is added.
    _META = {
        "gfx90a": ("AMD Instinct MI200", "CDNA 2"),
        "gfx942": ("AMD Instinct MI300", "CDNA 3"),
        "gfx950": ("AMD Instinct MI355X", "CDNA 4"),
        "gfx1030": ("AMD Radeon RX 6000", "RDNA 2"),
        "gfx1100": ("AMD Radeon RX 7900", "RDNA 3"),
        "gfx1151": ("AMD Strix Halo APU", "RDNA 3.5"),
        "gfx1201": ("AMD Radeon RX 9070", "RDNA 4"),
    }

    from pathlib import Path

    import metrix

    backends_dir = Path(metrix.__file__).parent / "backends"
    archs = sorted(p.stem for p in backends_dir.glob("gfx*.py"))

    if not archs:
        print("  (no backend modules found under metrix/backends/)")
    else:
        for arch in archs:
            name, generation = _META.get(arch, ("", "unknown"))
            print(f"  • {arch:10s}  {name:30s}  ({generation})")

    print("\nUse --device <arch> to specify target architecture")
