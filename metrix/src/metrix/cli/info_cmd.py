"""
Info command implementation
"""

from ..metrics import METRIC_CATALOG, METRIC_PROFILES
from ..backends import get_backend, detect_or_default


def info_command(args):
    """Execute info command"""

    if args.info_type == "metric":
        # Get architecture from args if available, otherwise auto-detect
        arch = getattr(args, 'arch', None) or detect_or_default()
        show_metric_info(args.name, arch)
    elif args.info_type == "profile":
        show_profile_info(args.name)
    elif args.info_type == "counter":
        print("⚠️  Counter info not yet implemented")

    return 0


def show_metric_info(metric_name, arch="gfx942"):
    """Show detailed metric information"""

    if metric_name not in METRIC_CATALOG:
        print(f"❌ Unknown metric: {metric_name}")
        print(f"\nRun 'metrix list metrics' to see available metrics")
        return 1

    metric_def = METRIC_CATALOG[metric_name]

    print("╔════════════════════════════════════════════════════════════════════╗")
    print(f"║  {metric_def['name']:66s} ║")
    print("╚════════════════════════════════════════════════════════════════════╝\n")

    print(f"Name:        {metric_name}")
    print(f"Description: {metric_def['description']}")
    print(f"Unit:        {metric_def['unit']}")
    print(f"Category:    {metric_def['category'].value}")

    # Show actual hardware counters from the backend (architecture-specific)
    print(f"\nRequired Hardware Counters ({arch}):")
    try:
        backend = get_backend(arch)
        actual_counters = backend.get_metric_counters(metric_name)
        for counter in actual_counters:
            print(f"  • {counter}")
    except ValueError as e:
        # Metric not implemented in this backend
        print(f"  ⚠️  Metric not implemented for {arch}")
        # Fall back to catalog's derived_from as documentation
        if 'derived_from' in metric_def:
            print(f"\n  Conceptual counters (from catalog):")
            for counter in metric_def['derived_from']:
                print(f"    • {counter}")

    if 'interpretation' in metric_def:
        print(f"\nInterpretation Guide:")
        for level, interpretation_data in metric_def['interpretation'].items():
            if isinstance(interpretation_data, tuple) and len(interpretation_data) == 3:
                low, high, desc = interpretation_data
                print(f"  • {level.upper():10s}: {low}-{high}% - {desc}")
            elif isinstance(interpretation_data, tuple) and len(interpretation_data) == 2:
                range_info, desc = interpretation_data
                if isinstance(range_info, tuple):
                    print(f"  • {level.upper():10s}: {range_info[0]}-{range_info[1]}% - {desc}")
                else:
                    print(f"  • {level.upper():10s}: {range_info} - {desc}")
            else:
                print(f"  • {level.upper():10s}: {interpretation_data}")

    if metric_def.get('device_specific'):
        print(f"\n⚠️  Note: This metric requires device-specific constants")

    if metric_def.get('requires_kernel_info'):
        print(f"\n⚠️  Note: This metric requires kernel metadata")


def show_profile_info(profile_name):
    """Show detailed profile information"""

    if profile_name not in METRIC_PROFILES:
        print(f"❌ Unknown profile: {profile_name}")
        print(f"\nRun 'metrix list profiles' to see available profiles")
        return 1

    profile_def = METRIC_PROFILES[profile_name]

    print("╔════════════════════════════════════════════════════════════════════╗")
    print(f"║  Profile: {profile_name.upper():57s} ║")
    print("╚════════════════════════════════════════════════════════════════════╝\n")

    print(f"Description: {profile_def['description']}")
    print(f"Estimated collection passes: {profile_def['estimated_passes']}")

    if 'focus' in profile_def:
        print(f"Focus area: {profile_def['focus']}")

    print(f"\nIncluded Metrics ({len(profile_def['metrics'])}):")
    for i, metric in enumerate(profile_def['metrics'], 1):
        metric_def = METRIC_CATALOG[metric]
        print(f"  {i:2d}. {metric}")
        print(f"      {metric_def['description']}")

    if 'typical_bottlenecks' in profile_def:
        print(f"\nTypical Bottlenecks Detected:")
        for bottleneck in profile_def['typical_bottlenecks']:
            print(f"  • {bottleneck.replace('_', ' ').title()}")

    print(f"\n💡 Usage:")
    print(f"   metrix profile --profile {profile_name} ./my_app")

