#!/usr/bin/env python3
"""Display dashboard configuration (data structure only, no actual UI)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tracking.dashboard import DashboardConfig


def main():
    dashboard = DashboardConfig()
    panels = dashboard.get_panels()

    print("Dashboard Panels:")
    print("-" * 60)
    for panel in panels:
        print(f"  [{panel.panel_type:12s}] {panel.name:25s} <- {panel.data_source}")
    print(f"\nTotal panels: {len(panels)}")


if __name__ == "__main__":
    main()
