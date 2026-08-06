#!/usr/bin/env python3
"""Inline generated skeleton JSON into the interactive visualization fragment."""

from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA = (
    HERE
    / "validation_0722_2_camc_hybrid_3d_comparison"
    / "data.json"
)
TEMPLATE = HERE / "skeleton_3d_comparison.template.html"
OUTPUT = Path(
    r"C:\Users\hand\.codex\visualizations\2026\07\22"
    r"\019f87f7-6961-7d51-8a57-2309e6505108\skeleton-3d-comparison.html"
)


def main() -> None:
    source = TEMPLATE.read_text(encoding="utf-8")
    payload = DATA.read_text(encoding="utf-8")
    if source.count("__DATA__") != 1:
        raise RuntimeError("Visualization template must contain exactly one data placeholder")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(source.replace("__DATA__", payload), encoding="utf-8")
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
