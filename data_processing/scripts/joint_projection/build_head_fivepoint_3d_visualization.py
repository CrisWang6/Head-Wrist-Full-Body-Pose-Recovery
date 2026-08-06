#!/usr/bin/env python3
"""Inline five-point 3D comparison data into the Codex visualization fragment."""

import argparse
from pathlib import Path


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    template=Path(__file__).with_name("head_fivepoint_3d_comparison.template.html").read_text(encoding="utf-8")
    if template.count("__DATA__") != 1: raise RuntimeError("Expected exactly one data placeholder")
    fragment=template.replace("__DATA__",a.data.read_text(encoding="utf-8"))
    # The template is also usable as an inline Codex visualization fragment.
    # Wrap it when exporting a local file so it has its own document, color
    # variables and UTF-8 declaration instead of depending on the host page.
    if "<!doctype html>" not in fragment.lower():
        fragment=("<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
                  "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
                  "<title>头部五点约束 3D 骨架对比</title><style>:root{--foreground:#e8eef7;"
                  "--muted-foreground:#9aa8bb;--border:#314158;--background:#09111d}"
                  "html,body{margin:0;min-height:100%;background:var(--background);color:var(--foreground)}"
                  "body{padding:16px;box-sizing:border-box}.btn{background:#18263a;color:#e8eef7;"
                  "border:1px solid var(--border);border-radius:6px;padding:6px 12px;cursor:pointer}"
                  ".form-range{accent-color:#ffd54a}</style></head><body>"+fragment+"</body></html>")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(fragment,encoding="utf-8")
    print(a.output,a.output.stat().st_size)


if __name__ == "__main__": main()
