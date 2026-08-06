#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine stage-1/2 pixel errors and stage-3 MPJPE across split experiments."
    )
    parser.add_argument(
        "--experiment",
        action="append",
        nargs=3,
        metavar=("ID", "DISPLAY_NAME", "METRICS_JSON"),
        required=True,
        help="Repeat once per experiment, in desired report column order.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def format_value(value: object, digits: int = 3) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments = []
    for experiment_id, display_name, metrics_path in args.experiment:
        source = Path(metrics_path).expanduser().resolve()
        metrics = json.loads(source.read_text(encoding="utf-8"))
        experiments.append(
            {
                "id": experiment_id,
                "name": display_name,
                "source": str(source),
                "metrics": metrics,
            }
        )

    overall_rows: list[dict[str, object]] = []
    for experiment in experiments:
        metrics = experiment["metrics"]
        overall_rows.append(
            {
                "experiment_id": experiment["id"],
                "experiment": experiment["name"],
                "test_frames_2d": metrics["test_frames_2d"],
                "test_frames_3d_valid": metrics["test_frames_3d_valid"],
                "stage1_best_epoch": metrics["stage1"]["checkpoint_epoch"],
                "stage1_mean_px": metrics["stage1"]["mean_visible_joint_pixel_error"],
                "stage2_best_epoch": metrics["stage2"]["checkpoint_epoch"],
                "stage2_mean_px": metrics["stage2"]["mean_visible_joint_pixel_error"],
                "stage3_best_epoch": metrics["stage3"]["checkpoint_epoch"],
                "stage3_mpjpe_mm": metrics["stage3"]["mpjpe_mm"],
            }
        )
    overall_fields = list(overall_rows[0])
    write_csv(output_dir / "overall_metrics.csv", overall_fields, overall_rows)

    joint_names = list(
        experiments[0]["metrics"]["stage3"]["per_joint_mpjpe_mm"].keys()
    )
    stage_specs = (
        ("stage1", "per_joint_visible_pixel_error", "stage1_per_joint_px.csv"),
        ("stage2", "per_joint_visible_pixel_error", "stage2_per_joint_px.csv"),
        ("stage3", "per_joint_mpjpe_mm", "stage3_per_joint_mpjpe_mm.csv"),
    )
    joint_tables: dict[str, list[dict[str, object]]] = {}
    for stage, metric_key, filename in stage_specs:
        rows = []
        for joint_name in joint_names:
            row: dict[str, object] = {"joint": joint_name}
            for experiment in experiments:
                row[experiment["id"]] = experiment["metrics"][stage][metric_key][
                    joint_name
                ]
            rows.append(row)
        fields = ["joint", *[experiment["id"] for experiment in experiments]]
        write_csv(output_dir / filename, fields, rows)
        joint_tables[stage] = rows

    names = [experiment["name"] for experiment in experiments]
    report = [
        "# 五种数据划分的三阶段测试性能对比",
        "",
        "所有数值均由各实验自己的留出集合独立推理得到。初版的“尾部 20%”曾参与模型选择，"
        "因此属于 held-out validation；其余实验使用固定 80/10/10 的独立 test。"
        "不同实验测试帧数不同，尤其 stride 90 的三维有效测试帧较少，横向比较时需结合样本数。",
        "",
        "## 整体性能",
        "",
        markdown_table(
            [
                "实验",
                "2D测试帧",
                "3D有效帧",
                "S1 best",
                "S1 px↓",
                "S2 best",
                "S2 px↓",
                "S3 best",
                "S3 MPJPE mm↓",
            ],
            [
                [
                    row["experiment"],
                    row["test_frames_2d"],
                    row["test_frames_3d_valid"],
                    row["stage1_best_epoch"],
                    format_value(row["stage1_mean_px"]),
                    row["stage2_best_epoch"],
                    format_value(row["stage2_mean_px"]),
                    row["stage3_best_epoch"],
                    format_value(row["stage3_mpjpe_mm"]),
                ]
                for row in overall_rows
            ],
        ),
    ]
    for stage, title, unit in (
        ("stage1", "阶段一逐关节误差", "px"),
        ("stage2", "阶段二逐关节误差", "px"),
        ("stage3", "阶段三逐关节 MPJPE", "mm"),
    ):
        report.extend(
            [
                "",
                f"## {title}（{unit}，越低越好）",
                "",
                markdown_table(
                    ["关节", *names],
                    [
                        [
                            row["joint"],
                            *[
                                format_value(row[experiment["id"]])
                                for experiment in experiments
                            ],
                        ]
                        for row in joint_tables[stage]
                    ],
                ),
            ]
        )
    report.extend(
        [
            "",
            "## 指标来源",
            "",
            *[
                f"- {experiment['name']}：`{experiment['source']}`"
                for experiment in experiments
            ],
            "",
        ]
    )
    (output_dir / "comparison_report.md").write_text(
        "\n".join(report), encoding="utf-8-sig"
    )
    manifest = {
        "experiments": experiments,
        "outputs": {
            "report": str(output_dir / "comparison_report.md"),
            "overall_csv": str(output_dir / "overall_metrics.csv"),
            "stage1_per_joint_csv": str(output_dir / "stage1_per_joint_px.csv"),
            "stage2_per_joint_csv": str(output_dir / "stage2_per_joint_px.csv"),
            "stage3_per_joint_csv": str(output_dir / "stage3_per_joint_mpjpe_mm.csv"),
        },
    }
    (output_dir / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest["outputs"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
