"""PCIe Host<->Device bandwidth microbench.

Measures sustained throughput for cudaMemcpy in four modes:
  - pageable H2D, pageable D2H
  - pinned H2D, pinned D2H
  - async (pinned, with stream) H2D, async D2H

Runs across transfer sizes from 1 MB to 1 GB (configurable). Records
median bandwidth across N repetitions per size. Outputs:
  - reports/bench/pcie_bench.csv
  - reports/bench/pcie_bench.json (summary stats)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch


def _now() -> float:
    return time.perf_counter()


def _format_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:g}{unit}"
        nbytes //= 1024
    return f"{nbytes}B"


def _bench_one(
    size_bytes: int,
    direction: str,
    pinned: bool,
    use_stream: bool,
    repeats: int,
    device: torch.device,
) -> dict:
    """Run one configuration and return median + p95 bandwidth (GB/s)."""
    if direction not in {"h2d", "d2h"}:
        raise ValueError(direction)

    nfloats = size_bytes // 4
    host = torch.empty(nfloats, dtype=torch.float32, pin_memory=pinned)
    host.uniform_(-1.0, 1.0)
    dev = torch.empty(nfloats, dtype=torch.float32, device=device)

    stream = torch.cuda.Stream(device=device) if use_stream else None

    durations_s: list[float] = []
    for _ in range(repeats + 1):  # +1 warmup
        torch.cuda.synchronize(device)
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        if use_stream:
            with torch.cuda.stream(stream):
                start_evt.record(stream)
                if direction == "h2d":
                    dev.copy_(host, non_blocking=True)
                else:
                    host.copy_(dev, non_blocking=True)
                end_evt.record(stream)
            stream.synchronize()
        else:
            start_evt.record()
            if direction == "h2d":
                dev.copy_(host, non_blocking=False)
            else:
                host.copy_(dev, non_blocking=False)
            end_evt.record()
            torch.cuda.synchronize(device)

        ms = start_evt.elapsed_time(end_evt)
        durations_s.append(ms / 1000.0)

    durations_s = durations_s[1:]  # drop warmup
    bw_gbps = [size_bytes / d / 1e9 for d in durations_s]
    return {
        "median_GBps": statistics.median(bw_gbps),
        "p95_GBps": sorted(bw_gbps)[max(0, int(0.95 * len(bw_gbps)) - 1)],
        "min_GBps": min(bw_gbps),
        "max_GBps": max(bw_gbps),
        "samples": len(bw_gbps),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PCIe H<->D bandwidth bench")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--sizes-mb",
        nargs="+",
        type=int,
        default=[1, 4, 16, 64, 256, 512, 1024],
        help="Transfer sizes in MB",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/bench"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available", file=sys.stderr)
        return 2

    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    props = torch.cuda.get_device_properties(device)
    env = {
        "device": str(device),
        "name": props.name,
        "vram_total_GB": props.total_memory / 1e9,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "repeats": args.repeats,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print(f"# Device: {env['name']} ({env['vram_total_GB']:.1f} GB)")
    print(f"# Torch: {env['torch_version']} CUDA: {env['cuda_version']}")
    print()

    configs = [
        ("h2d", False, False, "pageable_h2d_sync"),
        ("d2h", False, False, "pageable_d2h_sync"),
        ("h2d", True, False, "pinned_h2d_sync"),
        ("d2h", True, False, "pinned_d2h_sync"),
        ("h2d", True, True, "pinned_h2d_async"),
        ("d2h", True, True, "pinned_d2h_async"),
    ]

    rows: list[dict] = []
    print(f"{'size':>8} {'mode':<22} {'median GB/s':>12} {'p95 GB/s':>10} {'min':>8} {'max':>8}")
    for size_mb in args.sizes_mb:
        size_bytes = size_mb * 1024 * 1024
        for direction, pinned, use_stream, label in configs:
            try:
                r = _bench_one(
                    size_bytes=size_bytes,
                    direction=direction,
                    pinned=pinned,
                    use_stream=use_stream,
                    repeats=args.repeats,
                    device=device,
                )
            except RuntimeError as exc:
                print(f"{_format_size(size_bytes):>8} {label:<22} FAILED: {exc}")
                continue
            row = {
                "size_MB": size_mb,
                "mode": label,
                **r,
            }
            rows.append(row)
            print(
                f"{_format_size(size_bytes):>8} {label:<22}"
                f" {r['median_GBps']:>12.2f} {r['p95_GBps']:>10.2f}"
                f" {r['min_GBps']:>8.2f} {r['max_GBps']:>8.2f}"
            )

    csv_path = args.out_dir / "pcie_bench.csv"
    with csv_path.open("w") as f:
        if rows:
            keys = list(rows[0].keys())
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(str(r[k]) for k in keys) + "\n")

    summary = {
        "env": env,
        "results": rows,
        "verdict_for_idea6": {
            "threshold_GBps": 15.0,
            "best_sustained_pinned_h2d_GBps": max(
                (r["median_GBps"] for r in rows if r["mode"].startswith("pinned_h2d")),
                default=0.0,
            ),
            "best_sustained_pinned_d2h_GBps": max(
                (r["median_GBps"] for r in rows if r["mode"].startswith("pinned_d2h")),
                default=0.0,
            ),
        },
    }
    verdict = summary["verdict_for_idea6"]
    summary["verdict_for_idea6"]["go_for_idea6"] = (
        verdict["best_sustained_pinned_h2d_GBps"] >= verdict["threshold_GBps"]
        and verdict["best_sustained_pinned_d2h_GBps"] >= verdict["threshold_GBps"]
    )

    json_path = args.out_dir / "pcie_bench.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"# Wrote {csv_path}")
    print(f"# Wrote {json_path}")
    print()
    v = summary["verdict_for_idea6"]
    go = "GO" if v["go_for_idea6"] else "NO-GO"
    print(
        f"# Idea 6 (pipelined staging) verdict: {go}"
        f" — best pinned H2D {v['best_sustained_pinned_h2d_GBps']:.1f} GB/s,"
        f" D2H {v['best_sustained_pinned_d2h_GBps']:.1f} GB/s"
        f" (threshold {v['threshold_GBps']:.0f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
