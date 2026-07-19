"""End-to-end pipeline orchestrator.

Runs the full HAI medallion pipeline in order:
    Silver (build_silver) -> Gold (build_gold) -> verify (verify_gold)

Individual layers can still be run on their own via their own modules
(`python -m cps_anomaly_pipeline.silver`, `... .gold`).
"""

from __future__ import annotations

from cps_anomaly_pipeline.gold import build_gold, verify_gold
from cps_anomaly_pipeline.paths import PathConfig
from cps_anomaly_pipeline.silver import build_silver


def run_pipeline(paths: PathConfig | None = None) -> None:
    paths = paths or PathConfig()

    print("== Silver ==")
    build_silver(paths)

    print("== Gold ==")
    build_gold(paths)

    print("== Verify ==")
    verify_gold(paths)

    print("Pipeline complete.")


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
