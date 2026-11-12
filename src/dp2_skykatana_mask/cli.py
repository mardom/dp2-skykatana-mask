"""Command-line interface (`dp2-mask`) for RSP and local workflows."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

from .butler_io import (
    discover_monster_dataset,
    find_rsp_stack_loader,
    list_consolidated_maps,
    open_dp2_butler,
)
from .config import (
    BandCriteria,
    MaskPipelineConfig,
    MonsterStarConfig,
    OutputConfig,
    PropertyMapsConfig,
    load_config,
)
from .memory import detect_effective_memory_limit_gib
from .pipeline import DP2MaskBuilder
from .synthetic import SyntheticDP2Butler


def _configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dp2-mask",
        description=(
            "Build memory-safe (32 GB RSP) Vera C. Rubin DP2 spatial footprints & "
            "Monster bright-star masks with SkyKatana."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. build
    p_build = sub.add_parser("build", help="Run the DP2 SkyKatana mask builder from a YAML config")
    p_build.add_argument("-c", "--config", required=True, help="Path to YAML configuration file")
    p_build.add_argument("-o", "--output-dir", default=None, help="Override output directory")
    p_build.add_argument(
        "--token-file",
        default=None,
        help="Optional external RSP Gafaelfawr token file (when running outside RSP)",
    )

    # 2. estimate-ram
    p_ram = sub.add_parser(
        "estimate-ram",
        help="Check 32 GB container RAM ceiling and analytic stage-by-stage memory budget",
    )
    p_ram.add_argument("-c", "--config", required=True, help="Path to YAML configuration file")
    p_ram.add_argument(
        "--area-deg2",
        type=float,
        default=6000.0,
        help="Assumed survey footprint area in deg^2 (default: 6000.0)",
    )

    # 3. inspect-butler
    p_ins = sub.add_parser(
        "inspect-butler",
        help="List available DP2 consolidated sp_maps and the_monster datasets in Butler",
    )
    p_ins.add_argument("--repo", default="dp2", help="Butler repository label or path (default: dp2)")
    p_ins.add_argument(
        "--collection",
        action="append",
        default=None,
        help="Butler collection(s) to search (default: dp2)",
    )
    p_ins.add_argument("--token-file", default=None, help="Optional external RSP token file")

    # 4. demo-synthetic
    p_demo = sub.add_parser(
        "demo-synthetic",
        help="Run a full offline verification using synthetic DP2 sp_maps + Monster shards",
    )
    p_demo.add_argument(
        "-o",
        "--output-dir",
        default="outputs/demo_synthetic",
        help="Directory for synthetic demo artifacts",
    )
    p_demo.add_argument(
        "--order",
        type=int,
        default=13,
        help="Sparse HEALPix order for synthetic demo (default: 13 -> NSIDE=8192)",
    )

    return parser


def _cmd_build(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.output_dir:
        cfg.output.output_dir = str(args.output_dir)
    if args.token_file:
        cfg.access_token_file = str(args.token_file)

    builder = DP2MaskBuilder(cfg)
    est = builder.estimate_memory()
    logging.getLogger(__name__).info("Estimated peak working set: %s", json.dumps(est))
    _, summary = builder.run()
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


def _cmd_estimate_ram(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    builder = DP2MaskBuilder(cfg)
    limit_gib, limit_src = detect_effective_memory_limit_gib(cfg.memory.max_ram_gib)
    est = builder.estimate_memory(assumed_area_deg2=float(args.area_deg2))
    payload = {
        "effective_limit_gib": round(limit_gib, 2),
        "limit_source": limit_src,
        "assumed_area_deg2": float(args.area_deg2),
        "budget_gib": est,
        "fits_in_limit": bool(est["estimated_peak_ram_gib"] < 0.85 * limit_gib),
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["fits_in_limit"] else 2


def _cmd_inspect_butler(args: argparse.Namespace) -> int:
    colls = args.collection if args.collection else ["dp2"]
    cfg = MaskPipelineConfig(
        butler_repo=args.repo,
        butler_collections=colls,
        access_token_file=args.token_file,
    )
    butler = open_dp2_butler(cfg)
    sp_maps = list_consolidated_maps(butler)
    monster_ds = discover_monster_dataset(butler, "auto")
    print(
        json.dumps(
            {
                "butler_repo": args.repo,
                "collections": colls,
                "n_consolidated_sp_maps": len(sp_maps),
                "consolidated_sp_maps": sp_maps,
                "monster_dataset": monster_ds,
            },
            indent=2,
        )
    )
    return 0


def _cmd_demo_synthetic(args: argparse.Namespace) -> int:
    order = int(args.order)
    nside = 1 << order
    mock_butler = SyntheticDP2Butler(
        center_ra_deg=53.1222,
        center_dec_deg=-28.0870,
        radius_deg=1.4,
        nside_cov=32,
        nside_sparse=nside,
        n_stars=220,
    )
    cfg = MaskPipelineConfig(
        butler_repo="synthetic_dp2",
        butler_collections=["dp2"],
        property_maps=PropertyMapsConfig(
            order_sparse=order,
            order_cov=5,
            bands=["u", "g", "r", "i", "z", "y"],
            min_filters=4,
            required_bands=["r", "i"],
            per_band={
                "u": BandCriteria(min_exposures=3, min_maglim=23.9),
                "g": BandCriteria(min_exposures=5, min_maglim=25.0),
                "r": BandCriteria(min_exposures=6, min_maglim=24.8, max_psf_fwhm_arcsec=1.25),
                "i": BandCriteria(min_exposures=6, min_maglim=24.4, max_psf_fwhm_arcsec=1.25),
                "z": BandCriteria(min_exposures=4, min_maglim=23.7),
                "y": BandCriteria(min_exposures=3, min_maglim=22.5),
            },
        ),
        monster_stars=MonsterStarConfig(
            enabled=True,
            order_sparse=order,
            reference_bands=["g", "r", "i", "z"],
            mag_max=18.0,
            enable_bleed_boxes=True,
            bleed_mag_max=11.5,
        ),
        output=OutputConfig(
            output_dir=str(Path(args.output_dir)),
            order_out=order,
            order_cov=5,
            moc_max_order=min(12, order),
            fracdet_order=min(10, order - 1),
            n_random_points=25_000,
            make_qa_plots=True,
        ),
    )
    builder = DP2MaskBuilder(cfg, butler=mock_butler)
    _, summary = builder.run()
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


def _maybe_reexec_with_rsp_lsst_distrib(argv: Sequence[str] | None, command: str) -> None:
    """Auto-bootstrap `loadLSST.bash` + `setup lsst_distrib` once in fresh RSP terminals."""
    if argv is not None or command not in ("build", "inspect-butler"):
        return
    if os.environ.get("_DP2_MASK_RSP_BOOTSTRAPPED") == "1":
        return
    try:
        import lsst.daf.butler  # noqa: F401
        return
    except ImportError:
        pass

    loader = find_rsp_stack_loader()
    if loader is None:
        return

    src_root = str(Path(__file__).resolve().parents[1])
    logging.getLogger(__name__).info(
        "RSP terminal without active `lsst_distrib` detected; bootstrapping `%s` + `setup lsst_distrib`...",
        loader,
    )
    os.environ["_DP2_MASK_RSP_BOOTSTRAPPED"] = "1"
    bash_cmd = (
        f'source "{loader}" && setup lsst_distrib && '
        f'PYTHONPATH="{src_root}:${{PYTHONPATH:-}}" exec python -m dp2_skykatana_mask.cli "$@"'
    )
    os.execvp("bash", ["bash", "-c", bash_cmd, "dp2-mask", *sys.argv[1:]])


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(verbose=bool(args.verbose))
    _maybe_reexec_with_rsp_lsst_distrib(argv, str(args.command))

    dispatch = {
        "build": _cmd_build,
        "estimate-ram": _cmd_estimate_ram,
        "inspect-butler": _cmd_inspect_butler,
        "demo-synthetic": _cmd_demo_synthetic,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
