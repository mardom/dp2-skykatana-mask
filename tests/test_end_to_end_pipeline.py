"""End-to-end integration tests with real SkyMaskPipe, HealSparse, MOCPy, and SyntheticDP2Butler."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import healsparse as hsp
import numpy as np
import pandas as pd
from mocpy import MOC
from skykatana import SkyMaskPipe

from dp2_skykatana_mask import (
    BandCriteria,
    DP2MaskBuilder,
    GalacticMaskConfig,
    MaskPipelineConfig,
    MonsterStarConfig,
    OutputConfig,
    PropertyMapsConfig,
    RegionSelector,
)
from dp2_skykatana_mask.cli import main as cli_main
from dp2_skykatana_mask.synthetic import SyntheticDP2Butler

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestEndToEndPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.butler = SyntheticDP2Butler(
            center_ra_deg=53.1222,
            center_dec_deg=-28.0870,
            radius_deg=1.2,
            nside_cov=32,
            nside_sparse=4096,  # order=12 for fast crisp tests
            n_stars=140,
            seed=42,
        )

    def test_full_pipeline_stages_persistence_randoms_and_apply(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = MaskPipelineConfig(
                butler_repo="synthetic_dp2",
                butler_collections=["dp2"],
                region=RegionSelector(
                    mode="cone",
                    ra_deg=53.1222,
                    dec_deg=-28.0870,
                    radius_deg=1.1,
                ),
                property_maps=PropertyMapsConfig(
                    order_sparse=12,
                    order_cov=5,
                    bands=["u", "g", "r", "i", "z", "y"],
                    min_filters=4,
                    required_bands=["r", "i"],
                    keep_per_band_stages=True,
                    per_band={
                        "u": BandCriteria(min_exposures=3, min_maglim=23.9),
                        "g": BandCriteria(min_exposures=5, min_maglim=25.0),
                        "r": BandCriteria(min_exposures=6, min_maglim=24.7, max_psf_fwhm_arcsec=1.3),
                        "i": BandCriteria(min_exposures=6, min_maglim=24.3, max_psf_fwhm_arcsec=1.3),
                        "z": BandCriteria(min_exposures=4, min_maglim=23.6),
                        "y": BandCriteria(min_exposures=3, min_maglim=22.4),
                    },
                ),
                monster_stars=MonsterStarConfig(
                    enabled=True,
                    order_sparse=12,
                    reference_bands=["g", "r", "i", "z"],
                    mag_max=18.0,
                    enable_bleed_boxes=True,
                    bleed_mag_max=11.5,
                ),
                galactic_mask=GalacticMaskConfig(
                    enabled=True,
                    order_sparse=9,
                    b0_deg=10.0,
                ),
                output=OutputConfig(
                    output_dir=tmpdir,
                    order_out=12,
                    order_cov=5,
                    moc_max_order=11,
                    fracdet_order=9,
                    n_random_points=5_000,
                    make_qa_plots=True,
                ),
            )

            builder = DP2MaskBuilder(cfg, butler=self.butler)
            pipe, summary = builder.run()

            # Verify positive propmask > final mask (star halos punched holes)
            self.assertGreater(summary.property_mask.area_deg2, 0.5)
            self.assertIsNotNone(summary.monster_stars)
            assert summary.monster_stars is not None
            self.assertGreater(summary.monster_stars.n_stars_selected, 10)
            self.assertGreater(summary.monster_stars.n_bleed_boxes, 0)
            self.assertLess(summary.final_area_deg2, summary.property_mask.area_deg2)
            self.assertGreater(summary.final_area_deg2, 0.2)

            # Verify bit-packed memory efficiency on all SkyMaskPipe stages
            self.assertTrue(pipe.propmask.is_bit_packed_map)
            self.assertTrue(pipe.starmask.is_bit_packed_map)
            self.assertTrue(pipe.mask.is_bit_packed_map)

            # Verify all expected output artifacts exist and reload cleanly
            for key in (
                "skykatana_pipe_dir",
                "combined_healsparse_hs",
                "moc_fits",
                "fracdet_healsparse_hs",
                "randoms_parquet",
                "qa_plot_png",
                "summary_json",
            ):
                self.assertIn(key, summary.output_files)
                self.assertTrue(Path(summary.output_files[key]).exists(), f"Missing {key}")

            reloaded_pipe = SkyMaskPipe.read(summary.output_files["skykatana_pipe_dir"], verbose=False)
            self.assertEqual(int(reloaded_pipe.mask.n_valid), summary.final_valid_pixels)

            hs_map = hsp.HealSparseMap.read(summary.output_files["combined_healsparse_hs"])
            self.assertEqual(int(hs_map.n_valid), summary.final_valid_pixels)

            moc = MOC.from_fits(summary.output_files["moc_fits"])
            self.assertGreater(moc.sky_fraction, 0.0)

            # Every point produced by SkyMaskPipe.makerans MUST survive SkyMaskPipe.apply!
            rans = pd.read_parquet(summary.output_files["randoms_parquet"])
            self.assertEqual(len(rans), 5_000)
            filtered_rans = reloaded_pipe.apply(stage="mask", cat=rans, columns=("ra", "dec"))
            self.assertEqual(len(filtered_rans), len(rans))

            # Stars masked by Monster (`mag <= 18`) inside footprint must be rejected by `.apply()`
            assert summary.monster_stars.circles_parquet_path is not None
            stars_df = pd.read_parquet(summary.monster_stars.circles_parquet_path)
            surviving_star_centers = reloaded_pipe.apply(
                stage="mask", cat=stars_df, columns=("ra", "dec")
            )
            self.assertEqual(len(surviving_star_centers), 0)

    def test_min_filters_monotonicity(self) -> None:
        areas: list[float] = []
        for min_f in (2, 4, 6):
            with tempfile.TemporaryDirectory() as tmpdir:
                cfg = MaskPipelineConfig(
                    property_maps=PropertyMapsConfig(
                        order_sparse=12,
                        order_cov=5,
                        bands=["u", "g", "r", "i", "z", "y"],
                        min_filters=min_f,
                        per_band={
                            "u": BandCriteria(min_exposures=5, min_maglim=24.1),
                            "g": BandCriteria(min_exposures=9, min_maglim=25.2),
                            "r": BandCriteria(min_exposures=11, min_maglim=24.9),
                            "i": BandCriteria(min_exposures=11, min_maglim=24.5),
                            "z": BandCriteria(min_exposures=8, min_maglim=23.8),
                            "y": BandCriteria(min_exposures=7, min_maglim=22.7),
                        },
                    ),
                    monster_stars=MonsterStarConfig(enabled=False),
                    output=OutputConfig(
                        output_dir=tmpdir,
                        order_out=12,
                        order_cov=5,
                        write_skykatana_pipe=False,
                        write_moc_fits=False,
                        write_fracdet_map=False,
                        n_random_points=0,
                        make_qa_plots=False,
                    ),
                )
                _, summary = DP2MaskBuilder(cfg, butler=self.butler).run()
                areas.append(summary.final_area_deg2)

        self.assertGreater(areas[0], areas[1])
        self.assertGreater(areas[1], areas[2])

    def test_cli_estimate_ram_and_demo_synthetic(self) -> None:
        rc_ram = cli_main(
            ["estimate-ram", "--config", str(REPO_ROOT / "configs" / "dp2_wfd_default.yaml")]
        )
        self.assertEqual(rc_ram, 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            rc_demo = cli_main(
                ["demo-synthetic", "--output-dir", tmpdir, "--order", "12"]
            )
            self.assertEqual(rc_demo, 0)
            self.assertTrue((Path(tmpdir) / "dp2_combined_mask.hs").exists())


if __name__ == "__main__":
    unittest.main()
