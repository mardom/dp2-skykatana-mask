"""Unit tests for YAML configurations, 32 GiB RSP memory guards, Tutorial 311.1 dust math, and SkyKatana radfunction."""

from __future__ import annotations

import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from dp2_skykatana_mask import (
    BandCriteria,
    DEFAULT_RUBIN_RX_RV31,
    DP2MaskBuilder,
    MonsterStarConfig,
    compute_band_extinction_mag,
    deredden_magnitude,
    get_rubin_band_extinction_rx,
    load_config,
    make_skykatana_radfunction,
    skykatana_lsst_radfunction,
)
from dp2_skykatana_mask.butler_io import resolve_monster_flux_columns
from dp2_skykatana_mask.memory import (
    GIB,
    MemoryGuard,
    detect_effective_memory_limit_gib,
    estimate_working_set_gib,
)
from dp2_skykatana_mask.monster_stars import (
    AB_MAG_ZEROPOINT_NJY,
    compute_star_halo_radius_arcsec,
    flux_njy_to_abmag,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestConfigAndMemory(unittest.TestCase):
    def test_all_shipped_yaml_configs_load_and_fit_32gb(self) -> None:
        configs_dir = REPO_ROOT / "configs"
        yaml_files = sorted(configs_dir.glob("*.yaml"))
        self.assertGreaterEqual(len(yaml_files), 3)

        for yf in yaml_files:
            cfg = load_config(yf)
            self.assertTrue(cfg.dust.enabled)
            self.assertEqual(cfg.monster_stars.radius_model, "skykatana_rubin")
            builder = DP2MaskBuilder(cfg)
            est = builder.estimate_memory(assumed_area_deg2=6500.0)
            self.assertLess(
                est["estimated_peak_ram_gib"],
                27.2,  # < 85% of 32 GiB RSP ceiling
                msg=f"Config {yf.name} exceeded 32 GiB RSP budget: {est}",
            )

    def test_nublado_mem_limit_detection(self) -> None:
        fake_bytes = str(int(16 * GIB))
        with mock.patch.dict(os.environ, {"MEM_LIMIT": fake_bytes}):
            lim, src = detect_effective_memory_limit_gib(configured_limit_gib=32.0)
            self.assertAlmostEqual(lim, 16.0, places=2)
            self.assertEqual(src, "RSP MEM_LIMIT")

    def test_memory_guard_snapshot(self) -> None:
        guard = MemoryGuard(max_ram_gib=32.0)
        snap = guard.check("unit-test")
        self.assertGreater(snap.process_rss_gib, 0.0)
        self.assertLessEqual(snap.effective_limit_gib, 32.0)

    def test_band_criteria_conversions(self) -> None:
        crit = BandCriteria(
            min_exposures=5,
            nominal_exptime_sec=30.0,
            exposure_Subpixel_tol=0.0,
            max_psf_fwhm_arcsec=1.17741,
        )
        self.assertAlmostEqual(crit.effective_min_exptime_sec() or 0.0, 150.0)
        # 1.17741" / 2.35482 = 0.5" sigma -> 0.5" / 0.2"/pix = 2.5 pixels
        self.assertAlmostEqual(crit.effective_max_psf_pixels() or 0.0, 2.5, places=4)

    def test_tutorial_311_1_rubin_dust_extinction(self) -> None:
        rx = get_rubin_band_extinction_rx(r_v=3.1)
        for band, expected in DEFAULT_RUBIN_RX_RV31.items():
            # rubin_sim.phot_utils.DustValues (when installed) uses live SysEng throughputs (~4.76 in u)
            self.assertAlmostEqual(rx[band], expected, delta=0.10)

        # Monotonic wavelength dependence: R_u > R_g > R_r > R_i > R_z > R_y
        ordered = [rx[b] for b in ("u", "g", "r", "i", "z", "y")]
        self.assertEqual(ordered, sorted(ordered, reverse=True))

        # Verify A_b = R_b * E(B-V) and m_dered = m_obs - A_b
        ebv = np.array([0.0, 0.10, 0.25])
        a_g = compute_band_extinction_mag(ebv, band="g", rx_map=DEFAULT_RUBIN_RX_RV31)
        np.testing.assert_allclose(a_g, np.array([0.0, 0.364, 0.910]), rtol=1e-4)

        m_obs = np.array([25.0, 25.0, 25.0])
        m_dered = deredden_magnitude(m_obs, ebv, band="g", rx_map=DEFAULT_RUBIN_RX_RV31)
        np.testing.assert_allclose(m_dered, np.array([25.0, 24.636, 24.090]), rtol=1e-4)

    def test_skykatana_lsst_radfunction_and_models(self) -> None:
        df = pd.DataFrame(
            {
                "mag": [5.0, 10.0, 14.0, 17.5],
                "lsst_g_mag": [5.4, 10.3, 14.2, 17.8],
                "lsst_r_mag": [5.0, 10.0, 14.0, 17.5],
                "lsst_i_mag": [4.6, 9.7, 13.8, 17.2],
            }
        )
        out_r85 = skykatana_lsst_radfunction(
            df.copy(), band="r", completeness=0.85, min_radius_arcsec=0.0, max_radius_arcsec=3600.0
        )
        self.assertIn("radius", out_r85.columns)
        r_deg = out_r85["radius"].to_numpy()
        # Strictly decreasing with fainter LSST magnitude
        self.assertTrue(np.all(np.diff(r_deg) < 0.0))
        # Exact Rubin r-band 85% completeness calibration at m_LSST = 10.0 (converted to degrees)
        expected_deg_m10 = (1068.4259556870957 * math.exp(-0.23979264155393915 * 10.0)) / 3600.0
        self.assertAlmostEqual(float(r_deg[1]), expected_deg_m10, places=6)

        # Completeness monotonicity: R(c=0.90) > R(c=0.85) > R(c=0.80)
        out_r90 = skykatana_lsst_radfunction(df.copy(), band="r", completeness=0.90, min_radius_arcsec=0.0)
        out_r80 = skykatana_lsst_radfunction(df.copy(), band="r", completeness=0.80, min_radius_arcsec=0.0)
        self.assertTrue(np.all(out_r90["radius"].to_numpy() > out_r85["radius"].to_numpy()))
        self.assertTrue(np.all(out_r85["radius"].to_numpy() > out_r80["radius"].to_numpy()))

        # 3-piece scattered-light r_halo gives wider halo at m_r=5.0 than single-exponential core
        out_halo = skykatana_lsst_radfunction(df.copy(), band="r_halo", completeness=0.85, min_radius_arcsec=0.0)
        self.assertGreater(float(out_halo["radius"].iloc[0]), float(out_r85["radius"].iloc[0]))

        # Multi-band envelope >= single r-band radius
        ms_multi = MonsterStarConfig(
            radius_model="skykatana_rubin",
            profile_band="multi",
            completeness=0.85,
            min_radius_arcsec=0.0,
            max_radius_arcsec=3600.0,
        )
        radfn_multi = make_skykatana_radfunction(ms_multi)
        out_multi = radfn_multi(df.copy())
        self.assertTrue(np.all(out_multi["radius"].to_numpy() >= out_r85["radius"].to_numpy() - 1e-9))

    def test_flux_njy_to_abmag_and_legacy_radii(self) -> None:
        # Exact Oke & Gunn AB zero-point flux: 10**(31.4/2.5) nJy ~= 3.6307805e12 nJy -> AB mag = 0.0
        zero_mag_njy = 10.0 ** (AB_MAG_ZEROPOINT_NJY / 2.5)
        mags = flux_njy_to_abmag(np.array([zero_mag_njy, 3.631e12, 1.0, 0.0, -5.0]))
        self.assertAlmostEqual(float(mags[0]), 0.0, places=6)
        self.assertAlmostEqual(float(mags[1]), 0.0, places=3)
        self.assertAlmostEqual(float(mags[2]), AB_MAG_ZEROPOINT_NJY, places=6)
        self.assertTrue(math.isnan(float(mags[3])))
        self.assertTrue(math.isnan(float(mags[4])))

        ms_exp = MonsterStarConfig(
            radius_model="exponential",
            A_arcsec=1000.0,
            B_per_mag=-0.25,
            mag_knee=18.0,
            floor_arcsec=7.5,
            min_radius_arcsec=5.0,
            max_radius_arcsec=500.0,
        )
        r_exp = compute_star_halo_radius_arcsec(np.array([8.0, 14.0, 19.0]), ms_exp)
        self.assertGreater(r_exp[0], r_exp[1])
        self.assertAlmostEqual(float(r_exp[2]), 7.5, places=4)

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "coefs.csv"
            csv_path.write_text(
                "profile_behavior,A_arcsec,B_per_mag\ndeficit,1100.0,-0.23\nexceso,900.0,-0.21\n",
                encoding="utf-8",
            )
            ms_csv = MonsterStarConfig(
                radius_model="csv_fit",
                fit_coefficients_csv=str(csv_path),
                fit_behavior_branch="deficit",
            )
            ms_csv.validate()
            r_csv = compute_star_halo_radius_arcsec(np.array([12.0]), ms_csv)
            expected = 1100.0 * math.exp(-0.23 * 12.0)
            self.assertAlmostEqual(float(r_csv[0]), expected, places=4)

    def test_monster_flux_column_resolution_priority(self) -> None:
        comcam_cols = ["coord_ra", "coord_dec", "monster_ComCam_g_flux", "monster_ComCam_r_flux"]
        resolved = resolve_monster_flux_columns(comcam_cols, ["g", "r", "i"])
        self.assertEqual(resolved, ["monster_ComCam_g_flux", "monster_ComCam_r_flux"])

        lsst_cols = comcam_cols + ["monster_LSST_r_flux"]
        resolved2 = resolve_monster_flux_columns(lsst_cols, ["r"])
        self.assertEqual(resolved2, ["monster_LSST_r_flux"])

    def test_working_set_estimator_scales_with_order(self) -> None:
        est14 = estimate_working_set_gib(6000.0, order_prop=14, order_star=14, order_out=14)
        est15 = estimate_working_set_gib(6000.0, order_prop=15, order_star=15, order_out=15)
        self.assertGreater(est15["estimated_peak_ram_gib"], est14["estimated_peak_ram_gib"])


if __name__ == "__main__":
    unittest.main()
