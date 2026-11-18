"""Unit and end-to-end notebook tests for local IVOA TAP execution (`tap_local` + Notebook 02)."""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib
import numpy as np
import pandas as pd
from astropy.io.votable import from_table
from astropy.table import Table

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dp2_skykatana_mask import (
    IVOATapClient,
    SyntheticDP2TapService,
    discover_dp2_tap_schema,
    gaia_to_synthetic_lsst_magnitudes,
    open_rsp_tap_client,
    parse_stcs_region_pixels,
    resolve_rsp_tap_token,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestLocalTapPipeline(unittest.TestCase):
    def test_token_resolution_and_bearer_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tok_file = Path(tmp) / "rsp.token"
            tok_file.write_text("  gt-secret-test-token-123  \n", encoding="utf-8")
            resolved = resolve_rsp_tap_token(token_file=tok_file)
            self.assertEqual(resolved, "gt-secret-test-token-123")

            client = IVOATapClient(
                "https://data.lsst.cloud/api/tap",
                token=resolved,
                prefer_pyvo=False,
            )
            self.assertEqual(
                client._session.headers.get("Authorization"),
                "Bearer gt-secret-test-token-123",
            )

            # Verify VOTable HTTP sync parsing without network
            mock_tbl = Table({"ra": [53.1, 53.2], "dec": [-28.1, -28.0]})
            vot = from_table(mock_tbl)
            buf = io.BytesIO()
            vot.to_xml(buf)

            fake_resp = mock.MagicMock()
            fake_resp.content = buf.getvalue()
            fake_resp.raise_for_status.return_value = None
            with mock.patch.object(client._session, "post", return_value=fake_resp) as mock_post:
                out = client.search("SELECT ra, dec FROM dp2.Object", maxrec=10).to_table()
                mock_post.assert_called_once()
                self.assertEqual(len(out), 2)
                self.assertAlmostEqual(float(out["ra"][0]), 53.1, places=4)

    def test_stcs_parsing_and_gaia_lsst_color_locus(self) -> None:
        poly_pix = parse_stcs_region_pixels(
            "POLYGON ICRS 53.0 -28.2 53.3 -28.2 53.3 -27.9 53.0 -27.9",
            nside=1024,
        )
        self.assertGreater(poly_pix.size, 20)

        circ_pix = parse_stcs_region_pixels("CIRCLE ICRS 53.1 -28.1 0.15", nside=1024)
        self.assertGreater(circ_pix.size, 15)

        # Solar-type star (G=12.0, BP-RP=0.82): g - r ~ 0.46, r - i ~ 0.15, u > g > r > i > z
        mags = gaia_to_synthetic_lsst_magnitudes([12.0], [0.82])
        self.assertAlmostEqual(
            float(mags["lsst_g_mag"][0] - mags["lsst_r_mag"][0]), 0.46, delta=0.08
        )
        self.assertGreater(float(mags["lsst_u_mag"][0]), float(mags["lsst_g_mag"][0]))
        self.assertGreater(float(mags["lsst_g_mag"][0]), float(mags["lsst_r_mag"][0]))
        self.assertGreater(float(mags["lsst_r_mag"][0]), float(mags["lsst_i_mag"][0]))

    def test_schema_discovery_and_offline_fallback(self) -> None:
        with mock.patch.dict(os.environ, {"RSP_ACCESS_TOKEN": "", "ACCESS_TOKEN": ""}, clear=False):
            with mock.patch("pathlib.Path.home", return_value=Path("/nonexistent_home_for_test")):
                tap = open_rsp_tap_client(allow_synthetic_fallback=True)
                self.assertIsInstance(tap, SyntheticDP2TapService)
                sinfo = discover_dp2_tap_schema(tap)
                self.assertEqual(sinfo.catalog_schema, "dp2")
                self.assertEqual(sinfo.visit_detector_table, "dp2.VisitDetector")
                self.assertTrue(sinfo.has_obscore)

    def test_local_tap_notebook_02_executes_cleanly(self) -> None:
        nb_path = REPO_ROOT / "notebooks" / "02_DP2_Skykatana_Mask_Local_TAP.ipynb"
        nb_json = json.loads(nb_path.read_text(encoding="utf-8"))
        code_cells = [c for c in nb_json["cells"] if c.get("cell_type") == "code"]
        self.assertGreaterEqual(len(code_cells), 6)

        ns: dict[str, object] = {"display": lambda *args, **kwargs: None}
        outputs_dir = REPO_ROOT / "outputs"
        existed_before = outputs_dir.exists()
        orig_cwd = Path.cwd()
        orig_show = plt.show
        plt.show = lambda *args, **kwargs: None
        try:
            os.chdir(REPO_ROOT)
            with mock.patch.dict(os.environ, {"RSP_ACCESS_TOKEN": "", "ACCESS_TOKEN": ""}, clear=False):
                with mock.patch("pathlib.Path.home", return_value=Path("/nonexistent_home_for_test")):
                    for idx, cell in enumerate(code_cells):
                        src = "".join(cell.get("source", []))
                        exec(compile(src, f"{nb_path.name}#cell-{idx}-{cell.get('id')}", "exec"), ns)
                        plt.close("all")

            # Verify output artifacts from Notebook 02 before cleanup
            local_out = REPO_ROOT / "outputs" / "local_tap_ecdfs"
            self.assertTrue((local_out / "dp2_local_tap_complete_mask.png").is_file())
            self.assertTrue((local_out / "dp2_local_tap_fracmap.png").is_file())
            self.assertTrue((local_out / "dp2_local_tap_allsky.png").is_file())
            self.assertTrue((local_out / "dp2_local_tap_mask_moc.fits").is_file())
            self.assertTrue((local_out / "dp2_local_tap_clean_galaxies.parquet").is_file())

            clean_gal = pd.read_parquet(local_out / "dp2_local_tap_clean_galaxies.parquet")
            self.assertGreater(len(clean_gal), 1_000)
            # Dust mask enforced max_ebv = 0.20 ceiling on surviving galaxies!
            self.assertLessEqual(float(clean_gal["ebv"].max()), 0.215)
        finally:
            os.chdir(orig_cwd)
            plt.show = orig_show
            plt.close("all")
            if not existed_before and outputs_dir.exists():
                shutil.rmtree(outputs_dir, ignore_errors=True)

        self.assertIn("mkp_local", ns)
        self.assertIn("clean_gal_df", ns)


if __name__ == "__main__":
    unittest.main()
