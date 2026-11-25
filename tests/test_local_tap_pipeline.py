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
from astropy.io.votable import from_table
from astropy.table import Table

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dp2_skykatana_mask import (
    DP2LocalTapMaskBuilder,
    IVOATapClient,
    SyntheticDP2TapService,
    discover_dp2_tap_schema,
    gaia_to_synthetic_lsst_magnitudes,
    load_config,
    open_rsp_tap_client,
    parse_stcs_region_pixels,
    resolve_rsp_tap_token,
)
from dp2_skykatana_mask.cli import main as cli_main

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

            # Verify sync timeout auto-promotes to strict 2-step UWS 1.1 /async + external GCS signed URL
            timeout_xml = (
                b'<?xml version="1.0"?><VOTABLE version="1.3"><RESOURCE type="results">'
                b'<INFO name="QUERY_STATUS" value="ERROR">'
                b"Query timeout exceeded for synchronous execution. Please use /async endpoint."
                b"</INFO></RESOURCE></VOTABLE>"
            )
            sync_err_resp = mock.MagicMock(status_code=200, content=timeout_xml)
            sync_err_resp.raise_for_status.return_value = None

            create_resp = mock.MagicMock(
                status_code=303,
                headers={"Location": "http://data.lsst.cloud/api/tap/async/job99"},
                url="https://data.lsst.cloud/api/tap/async",
            )
            run_resp = mock.MagicMock(status_code=303, headers={"Location": "https://data.lsst.cloud/api/tap/async/job99"})
            phase_pending = mock.MagicMock(status_code=200, text="PENDING", content=b"PENDING")
            phase_pending.raise_for_status.return_value = None
            phase_done = mock.MagicMock(status_code=200, text="COMPLETED", content=b"COMPLETED")
            phase_done.raise_for_status.return_value = None
            redir_gcs = mock.MagicMock(
                status_code=303,
                headers={"Location": "https://storage.googleapis.com/rubin-tap/res.vot?sig=abc"},
                url="https://data.lsst.cloud/api/tap/async/job99/results/result",
            )
            gcs_ok = mock.MagicMock(status_code=200, content=buf.getvalue())
            gcs_ok.raise_for_status.return_value = None

            with (
                mock.patch.object(client._session, "post", side_effect=[sync_err_resp, create_resp, run_resp]) as m_post,
                mock.patch.object(client._session, "get", side_effect=[phase_pending, phase_done, redir_gcs]),
                mock.patch.object(client._session, "delete") as m_del,
                mock.patch("requests.get", return_value=gcs_ok) as m_ext_get,
            ):
                promoted = client.search("SELECT ra, dec FROM dp2.VisitDetector", maxrec=100).to_table()
                self.assertEqual(len(promoted), 2)
                # Initial /async creation POST must NOT include PHASE=RUN; 2nd POST starts /phase
                self.assertNotIn("PHASE", m_post.call_args_list[1].kwargs["data"])
                self.assertEqual(m_post.call_args_list[2].kwargs["data"], {"PHASE": "RUN"})
                m_ext_get.assert_called_once()
                m_del.assert_called_once()

    def test_stcs_parsing_and_gaia_lsst_color_locus(self) -> None:
        poly_pix = parse_stcs_region_pixels(
            "POLYGON ICRS 53.0 -28.2 53.3 -28.2 53.3 -27.9 53.0 -27.9",
            nside=1024,
        )
        self.assertGreater(poly_pix.size, 20)

        circ_pix = parse_stcs_region_pixels("CIRCLE ICRS 53.1 -28.1 0.15", nside=1024)
        self.assertGreater(circ_pix.size, 15)

        mags = gaia_to_synthetic_lsst_magnitudes([12.0], [0.82])
        self.assertAlmostEqual(
            float(mags["lsst_g_mag"][0] - mags["lsst_r_mag"][0]), 0.46, delta=0.08
        )
        self.assertGreater(float(mags["lsst_u_mag"][0]), float(mags["lsst_g_mag"][0]))
        self.assertGreater(float(mags["lsst_g_mag"][0]), float(mags["lsst_r_mag"][0]))
        self.assertGreater(float(mags["lsst_r_mag"][0]), float(mags["lsst_i_mag"][0]))

    def test_full_mode_builder_and_cli_build_tap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            wfd_cfg = load_config(REPO_ROOT / "configs" / "dp2_wfd_default.yaml")
            wfd_cfg.output.output_dir = tmpdir
            wfd_cfg.property_maps.order_sparse = 12
            wfd_cfg.dust.order_sparse = 11
            wfd_cfg.monster_stars.order_sparse = 12
            wfd_cfg.output.order_out = 12
            wfd_cfg.output.moc_max_order = 11
            wfd_cfg.output.n_random_points = 3_000
            # Adjust depth thresholds slightly for synthetic visit counts
            for b, lim in (("g", 24.8), ("r", 24.5), ("i", 24.1), ("z", 23.4)):
                wfd_cfg.property_maps.per_band[b].min_exposures = 3
                wfd_cfg.property_maps.per_band[b].min_maglim = lim

            mock_tap = SyntheticDP2TapService(n_stars=120, n_objects=6_000)
            builder = DP2LocalTapMaskBuilder(wfd_cfg, tap_client=mock_tap)
            pipe, summary = builder.run()

            self.assertEqual(summary.region_mode, "full")
            self.assertGreater(summary.final_area_deg2, 1.0)
            self.assertIn("mwmask", summary.negative_stages)
            self.assertIn("complete_mask_multiscale_png", summary.output_files)
            self.assertTrue(Path(summary.output_files["complete_mask_multiscale_png"]).is_file())
            self.assertIsNotNone(pipe.mask)

        # Also test CLI `dp2-mask build-tap --allow-synthetic-fallback`
        with tempfile.TemporaryDirectory() as tmp_cli:
            # Use order=12 in CLI test so synthetic pilot completes in ~1s instead of order=15!
            pilot_fast_cfg = REPO_ROOT / "configs" / "dp2_ecdfs_pilot.yaml"
            fast_yaml = Path(tmp_cli) / "fast_pilot.yaml"
            cfg_obj = load_config(pilot_fast_cfg)
            cfg_obj.property_maps.order_sparse = 12
            cfg_obj.dust.order_sparse = 11
            cfg_obj.monster_stars.order_sparse = 12
            cfg_obj.output.order_out = 12
            cfg_obj.output.moc_max_order = 11
            cfg_obj.output.n_random_points = 2_000
            import yaml

            raw_dict = yaml.safe_load(pilot_fast_cfg.read_text(encoding="utf-8"))
            raw_dict["property_maps"]["order_sparse"] = 12
            raw_dict["dust"]["order_sparse"] = 11
            raw_dict["monster_stars"]["order_sparse"] = 12
            raw_dict["output"]["order_out"] = 12
            raw_dict["output"]["moc_max_order"] = 11
            raw_dict["output"]["n_random_points"] = 2000
            fast_yaml.write_text(yaml.safe_dump(raw_dict), encoding="utf-8")

            with mock.patch.dict(os.environ, {"RSP_ACCESS_TOKEN": "", "ACCESS_TOKEN": ""}, clear=False):
                with mock.patch("pathlib.Path.home", return_value=Path("/nonexistent_home_for_test")):
                    rc = cli_main(
                        [
                            "build-tap",
                            "--config",
                            str(fast_yaml),
                            "--output-dir",
                            tmp_cli,
                            "--allow-synthetic-fallback",
                        ]
                    )
                    self.assertEqual(rc, 0)
                    self.assertTrue((Path(tmp_cli) / "dp2_complete_mask_multiscale.png").is_file())

    def test_local_tap_notebook_02_executes_cleanly(self) -> None:
        nb_rsp = json.loads((REPO_ROOT / "notebooks" / "01_DP2_Skykatana_Mask_RSP.ipynb").read_text(encoding="utf-8"))
        nb_path = REPO_ROOT / "notebooks" / "02_DP2_Skykatana_Mask_Local_TAP.ipynb"
        nb_json = json.loads(nb_path.read_text(encoding="utf-8"))
        rsp_code_cells = [c for c in nb_rsp["cells"] if c.get("cell_type") == "code"]
        code_cells = [c for c in nb_json["cells"] if c.get("cell_type") == "code"]
        # Both notebooks share the exact same 21-cell / 11-or-13 code-cell structure!
        self.assertEqual(len(nb_json["cells"]), len(nb_rsp["cells"]))
        self.assertEqual(len(code_cells), len(rsp_code_cells))

        ns: dict[str, object] = {"display": lambda *args, **kwargs: None}
        outputs_dir = REPO_ROOT / "outputs"
        existed_before = outputs_dir.exists()
        orig_cwd = Path.cwd()
        orig_show = plt.show
        plt.show = lambda *args, **kwargs: None
        try:
            os.chdir(REPO_ROOT)
            with tempfile.TemporaryDirectory() as fake_home:
                with mock.patch.dict(
                    os.environ,
                    {"RSP_ACCESS_TOKEN": "", "ACCESS_TOKEN": "", "HOME": fake_home},
                    clear=False,
                ):
                    with mock.patch("pathlib.Path.home", return_value=Path(fake_home)):
                        for idx, cell in enumerate(code_cells):
                            src = "".join(cell.get("source", []))
                            exec(compile(src, f"{nb_path.name}#cell-{idx}-{cell.get('id')}", "exec"), ns)
                            plt.close("all")

            self.assertIn("mkp", ns)
            self.assertIn("auto_pipe", ns)
            self.assertIn("summary", ns)
            self.assertIn("wfd_summary", ns)
            self.assertIn("fig_ms", ns)
            self.assertTrue(bool(ns["sky_ext"]["wide_survey"]))  # type: ignore[index]
            self.assertEqual(int(ns["frac_map_order"]), 6)  # type: ignore[arg-type]
            self.assertGreater(float(ns["delta_area"]), 0.0)  # type: ignore[arg-type]

            clean_gal = ns["clean_galaxies"]
            self.assertGreater(len(clean_gal), 1_000)  # type: ignore[arg-type]
            self.assertLessEqual(float(clean_gal["ebv"].max()), 0.205)  # type: ignore[index]

            sinfo = discover_dp2_tap_schema(ns["tap_client"])  # type: ignore[arg-type]
            self.assertEqual(sinfo.visit_detector_table, "dp2.VisitDetector")
        finally:
            os.chdir(orig_cwd)
            plt.show = orig_show
            plt.close("all")
            if not existed_before and outputs_dir.exists():
                shutil.rmtree(outputs_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
