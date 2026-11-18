"""Local (outside-RSP) DP2 SkyKatana mask builder powered by IVOA TAP services.

Why a dedicated TAP workflow is needed outside the RSP
------------------------------------------------------
Inside the Rubin Science Platform (RSP), Notebook 01 reads binary Gen3 Butler datasets
directly (`deepCoadd_*_consolidated_map_*` ``HealSparseMap`` files and ``htm7``-sharded
``the_monster_*`` ``SimpleCatalog`` files). Those binary Butler datasets are not stored
as SQL rows in TAP, and local laptops/servers typically run lightweight Python environments
(such as ``lsdb_env-py313``) without the full EUPS ``lsst_distrib`` stack.

Rubin Observatory exposes its IVOA **Table Access Protocol (TAP)** endpoint externally at
``https://data.lsst.cloud/api/tap`` (authenticated via a user RSP Bearer token with
``read:tap`` scope). Together with public Virtual Observatory services (ESA Gaia DR3 TAP or
``lsdb`` HATS catalogs), every stage of the DP2 ``SkyMaskPipe`` pipeline can be executed
locally on any laptop or workstation:

1. **Stage 1 — ``propmask`` from ``dp2.VisitDetector`` (+ ``ivoa.ObsCore``) & Dust De-reddening**:
   Queries per-detector calibrated visit metadata (``band``, ``expTime``, ``seeing``,
   5-sigma ``magLim``, and 4-corner sky polygons ``llcra..lrcdec``) via RSP TAP, stacks
   per-pixel exposure counts, inverse-variance coadd $5\\sigma$ limiting magnitudes
   ($m_{\\mathrm{coadd}} = 1.25\\log_{10}\\sum_i 10^{0.8 m_i}$), and weighted PSF seeing
   onto HEALPix ``order_sparse``, optionally intersects ``ivoa.ObsCore`` ``lsst.deep_coadd``
   patch polygons, and applies **DP2 Tutorial 311.1** dust de-reddening
   ($m_{b,\\mathrm{lim}}^{\\mathrm{dered}} = m_{b,\\mathrm{lim}}^{\\mathrm{obs}} - R_b E(B-V)$).
2. **Stage 2 — ``dustmask`` from RSP TAP ``dp2.Object.ebv`` or Local SFD98**:
   Queries SFD98 ``ebv`` values from ``dp2.Object`` via TAP (or local ``dustmaps`` / HEALPix
   maps) to build ``EBVProvider`` and ``dustmask`` ($E(B-V) > E(B-V)_{\\max}$, $|b| < b_{\\min}$).
3. **Stage 3 — ``starmask`` + ``boxmask`` via Gaia DR3 TAP -> Synthetic LSST ``radfunction``**:
   Queries public Gaia DR3 TAP (``https://gea.esac.esa.int/tap-server/tap``, no token
   needed) over the buffered footprint, transforms $(G, G_{\\mathrm{BP}}-G_{\\mathrm{RP}})$
   into synthetic apparent LSST ``ugrizy`` AB magnitudes (matching ``the_monster``'s stellar
   locus calibration), evaluates ``skykatana_lsst_radfunction``, and pixelizes circular star
   halos + saturated CCD bleed trails into ``SkyMaskPipe``.
4. **Offline ``SyntheticDP2TapService`` Fallback**:
   When no RSP token is present locally, transparently emulates both RSP DP2 TAP and Gaia DR3
   TAP so notebooks and CI tests run end-to-end offline out of the box.
"""

from __future__ import annotations

import gc
import io
import logging
import math
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import astropy.units as u
import healsparse as hsp
import hpgeom as hpg
import numpy as np
import pandas as pd
import requests
from astropy.io.votable import parse_single_table
from astropy.table import Table
from skykatana import SkyMaskPipe

from .config import BandCriteria, MaskPipelineConfig
from .dust import (
    EBVProvider,
    build_dust_extinction_mask,
    compute_band_extinction_mag,
    get_rubin_band_extinction_rx,
)
from .memory import MemoryGuard
from .monster_stars import make_skykatana_radfunction, skykatana_lsst_radfunction
from .property_mask import PropertyMaskStats

logger = logging.getLogger(__name__)

DEFAULT_RSP_TAP_URL = "https://data.lsst.cloud/api/tap"
DEFAULT_GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap"


@dataclass
class TapSchemaInfo:
    """Resolved DP2 TAP table and column names discovered from ``TAP_SCHEMA``."""

    catalog_schema: str
    visit_table: str
    visit_detector_table: str
    object_table: str
    has_obscore: bool
    vd_has_band: bool
    vd_has_exptime: bool
    vd_visit_col: str
    visit_id_col: str


def resolve_rsp_tap_token(
    token: str | None = None,
    token_file: str | Path | None = None,
) -> str | None:
    """Resolve an external RSP Bearer token from argument, file, or environment safely.

    Checks in order:
      1. Explicit ``token`` string
      2. ``token_file`` path (if provided)
      3. ``$RSP_ACCESS_TOKEN`` or ``$ACCESS_TOKEN`` environment variable
      4. Default ``~/.rsp-tap.token`` if present on disk
    """
    if token and token.strip():
        return token.strip()

    if token_file is not None:
        p = Path(token_file).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"RSP token file not found: {p}")
        val = p.read_text(encoding="utf-8").strip()
        if not val:
            raise ValueError(f"RSP token file is empty: {p}")
        return val

    for env_key in ("RSP_ACCESS_TOKEN", "ACCESS_TOKEN"):
        env_val = os.environ.get(env_key, "").strip()
        if env_val:
            return env_val

    default_path = Path.home() / ".rsp-tap.token"
    if default_path.is_file():
        val = default_path.read_text(encoding="utf-8").strip()
        if val:
            return val

    return None


class TapQueryResult:
    """Uniform wrapper matching ``pyvo``'s ``.to_table()`` result interface."""

    def __init__(self, table: Table) -> None:
        self._table = table

    def to_table(self) -> Table:
        return self._table


class IVOATapClient:
    """Authenticated IVOA TAP 1.1 client using ``pyvo`` (if installed) or ``requests`` + Astropy VOTable.

    Supports both Rubin RSP TAP (``https://data.lsst.cloud/api/tap`` with ``Bearer`` token)
    and public VO TAP endpoints such as ESA Gaia DR3 (``https://gea.esac.esa.int/tap-server/tap``).
    """

    def __init__(
        self,
        tap_url: str = DEFAULT_RSP_TAP_URL,
        *,
        token: str | None = None,
        timeout_sec: float = 120.0,
        prefer_pyvo: bool = True,
    ) -> None:
        self.tap_url = tap_url.rstrip("/")
        self.timeout_sec = float(timeout_sec)
        self.is_synthetic = False
        self._session = requests.Session()
        if token:
            self._session.headers["Authorization"] = f"Bearer {token.strip()}"

        self.pyvo_service: Any | None = None
        if prefer_pyvo:
            try:
                import pyvo  # type: ignore[import-not-found]

                self.pyvo_service = pyvo.dal.TAPService(self.tap_url, session=self._session)
            except ImportError:
                self.pyvo_service = None

    @property
    def backend_name(self) -> str:
        return "pyvo" if self.pyvo_service is not None else "requests+votable"

    def search(
        self,
        adql: str,
        *,
        maxrec: int | None = 500_000,
        async_query: bool = False,
    ) -> TapQueryResult:
        """Execute an ADQL query synchronously or asynchronously and return a ``TapQueryResult``."""
        if self.pyvo_service is not None:
            if async_query:
                job = self.pyvo_service.submit_job(adql, maxrec=maxrec)
                job.run()
                job.wait(phases=["COMPLETED", "ERROR", "ABORTED"])
                job.raise_if_error()
                return TapQueryResult(job.fetch_result().to_table())
            return TapQueryResult(self.pyvo_service.search(adql, maxrec=maxrec).to_table())

        if async_query:
            return TapQueryResult(self._run_http_async(adql, maxrec=maxrec))
        return TapQueryResult(self._run_http_sync(adql, maxrec=maxrec))

    def _run_http_sync(self, adql: str, maxrec: int | None = None) -> Table:
        data: dict[str, Any] = {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "votable",
            "QUERY": adql,
        }
        if maxrec is not None:
            data["MAXREC"] = int(maxrec)
        resp = self._session.post(f"{self.tap_url}/sync", data=data, timeout=self.timeout_sec)
        resp.raise_for_status()
        return self._parse_votable_bytes(resp.content, adql)

    def _run_http_async(self, adql: str, maxrec: int | None = None) -> Table:
        data: dict[str, Any] = {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "votable",
            "PHASE": "RUN",
            "QUERY": adql,
        }
        if maxrec is not None:
            data["MAXREC"] = int(maxrec)
        resp = self._session.post(
            f"{self.tap_url}/async",
            data=data,
            allow_redirects=True,
            timeout=self.timeout_sec,
        )
        resp.raise_for_status()
        job_url = resp.url.rstrip("/")
        if job_url.endswith("/phase"):
            job_url = job_url[: -len("/phase")]

        deadline = time.monotonic() + max(self.timeout_sec, 600.0)
        sleep_s = 0.5
        while time.monotonic() < deadline:
            p_resp = self._session.get(f"{job_url}/phase", timeout=self.timeout_sec)
            p_resp.raise_for_status()
            phase = p_resp.text.strip().upper()
            if "<" in phase:
                root = ET.fromstring(p_resp.content)
                phase = (root.text or "").strip().upper()
            if phase == "COMPLETED":
                r_resp = self._session.get(
                    f"{job_url}/results/result", timeout=self.timeout_sec
                )
                r_resp.raise_for_status()
                return self._parse_votable_bytes(r_resp.content, adql)
            if phase in ("ERROR", "ABORTED"):
                err_resp = self._session.get(f"{job_url}/error", timeout=self.timeout_sec)
                raise RuntimeError(
                    f"TAP async job failed ({phase}): {err_resp.text[:600]} | ADQL={adql}"
                )
            time.sleep(sleep_s)
            sleep_s = min(5.0, sleep_s * 1.4)

        raise TimeoutError(f"TAP async query timed out at {job_url}")

    @staticmethod
    def _parse_votable_bytes(payload: bytes, adql: str) -> Table:
        # Check for UWS/VOTable QUERY_STATUS="ERROR" before parsing table rows
        head_txt = payload[:2500].decode("utf-8", errors="ignore")
        if 'QUERY_STATUS" value="ERROR"' in head_txt or " value='ERROR'" in head_txt:
            raise RuntimeError(f"TAP server returned VOTable ERROR for ADQL={adql!r}:\n{head_txt}")
        vot = parse_single_table(io.BytesIO(payload))
        return vot.to_table(use_names_over_ids=True)


def open_rsp_tap_client(
    tap_url: str = DEFAULT_RSP_TAP_URL,
    *,
    token: str | None = None,
    token_file: str | Path | None = None,
    allow_synthetic_fallback: bool = True,
    synthetic_kwargs: Mapping[str, Any] | None = None,
) -> IVOATapClient | "SyntheticDP2TapService":
    """Open an authenticated RSP TAP client, or fall back cleanly to ``SyntheticDP2TapService``."""
    resolved = resolve_rsp_tap_token(token=token, token_file=token_file)
    if resolved is not None:
        client = IVOATapClient(tap_url=tap_url, token=resolved)
        logger.info(
            "Connected to RSP TAP %s (backend=%s, token=Bearer ***)",
            tap_url,
            client.backend_name,
        )
        return client

    if not allow_synthetic_fallback:
        raise RuntimeError(
            "No RSP access token found. Create a token with `read:tap` scope at "
            "https://data.lsst.cloud -> Account -> Security tokens, save it to "
            "`~/.rsp-tap.token` (chmod 600 ~/.rsp-tap.token) or export `RSP_ACCESS_TOKEN`."
        )

    logger.warning(
        "No RSP access token found (`~/.rsp-tap.token` or `RSP_ACCESS_TOKEN`); "
        "using offline `SyntheticDP2TapService` emulator for local execution."
    )
    kw = dict(synthetic_kwargs) if synthetic_kwargs is not None else {}
    return SyntheticDP2TapService(**kw)


def discover_dp2_tap_schema(
    tap_client: IVOATapClient | "SyntheticDP2TapService",
    preferred_schema: str = "auto",
) -> TapSchemaInfo:
    """Inspect ``TAP_SCHEMA`` to resolve DP2 table & join column names automatically."""
    tbl_res = tap_client.search(
        "SELECT schema_name, table_name FROM TAP_SCHEMA.tables",
        maxrec=5_000,
    ).to_table()
    schemas_map: dict[str, set[str]] = {}
    all_tables_lower: dict[str, str] = {}
    for row in tbl_res:
        sname = str(row["schema_name"]).strip()
        tname = str(row["table_name"]).strip()
        short = tname.split(".")[-1]
        schemas_map.setdefault(sname, set()).add(short)
        all_tables_lower[f"{sname.lower()}.{short.lower()}"] = f"{sname}.{short}"

    if preferred_schema and preferred_schema != "auto":
        cat_schema = preferred_schema
    else:
        cat_schema = "dp2"
        for candidate in ("dp2", "dp2_prep", "dp1", "dp02_dc2_catalogs"):
            if candidate in schemas_map:
                cat_schema = candidate
                break

    short_tables = {t.lower(): t for t in schemas_map.get(cat_schema, set())}
    vd_short = (
        short_tables.get("visitdetector")
        or short_tables.get("ccdvisit")
        or "VisitDetector"
    )
    visit_short = short_tables.get("visit") or "Visit"
    obj_short = short_tables.get("object") or "Object"

    vd_full = f"{cat_schema}.{vd_short}"
    visit_full = f"{cat_schema}.{visit_short}"
    obj_full = f"{cat_schema}.{obj_short}"
    has_obscore = "ivoa.obscore" in all_tables_lower

    # Inspect VisitDetector & Visit columns so joins/columns never fail across DP1/DP2 revisions
    vd_cols: set[str] = set()
    visit_cols: set[str] = set()
    try:
        col_res = tap_client.search(
            f"SELECT table_name, column_name FROM TAP_SCHEMA.columns "
            f"WHERE table_name IN ('{vd_full}', '{visit_full}')",
            maxrec=2_000,
        ).to_table()
        for row in col_res:
            tn = str(row["table_name"]).strip().lower()
            cn = str(row["column_name"]).strip()
            if tn.endswith(vd_short.lower()):
                vd_cols.add(cn)
            elif tn.endswith(visit_short.lower()):
                visit_cols.add(cn)
    except Exception:
        pass

    vd_has_band = ("band" in vd_cols) if vd_cols else True
    vd_has_exptime = ("expTime" in vd_cols) if vd_cols else True
    vd_visit_col = "visitId" if ("visitId" in vd_cols or not vd_cols) else ("visit" if "visit" in vd_cols else "visitId")
    visit_id_col = "visitId" if ("visitId" in visit_cols or not visit_cols) else ("visit" if "visit" in visit_cols else "visitId")

    return TapSchemaInfo(
        catalog_schema=cat_schema,
        visit_table=visit_full,
        visit_detector_table=vd_full,
        object_table=obj_full,
        has_obscore=has_obscore,
        vd_has_band=vd_has_band,
        vd_has_exptime=vd_has_exptime,
        vd_visit_col=vd_visit_col,
        visit_id_col=visit_id_col,
    )


def parse_stcs_region_pixels(s_region: str, nside: int) -> np.ndarray:
    """Rasterize an IVOA STC-S ``POLYGON`` or ``CIRCLE`` string onto HEALPix NESTED pixels."""
    if not s_region or not isinstance(s_region, str):
        return np.empty(0, dtype=np.int64)
    tokens = s_region.strip().split()
    if not tokens:
        return np.empty(0, dtype=np.int64)

    kind = tokens[0].upper()
    nums: list[float] = []
    for tok in tokens[1:]:
        try:
            nums.append(float(tok))
        except ValueError:
            continue

    if kind == "CIRCLE" and len(nums) >= 3:
        return hpg.query_circle(
            int(nside),
            nums[0] % 360.0,
            nums[1],
            nums[2],
            nest=True,
            degrees=True,
        )

    if kind in ("POLYGON", "BOX") and len(nums) >= 6:
        if kind == "BOX" and len(nums) >= 4:
            rc, dc, w_deg, h_deg = nums[:4]
            cos_d = max(math.cos(math.radians(dc)), 0.1)
            hw = 0.5 * w_deg / cos_d
            hh = 0.5 * h_deg
            ra_v = np.array([rc - hw, rc + hw, rc + hw, rc - hw], dtype=np.float64) % 360.0
            dec_v = np.clip(np.array([dc - hh, dc - hh, dc + hh, dc + hh], dtype=np.float64), -89.9, 89.9)
            return _rasterize_convex_quad(nside, ra_v, dec_v)

        ra_v = np.asarray(nums[0::2], dtype=np.float64) % 360.0
        dec_v = np.asarray(nums[1::2], dtype=np.float64)
        # Drop duplicate closing vertex if repeated
        if ra_v.size >= 4 and abs(ra_v[0] - ra_v[-1]) < 1e-7 and abs(dec_v[0] - dec_v[-1]) < 1e-7:
            ra_v = ra_v[:-1]
            dec_v = dec_v[:-1]
        return _rasterize_convex_quad(nside, ra_v, dec_v)

    return np.empty(0, dtype=np.int64)


def _rasterize_convex_quad(nside: int, ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Order polygon vertices counter-clockwise around their tangent centroid and call ``hpgeom.query_polygon``."""
    ra_r = np.deg2rad(ra_deg)
    dec_r = np.deg2rad(dec_deg)
    vx = np.cos(dec_r) * np.cos(ra_r)
    vy = np.cos(dec_r) * np.sin(ra_r)
    vz = np.sin(dec_r)
    cx, cy, cz = float(vx.mean()), float(vy.mean()), float(vz.mean())
    norm = math.sqrt(cx * cx + cy * cy + cz * cz)
    if norm < 1e-8:
        return np.empty(0, dtype=np.int64)
    cx, cy, cz = cx / norm, cy / norm, cz / norm

    # Local East / North unit vectors at centroid
    ra0 = math.atan2(cy, cx)
    dec0 = math.asin(max(-1.0, min(1.0, cz)))
    ex, ey, ez = -math.sin(ra0), math.cos(ra0), 0.0
    nx = -math.sin(dec0) * math.cos(ra0)
    ny = -math.sin(dec0) * math.sin(ra0)
    nz = math.cos(dec0)

    u_east = vx * ex + vy * ey + vz * ez
    v_north = vx * nx + vy * ny + vz * nz
    angles = np.arctan2(v_north, u_east)
    order = np.argsort(angles)
    ra_ord = ra_deg[order] % 360.0
    dec_ord = np.clip(dec_deg[order], -89.999, 89.999)

    try:
        return hpg.query_polygon(int(nside), ra_ord, dec_ord, nest=True, degrees=True)
    except Exception:
        # Fallback to bounding circle around centroid if polygon corners are degenerate
        max_ang_rad = float(
            np.max(np.arccos(np.clip(vx * cx + vy * cy + vz * cz, -1.0, 1.0)))
        )
        return hpg.query_circle(
            int(nside),
            math.degrees(ra0) % 360.0,
            math.degrees(dec0),
            math.degrees(max_ang_rad),
            nest=True,
            degrees=True,
        )


def _adql_spatial_clause(cfg: MaskPipelineConfig, ra_col: str, dec_col: str, pad_deg: float = 0.25) -> str:
    reg = cfg.region
    if reg.mode == "cone":
        assert reg.ra_deg is not None and reg.dec_deg is not None and reg.radius_deg is not None
        rad = float(reg.radius_deg) + float(pad_deg)
        return (
            f"CONTAINS(POINT('ICRS', {ra_col}, {dec_col}), "
            f"CIRCLE('ICRS', {reg.ra_deg:.6f}, {reg.dec_deg:.6f}, {rad:.6f})) = 1"
        )
    if reg.mode == "box":
        assert (
            reg.ra_min is not None
            and reg.ra_max is not None
            and reg.dec_min is not None
            and reg.dec_max is not None
        )
        d0 = max(-90.0, float(reg.dec_min) - pad_deg)
        d1 = min(90.0, float(reg.dec_max) + pad_deg)
        cos_d = max(math.cos(math.radians(0.5 * (d0 + d1))), 0.1)
        p_ra = pad_deg / cos_d
        r0 = float(reg.ra_min) - p_ra
        r1 = float(reg.ra_max) + p_ra
        if 0.0 <= r0 and r1 <= 360.0 and r0 <= r1:
            return f"({ra_col} BETWEEN {r0:.6f} AND {r1:.6f}) AND ({dec_col} BETWEEN {d0:.6f} AND {d1:.6f})"
        r0_m = r0 % 360.0
        r1_m = r1 % 360.0
        return f"(({ra_col} >= {r0_m:.6f}) OR ({ra_col} <= {r1_m:.6f})) AND ({dec_col} BETWEEN {d0:.6f} AND {d1:.6f})"
    return "1 = 1"


def build_tap_ebv_provider(
    tap_client: IVOATapClient | "SyntheticDP2TapService",
    cfg: MaskPipelineConfig,
    schema_info: TapSchemaInfo | None = None,
    nside_ebv: int = 1024,
    max_objects: int = 350_000,
) -> EBVProvider:
    """Build an SFD98 ``EBVProvider`` locally: queries high-resolution ``dp2.Object.ebv``
    via RSP TAP on regional ``cone``/``box`` footprints (matching **DP2 Tutorial 311.1**'s
    6.1' SFD98 column), with automatic fallback to local ``dustmaps`` / ``rubin_scheduler``
    SFD98 maps for full-sky runs.
    """
    if cfg.dust.ebv_map_path:
        return EBVProvider.from_file(cfg.dust.ebv_map_path)

    # For full-sky runs, prefer a local all-sky SFD98 map before sampling millions of Object rows
    if cfg.region.mode == "full" and not getattr(tap_client, "is_synthetic", False):
        local_sfd = EBVProvider._try_from_dustmaps() or EBVProvider._try_from_rubin_sim_data()
        if local_sfd is not None:
            return local_sfd

    # Query SFD98 `ebv` sampled across `dp2.Object` via TAP (Tutorial 311.1's native TAP column!)
    sinfo = schema_info or discover_dp2_tap_schema(tap_client)
    where_spatial = _adql_spatial_clause(cfg, "coord_ra", "coord_dec", pad_deg=0.20)
    adql = (
        f"SELECT TOP {int(max_objects)} coord_ra, coord_dec, ebv "
        f"FROM {sinfo.object_table} "
        f"WHERE {where_spatial} AND ebv IS NOT NULL"
    )
    try:
        tbl = tap_client.search(adql, maxrec=max_objects).to_table()
        if len(tbl) > 0:
            ra = np.asarray(tbl["coord_ra"], dtype=np.float64) % 360.0
            dec = np.asarray(tbl["coord_dec"], dtype=np.float64)
            ebv = np.asarray(tbl["ebv"], dtype=np.float64)
            good = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(ebv) & (ebv >= 0.0)
            if np.any(good):
                # Multi-resolution hierarchical fill (256 -> 512 -> nside_ebv) using conservative
                # per-cell upper-envelope E(B-V) so sharp cirrus contours are never diluted.
                target_nside = int(nside_ebv)
                fill_map = hsp.HealSparseMap.make_empty(32, target_nside, dtype=np.float32)
                ebv_g = ebv[good]
                for level_nside in sorted({256, min(512, target_nside), target_nside}):
                    if level_nside > target_nside:
                        continue
                    lpix = hpg.angle_to_pixel(level_nside, ra[good], dec[good], nest=True, degrees=True)
                    l_uniq, l_inv = np.unique(lpix, return_inverse=True)
                    l_vals = np.zeros(l_uniq.size, dtype=np.float64)
                    np.maximum.at(l_vals, l_inv, ebv_g)
                    shift = 2 * int(round(math.log2(target_nside // level_nside)))
                    if shift > 0:
                        n_sub = 1 << shift
                        exp_pix = (l_uniq[:, None] << shift) + np.arange(n_sub, dtype=np.int64)[None, :]
                        exp_val = np.repeat(l_vals.astype(np.float32)[:, None], n_sub, axis=1)
                        fill_map.update_values_pix(exp_pix.ravel(), exp_val.ravel())
                    else:
                        fill_map.update_values_pix(l_uniq, l_vals.astype(np.float32))
                logger.info(
                    "Built SFD98 E(B-V) HealSparseMap from TAP %s.ebv (%d objects -> %d nside=%d cells)",
                    sinfo.object_table,
                    int(np.sum(good)),
                    int(fill_map.n_valid),
                    target_nside,
                )
                return EBVProvider(source_name=f"tap:{sinfo.object_table}.ebv", hsp_map=fill_map)
    except Exception as exc:
        logger.warning("TAP Object.ebv query unavailable (%s); falling back to Galactic cosecant model.", exc)

    return EBVProvider.from_config(cfg.dust)


def build_tap_property_mask(
    pipe: SkyMaskPipe,
    tap_client: IVOATapClient | "SyntheticDP2TapService",
    cfg: MaskPipelineConfig,
    guard: MemoryGuard,
    *,
    ebv_provider: EBVProvider | None = None,
    schema_info: TapSchemaInfo | None = None,
    use_obscore_coadd_footprint: bool = True,
    output_stage: str = "propmask",
) -> tuple[hsp.HealSparseMap, PropertyMaskStats, dict[str, hsp.HealSparseMap]]:
    """Build ``propmask`` locally from RSP TAP ``VisitDetector`` polygons + ``ObsCore`` + Dust De-reddening!

    Also returns ``coadd_maps`` containing the reconstructed ``HealSparseMap`` per-band coadd
    depth maps (``maglim_{band}_obs``, ``maglim_{band}_dered``, ``nexp_{band}``) for QA plotting.
    """
    pm = cfg.property_maps
    sinfo = schema_info or discover_dp2_tap_schema(tap_client)
    order_sparse = int(pm.order_sparse)
    order_cov = int(pm.order_cov)
    nside_sparse = 1 << order_sparse
    nside_cov = 1 << order_cov

    apply_dust = bool(cfg.dust.enabled and cfg.dust.apply_to_maglim)
    rx_map = (
        get_rubin_band_extinction_rx(r_v=cfg.dust.r_v, overrides=cfg.dust.r_x_overrides)
        if apply_dust
        else {}
    )
    active_ebv = (
        ebv_provider
        if (apply_dust and ebv_provider is not None)
        else (build_tap_ebv_provider(tap_client, cfg, schema_info=sinfo) if apply_dust else None)
    )

    bands_csv = ", ".join(f"'{b}'" for b in pm.bands)
    band_expr = "vd.band" if sinfo.vd_has_band else "v.band"
    exptime_expr = "vd.expTime" if sinfo.vd_has_exptime else "v.expTime"
    join_clause = (
        ""
        if (sinfo.vd_has_band and sinfo.vd_has_exptime)
        else f"JOIN {sinfo.visit_table} AS v ON vd.{sinfo.vd_visit_col} = v.{sinfo.visit_id_col} "
    )
    where_spatial = _adql_spatial_clause(cfg, "vd.ra", "vd.dec", pad_deg=0.30)

    adql_vd = (
        f"SELECT {band_expr} AS band, {exptime_expr} AS expTime, "
        f"vd.seeing AS seeing, vd.magLim AS magLim, vd.ra AS ra, vd.dec AS dec, "
        f"vd.llcra, vd.llcdec, vd.ulcra, vd.ulcdec, vd.urcra, vd.urcdec, vd.lrcra, vd.lrcdec "
        f"FROM {sinfo.visit_detector_table} AS vd "
        f"{join_clause}"
        f"WHERE {band_expr} IN ({bands_csv}) AND ({where_spatial})"
    )
    logger.info("Querying TAP %s detector pointings across bands=%s ...", sinfo.visit_detector_table, pm.bands)
    vd_tbl = tap_client.search(adql_vd, maxrec=1_000_000).to_table()
    logger.info("Retrieved %d detector-visit rows from TAP %s", len(vd_tbl), sinfo.visit_detector_table)

    # Optional IVOA ObsCore deep_coadd patch polygon constraint (calib_level=3)
    obscore_valid_by_band: dict[str, np.ndarray] = {}
    if use_obscore_coadd_footprint and sinfo.has_obscore:
        where_obs = _adql_spatial_clause(cfg, "s_ra", "s_dec", pad_deg=0.45)
        adql_obs = (
            f"SELECT lsst_band, s_region FROM ivoa.ObsCore "
            f"WHERE dataproduct_subtype = 'lsst.deep_coadd' AND calib_level = 3 "
            f"AND lsst_band IN ({bands_csv}) AND ({where_obs})"
        )
        try:
            obs_tbl = tap_client.search(adql_obs, maxrec=200_000).to_table()
            by_b: dict[str, list[np.ndarray]] = {}
            for row in obs_tbl:
                b_str = str(row["lsst_band"]).strip()
                s_reg = str(row["s_region"]).strip()
                p_arr = parse_stcs_region_pixels(s_reg, nside_sparse)
                if p_arr.size > 0:
                    by_b.setdefault(b_str, []).append(p_arr)
            for b_str, chunks in by_b.items():
                obscore_valid_by_band[b_str] = np.unique(np.concatenate(chunks))
        except Exception as exc:
            logger.warning("ObsCore deep_coadd polygon query skipped (%s); using VisitDetector union.", exc)

    # Explicit target region mask (matching RegionSelector cone/box boundaries cleanly)
    region_clip_pix: np.ndarray | None = None
    if cfg.region.mode == "cone":
        assert (
            cfg.region.ra_deg is not None
            and cfg.region.dec_deg is not None
            and cfg.region.radius_deg is not None
        )
        region_clip_pix = np.sort(
            hpg.query_circle(
                nside_sparse,
                float(cfg.region.ra_deg),
                float(cfg.region.dec_deg),
                float(cfg.region.radius_deg),
                nest=True,
                degrees=True,
            )
        )

    vd_df = vd_tbl.to_pandas() if len(vd_tbl) > 0 else pd.DataFrame()
    if "band" in vd_df.columns:
        vd_df["band"] = vd_df["band"].astype(str).str.strip()

    per_band_area: dict[str, float] = {}
    coadd_maps: dict[str, hsp.HealSparseMap] = {}

    # Accumulate multi-band counts in a sparse HealSparse int16 map + required-band bitmask!
    n_pass_map = hsp.HealSparseMap.make_empty(nside_cov, nside_sparse, dtype=np.int16, sentinel=0)
    req_bands = list(pm.required_bands)
    req_pass_maps: dict[str, hsp.HealSparseMap] = {}

    for band in pm.bands:
        crit = pm.per_band.get(band, BandCriteria())
        sub = vd_df[vd_df["band"] == band] if len(vd_df) > 0 else pd.DataFrame()

        if len(sub) == 0:
            per_band_area[band] = 0.0
            empty_b = hsp.HealSparseMap.make_empty(nside_cov, nside_sparse, dtype=np.bool_, bit_packed=True)
            if pm.keep_per_band_stages:
                setattr(pipe, f"prop_{band}_mask", empty_b)
            if band in req_bands:
                req_pass_maps[band] = empty_b
            continue

        pix_chunks: list[np.ndarray] = []
        exp_chunks: list[np.ndarray] = []
        wt_chunks: list[np.ndarray] = []
        see_wt_chunks: list[np.ndarray] = []

        has_corners = all(
            c in sub.columns
            for c in ("llcra", "llcdec", "ulcra", "ulcdec", "urcra", "urcdec", "lrcra", "lrcdec")
        )
        exptimes = sub["expTime"].to_numpy(dtype=np.float64)
        maglims = sub["magLim"].to_numpy(dtype=np.float64)
        seeings = sub["seeing"].to_numpy(dtype=np.float64)
        ras = sub["ra"].to_numpy(dtype=np.float64)
        decs = sub["dec"].to_numpy(dtype=np.float64)

        if has_corners:
            c_ra_mat = np.column_stack(
                [sub["llcra"], sub["ulcra"], sub["urcra"], sub["lrcra"]]
            ).astype(np.float64)
            c_dec_mat = np.column_stack(
                [sub["llcdec"], sub["ulcdec"], sub["urcdec"], sub["lrcdec"]]
            ).astype(np.float64)
        else:
            c_ra_mat = c_dec_mat = np.empty((0, 4))

        for i in range(len(sub)):
            ml_i = maglims[i]
            et_i = exptimes[i] if np.isfinite(exptimes[i]) and exptimes[i] > 0 else float(crit.nominal_exptime_sec)
            if not np.isfinite(ml_i):
                continue

            if has_corners and np.all(np.isfinite(c_ra_mat[i])) and np.all(np.isfinite(c_dec_mat[i])):
                dpix = _rasterize_convex_quad(nside_sparse, c_ra_mat[i], c_dec_mat[i])
            elif np.isfinite(ras[i]) and np.isfinite(decs[i]):
                # Single LSSTCam 4k x 4k x 0.2" CCD is ~0.227 deg x 0.227 deg
                cos_d = max(math.cos(math.radians(decs[i])), 0.1)
                hw = 0.1135 / cos_d
                hh = 0.1135
                dpix = _rasterize_convex_quad(
                    nside_sparse,
                    np.array([ras[i] - hw, ras[i] + hw, ras[i] + hw, ras[i] - hw]),
                    np.array([decs[i] - hh, decs[i] - hh, decs[i] + hh, decs[i] + hh]),
                )
            else:
                continue

            if dpix.size == 0:
                continue
            # Inverse-variance point-source flux weight: sigma_f^-2 proportional to 10^(0.8 * (magLim - 25))
            w_i = 10.0 ** (0.8 * (ml_i - 25.0))
            s_i = seeings[i] if np.isfinite(seeings[i]) else 1.0
            pix_chunks.append(dpix)
            exp_chunks.append(np.full(dpix.size, et_i, dtype=np.float32))
            wt_chunks.append(np.full(dpix.size, w_i, dtype=np.float64))
            see_wt_chunks.append(np.full(dpix.size, w_i * s_i, dtype=np.float64))

        if not pix_chunks:
            per_band_area[band] = 0.0
            empty_b = hsp.HealSparseMap.make_empty(nside_cov, nside_sparse, dtype=np.bool_, bit_packed=True)
            if band in req_bands:
                req_pass_maps[band] = empty_b
            continue

        all_p = np.concatenate(pix_chunks)
        all_et = np.concatenate(exp_chunks)
        all_w = np.concatenate(wt_chunks)
        all_sw = np.concatenate(see_wt_chunks)
        del pix_chunks, exp_chunks, wt_chunks, see_wt_chunks

        uniq_p, inv_idx = np.unique(all_p, return_inverse=True)
        n_exp_pix = np.bincount(inv_idx).astype(np.int32)
        sum_et = np.bincount(inv_idx, weights=all_et)
        sum_w = np.bincount(inv_idx, weights=all_w)
        sum_sw = np.bincount(inv_idx, weights=all_sw)
        del all_p, all_et, all_w, all_sw, inv_idx

        # Restrict to ObsCore deep_coadd patches and/or target RegionSelector if present
        keep_geom = np.ones(uniq_p.size, dtype=bool)
        if band in obscore_valid_by_band:
            keep_geom &= np.isin(uniq_p, obscore_valid_by_band[band], assume_unique=True)
        if region_clip_pix is not None:
            keep_geom &= np.isin(uniq_p, region_clip_pix, assume_unique=True)

        uniq_p = uniq_p[keep_geom]
        n_exp_pix = n_exp_pix[keep_geom]
        sum_et = sum_et[keep_geom]
        sum_w = sum_w[keep_geom]
        sum_sw = sum_sw[keep_geom]

        coadd_maglim_obs = 25.0 + 1.25 * np.log10(np.maximum(sum_w, 1e-30))
        mean_seeing_fwhm = sum_sw / np.maximum(sum_w, 1e-30)

        if apply_dust and active_ebv is not None and uniq_p.size > 0:
            ebv_pix = active_ebv.eval_nest_pixels(nside_sparse, uniq_p)
            a_band = compute_band_extinction_mag(ebv_pix, band=band, r_v=cfg.dust.r_v, rx_map=rx_map)
            coadd_maglim_eval = coadd_maglim_obs - a_band
        else:
            coadd_maglim_eval = coadd_maglim_obs

        # Save diagnostic coadd depth maps for `r` (or primary bands) so notebook cells can plot them!
        if band in ("g", "r", "i") and uniq_p.size > 0:
            m_obs = hsp.HealSparseMap.make_empty(nside_cov, nside_sparse, dtype=np.float32)
            m_obs.update_values_pix(uniq_p, coadd_maglim_obs.astype(np.float32))
            coadd_maps[f"maglim_{band}_obs"] = m_obs
            if apply_dust:
                m_dered = hsp.HealSparseMap.make_empty(nside_cov, nside_sparse, dtype=np.float32)
                m_dered.update_values_pix(uniq_p, coadd_maglim_eval.astype(np.float32))
                coadd_maps[f"maglim_{band}_dered"] = m_dered

        passed = np.ones(uniq_p.size, dtype=bool)
        if crit.min_exposures is not None:
            passed &= n_exp_pix >= int(crit.min_exposures)
        eff_exptime = crit.effective_min_exptime_sec()
        if eff_exptime is not None:
            passed &= sum_et >= float(eff_exptime)
        if crit.min_maglim is not None:
            passed &= coadd_maglim_eval >= float(crit.min_maglim)
        if crit.max_psf_fwhm_arcsec is not None:
            passed &= mean_seeing_fwhm <= float(crit.max_psf_fwhm_arcsec)

        valid_b_pix = uniq_p[passed]
        band_map = hsp.HealSparseMap.make_empty(nside_cov, nside_sparse, dtype=np.bool_, bit_packed=True)
        if valid_b_pix.size > 0:
            band_map.update_values_pix(valid_b_pix, True)
            cur_counts = n_pass_map.get_values_pix(valid_b_pix)
            n_pass_map.update_values_pix(valid_b_pix, (cur_counts + 1).astype(np.int16))

        per_band_area[band] = round(float(band_map.get_valid_area(degrees=True)), 4)
        if pm.keep_per_band_stages:
            if hasattr(pipe, "_finalize_stage"):
                pipe._finalize_stage(
                    band_map,
                    default_name=f"prop_{band}_mask",
                    output_stage=f"prop_{band}_mask",
                    extra_meta={"band": band, "source": f"tap:{sinfo.visit_detector_table}", "area_deg2": per_band_area[band]},
                )
            else:
                setattr(pipe, f"prop_{band}_mask", band_map)
        if band in req_bands:
            req_pass_maps[band] = band_map

        guard.check(f"TAP band {band} stacked ({valid_b_pix.size:,} pixels)")

    # Finalize multi-band K-of-N (`min_filters`) + `required_bands` intersection
    propmask = hsp.HealSparseMap.make_empty(nside_cov, nside_sparse, dtype=np.bool_, bit_packed=True)
    for cov_chunk in n_pass_map.iter_valid_pixels_by_covpix():
        p_arr = np.asarray(cov_chunk, dtype=np.int64)
        if p_arr.size == 0:
            continue
        ok = n_pass_map.get_values_pix(p_arr) >= int(pm.min_filters)
        for rb in req_bands:
            rmap = req_pass_maps.get(rb)
            if rmap is None:
                ok &= False
            else:
                ok &= np.asarray(rmap.get_values_pix(p_arr), dtype=bool)
        if np.any(ok):
            propmask.update_values_pix(p_arr[ok], True)

    del n_pass_map, req_pass_maps
    gc.collect()

    n_valid = int(propmask.n_valid)
    area_deg2 = float(propmask.get_valid_area(degrees=True))
    ebv_src = active_ebv.source_name if (apply_dust and active_ebv is not None) else None

    extra_meta = {
        "source": f"tap:{sinfo.visit_detector_table}",
        "obscore_clipped": bool(obscore_valid_by_band),
        "bands": list(pm.bands),
        "min_filters": int(pm.min_filters),
        "required_bands": req_bands,
        "dust_dereddened_maglim": apply_dust,
        "ebv_source_used": ebv_src,
        "dust_rx_by_band": {b: round(float(v), 4) for b, v in rx_map.items()} if apply_dust else None,
        "pixels": n_valid,
        "area_deg2": area_deg2,
        "per_band_area_deg2": per_band_area,
        "bit_packed": True,
    }
    if hasattr(pipe, "_finalize_stage"):
        pipe._finalize_stage(propmask, default_name="propmask", output_stage=output_stage, extra_meta=extra_meta)
    else:
        setattr(pipe, output_stage, propmask)

    stats = PropertyMaskStats(
        order_sparse=order_sparse,
        order_cov=order_cov,
        nside_sparse=nside_sparse,
        nside_cov=nside_cov,
        bands_evaluated=list(pm.bands),
        min_filters=int(pm.min_filters),
        required_bands=req_bands,
        n_cov_pixels_scanned=int(np.sum(propmask.coverage_mask)),
        n_valid_pixels=n_valid,
        area_deg2=area_deg2,
        per_band_area_deg2=per_band_area,
        dust_dereddened_maglim=apply_dust,
        ebv_source_used=ebv_src,
        dust_rx_by_band={b: round(float(v), 4) for b, v in rx_map.items()} if apply_dust else {},
    )
    return propmask, stats, coadd_maps


def gaia_to_synthetic_lsst_magnitudes(
    phot_g_mean_mag: np.ndarray | Sequence[float],
    bp_rp: np.ndarray | Sequence[float] | None = None,
) -> dict[str, np.ndarray]:
    """Transform Gaia DR3 $(G, G_{\\mathrm{BP}} - G_{\\mathrm{RP}})$ into synthetic apparent LSST
    ``ugrizy`` AB magnitudes along the stellar locus (used locally when ``the_monster`` Butler
    shards are not present).
    """
    g_vega = np.asarray(phot_g_mean_mag, dtype=np.float64)
    if bp_rp is None:
        col = np.full_like(g_vega, 0.82)
    else:
        raw_c = np.asarray(bp_rp, dtype=np.float64)
        col = np.where(np.isfinite(raw_c), np.clip(raw_c, -0.5, 3.5), 0.82)

    return {
        "lsst_u_mag": g_vega + 0.950 + 0.920 * col + 0.180 * (col**2),
        "lsst_g_mag": g_vega + 0.135 + 0.365 * col + 0.085 * (col**2),
        "lsst_r_mag": g_vega + 0.160 - 0.225 * col + 0.080 * (col**2),
        "lsst_i_mag": g_vega + 0.295 - 0.565 * col + 0.075 * (col**2),
        "lsst_z_mag": g_vega + 0.390 - 0.785 * col + 0.075 * (col**2),
        "lsst_y_mag": g_vega + 0.455 - 0.910 * col + 0.075 * (col**2),
    }


def build_local_tap_star_mask(
    pipe: SkyMaskPipe,
    footprint_mask: hsp.HealSparseMap,
    cfg: MaskPipelineConfig,
    guard: MemoryGuard,
    work_dir: Path,
    *,
    gaia_tap_client: IVOATapClient | "SyntheticDP2TapService" | None = None,
    rsp_tap_client: IVOATapClient | "SyntheticDP2TapService" | None = None,
    star_stage: str = "starmask",
    box_stage: str = "boxmask",
) -> dict[str, Any]:
    """Query bright stars via Gaia DR3 TAP (or ``SyntheticDP2TapService``), compute synthetic
    LSST magnitudes, evaluate ``skykatana_lsst_radfunction``, and pixelize ``starmask`` + ``boxmask``.
    """
    ms_cfg = cfg.monster_stars
    work_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = work_dir / "monster_bright_stars_circles.parquet"
    boxes_fits_path = work_dir / "monster_bleed_boxes.fits"

    buffer_deg = float(ms_cfg.max_radius_arcsec) / 3600.0 + 0.05
    if gaia_tap_client is None:
        if isinstance(rsp_tap_client, SyntheticDP2TapService):
            gaia_tap_client = rsp_tap_client
        else:
            gaia_tap_client = IVOATapClient(DEFAULT_GAIA_TAP_URL)

    where_stars = _adql_spatial_clause(cfg, "ra", "dec", pad_deg=buffer_deg)
    g_cut = float(ms_cfg.mag_max) + 0.6  # pad for red/blue G -> LSST color terms
    adql_gaia = (
        f"SELECT ra, dec, phot_g_mean_mag, bp_rp "
        f"FROM gaiadr3.gaia_source "
        f"WHERE ({where_stars}) AND phot_g_mean_mag <= {g_cut:.3f}"
    )

    star_source = "gaia_dr3_tap"
    try:
        g_tbl = gaia_tap_client.search(adql_gaia, maxrec=1_000_000).to_table()
        ra_deg = np.asarray(g_tbl["ra"], dtype=np.float64) % 360.0
        dec_deg = np.asarray(g_tbl["dec"], dtype=np.float64)
        g_mag = np.asarray(g_tbl["phot_g_mean_mag"], dtype=np.float64)
        bp_rp = np.asarray(g_tbl["bp_rp"], dtype=np.float64)
        synth_lsst = gaia_to_synthetic_lsst_magnitudes(g_mag, bp_rp)
    except Exception as exc:
        if rsp_tap_client is None:
            raise
        logger.warning(
            "Public Gaia DR3 TAP unreachable (%s); falling back to RSP TAP Object bright point sources.",
            exc,
        )
        sinfo = discover_dp2_tap_schema(rsp_tap_client)
        where_obj = _adql_spatial_clause(cfg, "coord_ra", "coord_dec", pad_deg=buffer_deg)
        adql_obj = (
            f"SELECT coord_ra AS ra, coord_dec AS dec, g_psfMag, r_psfMag, i_psfMag "
            f"FROM {sinfo.object_table} "
            f"WHERE ({where_obj}) AND refExtendedness = 0 AND r_psfMag <= {float(ms_cfg.mag_max):.3f}"
        )
        o_tbl = rsp_tap_client.search(adql_obj, maxrec=500_000).to_table()
        ra_deg = np.asarray(o_tbl["ra"], dtype=np.float64) % 360.0
        dec_deg = np.asarray(o_tbl["dec"], dtype=np.float64)
        r_m = np.asarray(o_tbl["r_psfMag"], dtype=np.float64)
        synth_lsst = {
            "lsst_g_mag": np.asarray(o_tbl["g_psfMag"], dtype=np.float64),
            "lsst_r_mag": r_m,
            "lsst_i_mag": np.asarray(o_tbl["i_psfMag"], dtype=np.float64),
        }
        star_source = f"rsp_tap:{sinfo.object_table}"

    target_band = "r" if ms_cfg.profile_band in ("r_halo", "multi") else ms_cfg.profile_band
    primary_mag = synth_lsst.get(f"lsst_{target_band}_mag", synth_lsst["lsst_r_mag"])

    # Filter to dilated footprint + [mag_min, mag_max]
    cov_pix = np.where(footprint_mask.coverage_mask)[0]
    nside_cov = footprint_mask.nside_coverage
    dilated_128 = np.zeros(hpg.nside_to_npixel(128), dtype=bool)
    if cov_pix.size > 0:
        c_ra, c_dec = hpg.pixel_to_angle(nside_cov, cov_pix, nest=True, degrees=True)
        tile_rad = math.sqrt(4.0 * math.pi * (180.0 / math.pi) ** 2 / hpg.nside_to_npixel(nside_cov))
        for r0, d0 in zip(c_ra, c_dec):
            dp = hpg.query_circle(128, float(r0), float(d0), tile_rad + buffer_deg, nest=True, degrees=True)
            dilated_128[dp] = True

    keep = (
        np.isfinite(ra_deg)
        & np.isfinite(dec_deg)
        & np.isfinite(primary_mag)
        & (primary_mag >= float(ms_cfg.mag_min))
        & (primary_mag <= float(ms_cfg.mag_max))
    )
    if np.any(keep):
        p128 = hpg.angle_to_pixel(128, ra_deg[keep], dec_deg[keep], nest=True, degrees=True)
        in_foot = dilated_128[p128]
        idx_sel = np.flatnonzero(keep)[in_foot]
    else:
        idx_sel = np.empty(0, dtype=np.int64)

    cols_df: dict[str, np.ndarray] = {
        "ra": ra_deg[idx_sel],
        "dec": dec_deg[idx_sel],
        "mag": primary_mag[idx_sel],
    }
    for k, arr in synth_lsst.items():
        cols_df[k] = arr[idx_sel]
    stars_df = pd.DataFrame(cols_df)

    rad_fn = make_skykatana_radfunction(ms_cfg)
    if len(stars_df) > 0:
        rad_fn(stars_df)
    else:
        stars_df["radius"] = np.empty(0, dtype=np.float64)

    stars_df[["ra", "dec", "mag", "radius"]].to_parquet(parquet_path, index=False)

    order_star = int(ms_cfg.order_sparse)
    order_cov = int(cfg.property_maps.order_cov)
    if len(stars_df) > 0:
        starmask = pipe.build_circ_mask(
            data=stars_df[["ra", "dec", "radius"]],
            order_sparse=order_star,
            order_cov=order_cov,
            columns=["ra", "dec", "radius"],
            bit_packed=True,
            n_threads=int(cfg.memory.skykatana_threads),
            chunk_size=int(cfg.memory.circle_chunk_size),
            output_stage=star_stage,
        )
    else:
        starmask = hsp.HealSparseMap.make_empty(1 << order_cov, 1 << order_star, dtype=np.bool_, bit_packed=True)
        setattr(pipe, star_stage, starmask)

    n_bleed = 0
    box_area = 0.0
    if ms_cfg.enable_bleed_boxes and len(stars_df) > 0:
        bright = stars_df[stars_df["mag"] <= float(ms_cfg.bleed_mag_max)].reset_index(drop=True)
        n_bleed = len(bright)
        if n_bleed > 0:
            Table(
                {
                    "ra_c": bright["ra"].to_numpy(dtype=np.float64),
                    "dec_c": bright["dec"].to_numpy(dtype=np.float64),
                    "width": bright["radius"].to_numpy(dtype=np.float64) * float(ms_cfg.bleed_width_factor),
                    "height": bright["radius"].to_numpy(dtype=np.float64) * float(ms_cfg.bleed_height_factor),
                }
            ).write(boxes_fits_path, format="fits", overwrite=True)
            boxmask = pipe.build_box_mask(
                data=boxes_fits_path,
                order_sparse=order_star,
                order_cov=order_cov,
                fmt="fits",
                columns=["ra_c", "dec_c", "width", "height"],
                bit_packed=True,
                n_threads=int(cfg.memory.skykatana_threads),
                output_stage=box_stage,
            )
            box_area = float(boxmask.get_valid_area(degrees=True))

    guard.check("TAP bright-star & bleed stages pixelized")
    return {
        "star_source": star_source,
        "radius_model": ms_cfg.radius_model,
        "profile_band": ms_cfg.profile_band,
        "completeness": float(ms_cfg.completeness),
        "n_stars_selected": int(len(stars_df)),
        "n_bleed_boxes": int(n_bleed),
        "starmask_area_deg2": round(float(starmask.get_valid_area(degrees=True)), 5),
        "boxmask_area_deg2": round(box_area, 5),
        "circles_parquet_path": str(parquet_path),
        "boxes_fits_path": str(boxes_fits_path) if (n_bleed > 0 and boxes_fits_path.is_file()) else None,
    }


class SyntheticDP2TapService:
    """In-memory emulator of RSP DP2 TAP (``dp2.*``, ``ivoa.ObsCore``) + ESA Gaia DR3 TAP
    (``gaiadr3.gaia_source``) for local offline execution and automated testing.
    """

    def __init__(
        self,
        center_ra_deg: float = 53.1222,
        center_dec_deg: float = -28.0870,
        radius_deg: float = 1.4,
        n_stars: int = 210,
        n_objects: int = 18_000,
        seed: int = 2026,
    ) -> None:
        self.tap_url = "synthetic://data.lsst.cloud/api/tap"
        self.is_synthetic = True
        self.center_ra_deg = float(center_ra_deg)
        self.center_dec_deg = float(center_dec_deg)
        self.radius_deg = float(radius_deg)
        self._populate_tables(n_stars=n_stars, n_objects=n_objects, seed=seed)

    @property
    def backend_name(self) -> str:
        return "synthetic_dp2_tap"

    def _populate_tables(self, n_stars: int, n_objects: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        ra0, dec0, rad = self.center_ra_deg, self.center_dec_deg, self.radius_deg
        cos_d = math.cos(math.radians(dec0))

        # 1. Realistic dithered multi-band `dp2.Visit` & `dp2.VisitDetector` tiles
        nominal_visits = {"u": 7, "g": 12, "r": 16, "i": 16, "z": 11, "y": 9}
        single_visit_maglim = {"u": 23.45, "g": 24.55, "r": 24.20, "i": 23.80, "z": 23.15, "y": 22.05}
        nominal_seeing = {"u": 1.12, "g": 1.02, "r": 0.94, "i": 0.89, "z": 0.87, "y": 0.86}

        vd_rows: list[dict[str, Any]] = []
        visit_rows: list[dict[str, Any]] = []
        visit_id = 2025050001

        # Build a 4x4 focal-plane grid of ~0.68 deg detectors per dithered pointing
        det_half_deg = 0.34
        grid_offsets = (-0.88, -0.29, 0.29, 0.88)

        for b_idx, (band, n_vis) in enumerate(nominal_visits.items()):
            # Slight per-band dither scale variation so K-of-N (`min_filters`) border roll-off is realistic
            dither_scale = 0.42 * (1.0 - 0.04 * (b_idx % 3))
            for v_k in range(n_vis):
                ang = 2.0 * math.pi * v_k / max(1, n_vis)
                d_ra = (dither_scale * math.cos(ang)) / cos_d
                d_dec = dither_scale * math.sin(ang)
                v_ra = (ra0 + d_ra) % 360.0
                v_dec = dec0 + d_dec
                seeing_v = float(np.clip(nominal_seeing[band] + rng.normal(0.0, 0.07), 0.65, 1.45))
                visit_rows.append(
                    {"visitId": visit_id, "band": band, "ra": v_ra, "dec": v_dec, "expTime": 30.0}
                )
                det_num = 0
                for gx in grid_offsets:
                    for gy in grid_offsets:
                        if math.hypot(gx, gy) > 1.26:
                            continue  # circularized LSSTCam focal-plane footprint!
                        rc = (v_ra + gx / cos_d) % 360.0
                        dc = v_dec + gy
                        hw = det_half_deg / max(math.cos(math.radians(dc)), 0.1)
                        hh = det_half_deg
                        ml = float(single_visit_maglim[band] + rng.normal(0.0, 0.06))
                        vd_rows.append(
                            {
                                "visitId": visit_id,
                                "detector": det_num,
                                "band": band,
                                "expTime": 30.0,
                                "seeing": seeing_v,
                                "magLim": ml,
                                "ra": rc,
                                "dec": dc,
                                "llcra": (rc - hw) % 360.0,
                                "llcdec": dc - hh,
                                "ulcra": (rc - hw) % 360.0,
                                "ulcdec": dc + hh,
                                "urcra": (rc + hw) % 360.0,
                                "urcdec": dc + hh,
                                "lrcra": (rc + hw) % 360.0,
                                "lrcdec": dc - hh,
                            }
                        )
                        det_num += 1
                visit_id += 1

        self._visit_tbl = Table(rows=visit_rows)
        self._vd_tbl = Table(rows=vd_rows)

        # 2. `ivoa.ObsCore` `lsst.deep_coadd` patch polygons covering the stacked tract cells
        obs_rows: list[dict[str, Any]] = []
        patch_centers = np.linspace(-1.15, 1.15, 6)
        ph = 0.24
        for band in nominal_visits:
            for px in patch_centers:
                for py in patch_centers:
                    if math.hypot(px, py) > rad * 1.05:
                        continue
                    prc = (ra0 + px / cos_d) % 360.0
                    pdc = dec0 + py
                    phw = ph / max(math.cos(math.radians(pdc)), 0.1)
                    s_reg = (
                        f"POLYGON ICRS "
                        f"{(prc - phw) % 360.0:.6f} {pdc - ph:.6f} "
                        f"{(prc + phw) % 360.0:.6f} {pdc - ph:.6f} "
                        f"{(prc + phw) % 360.0:.6f} {pdc + ph:.6f} "
                        f"{(prc - phw) % 360.0:.6f} {pdc + ph:.6f}"
                    )
                    obs_rows.append(
                        {
                            "obs_collection": "LSST.DP2",
                            "dataproduct_type": "image",
                            "dataproduct_subtype": "lsst.deep_coadd",
                            "calib_level": 3,
                            "lsst_band": band,
                            "s_ra": prc,
                            "s_dec": pdc,
                            "s_region": s_reg,
                        }
                    )
        self._obscore_tbl = Table(rows=obs_rows)

        # 3. `dp2.Object` table with SFD98 cirrus cloud (`ebv`), galaxies (`refExtendedness=1`), and stars
        u_r = np.sqrt(rng.uniform(0.0, 1.0, size=n_objects)) * (rad * 1.08)
        theta = rng.uniform(0.0, 2.0 * math.pi, size=n_objects)
        obj_dec = dec0 + u_r * np.sin(theta)
        obj_ra = (ra0 + (u_r * np.cos(theta)) / cos_d) % 360.0

        dra_n = ((obj_ra - ra0 + 180.0) % 360.0 - 180.0) * cos_d / rad
        ddec_n = (obj_dec - dec0) / rad
        ebv = (
            0.018
            + 0.24 * np.exp(-(((dra_n - 0.42) / 0.27) ** 2 + ((ddec_n - 0.35) / 0.22) ** 2))
            + 0.045 * np.clip(0.5 * (dra_n + ddec_n) + 0.5, 0.0, 1.5)
        )
        g_intr = rng.uniform(20.2, 25.0, size=n_objects)
        r_intr = g_intr - rng.uniform(0.3, 0.9, size=n_objects)
        i_intr = r_intr - rng.uniform(0.15, 0.45, size=n_objects)
        # Observed photometry suffers Galactic reddening A_b = R_b * E(B-V)!
        rx = get_rubin_band_extinction_rx(r_v=3.1)
        self._object_tbl = Table(
            {
                "objectId": np.arange(5_000_000, 5_000_000 + n_objects, dtype=np.int64),
                "coord_ra": obj_ra,
                "coord_dec": obj_dec,
                "ebv": ebv.astype(np.float32),
                "refExtendedness": np.ones(n_objects, dtype=np.int16),
                "g_cModelMag": (g_intr + rx["g"] * ebv).astype(np.float32),
                "r_cModelMag": (r_intr + rx["r"] * ebv).astype(np.float32),
                "i_cModelMag": (i_intr + rx["i"] * ebv).astype(np.float32),
                "g_psfMag": (g_intr + rx["g"] * ebv).astype(np.float32),
                "r_psfMag": (r_intr + rx["r"] * ebv).astype(np.float32),
                "i_psfMag": (i_intr + rx["i"] * ebv).astype(np.float32),
            }
        )

        # 4. `gaiadr3.gaia_source` bright foreground stars (including saturated stars with bleed trails)
        su_r = np.sqrt(rng.uniform(0.0, 1.0, size=n_stars)) * (rad * 1.15)
        stheta = rng.uniform(0.0, 2.0 * math.pi, size=n_stars)
        s_dec = dec0 + su_r * np.sin(stheta)
        s_ra = (ra0 + (su_r * np.cos(stheta)) / cos_d) % 360.0
        s_g = np.concatenate(
            [
                np.array([9.1, 10.2, 11.1, 11.8], dtype=np.float64),
                rng.uniform(12.2, 18.8, size=max(0, n_stars - 4)),
            ]
        )[:n_stars]
        s_bprp = rng.uniform(0.35, 2.1, size=n_stars)
        self._gaia_tbl = Table(
            {
                "ra": s_ra,
                "dec": s_dec,
                "phot_g_mean_mag": s_g,
                "bp_rp": s_bprp,
            }
        )

    def search(
        self,
        adql: str,
        *,
        maxrec: int | None = 500_000,
        async_query: bool = False,
    ) -> TapQueryResult:
        del async_query
        q = " ".join(adql.strip().split())
        q_low = q.lower()

        if "tap_schema.tables" in q_low:
            return TapQueryResult(
                Table(
                    rows=[
                        {"schema_name": "dp2", "table_name": "dp2.Visit"},
                        {"schema_name": "dp2", "table_name": "dp2.VisitDetector"},
                        {"schema_name": "dp2", "table_name": "dp2.Object"},
                        {"schema_name": "ivoa", "table_name": "ivoa.ObsCore"},
                    ]
                )
            )

        if "tap_schema.columns" in q_low:
            rows = [
                {"table_name": "dp2.VisitDetector", "column_name": c}
                for c in self._vd_tbl.colnames
            ] + [
                {"table_name": "dp2.Visit", "column_name": c}
                for c in self._visit_tbl.colnames
            ]
            return TapQueryResult(Table(rows=rows))

        if "visitdetector" in q_low:
            tbl = self._vd_tbl
            m_in = re.search(r"band\s+in\s*\(([^)]+)\)", q_low)
            if m_in:
                wanted = {tok.strip().strip("'\"") for tok in m_in.group(1).split(",")}
                mask = np.isin(np.asarray(tbl["band"], dtype=str), list(wanted))
                tbl = tbl[mask]
            return TapQueryResult(tbl[:maxrec] if maxrec else tbl)

        if "ivoa.obscore" in q_low:
            tbl = self._obscore_tbl
            m_in = re.search(r"lsst_band\s+in\s*\(([^)]+)\)", q_low)
            if m_in:
                wanted = {tok.strip().strip("'\"") for tok in m_in.group(1).split(",")}
                mask = np.isin(np.asarray(tbl["lsst_band"], dtype=str), list(wanted))
                tbl = tbl[mask]
            return TapQueryResult(tbl[:maxrec] if maxrec else tbl)

        if "gaiadr3.gaia_source" in q_low:
            tbl = self._gaia_tbl
            m_mag = re.search(r"phot_g_mean_mag\s*<=\s*([0-9.]+)", q_low)
            if m_mag:
                cut = float(m_mag.group(1))
                tbl = tbl[np.asarray(tbl["phot_g_mean_mag"]) <= cut]
            return TapQueryResult(tbl[:maxrec] if maxrec else tbl)

        if ".object" in q_low:
            tbl = self._object_tbl
            m_top = re.search(r"select\s+top\s+(\d+)", q_low)
            lim = int(m_top.group(1)) if m_top else maxrec
            return TapQueryResult(tbl[:lim] if lim else tbl)

        raise ValueError(f"Unsupported ADQL query in SyntheticDP2TapService: {adql}")
