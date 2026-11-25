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
``lsdb`` HATS catalogs), every stage of the DP2 ``SkyMaskPipe`` pipeline — from regional
pilots up to the **complete full-survey DP2 footprint** — can be executed locally:

1. **Stage 1 — ``propmask`` from ``dp2.VisitDetector`` (+ ``ivoa.ObsCore``) & Dust De-reddening**:
   Queries per-detector calibrated visit metadata (``band``, ``expTime``, ``seeing``,
   5-sigma ``magLim``, and 4-corner sky polygons ``llcra..lrcdec``) via RSP TAP (streaming
   in RA stripes for ``region.mode == 'full'``), stacks per-pixel exposure counts,
   inverse-variance coadd $5\\sigma$ limiting magnitudes
   ($m_{\\mathrm{coadd}} = 1.25\\log_{10}\\sum_i 10^{0.8 m_i}$), and weighted PSF seeing
   onto HEALPix, optionally intersects ``ivoa.ObsCore`` ``lsst.deep_coadd`` patch polygons,
   and applies **DP2 Tutorial 311.1** dust de-reddening
   ($m_{b,\\mathrm{lim}}^{\\mathrm{dered}} = m_{b,\\mathrm{lim}}^{\\mathrm{obs}} - R_b E(B-V)$).
2. **Stage 2 — ``dustmask`` from RSP TAP ``dp2.Object.ebv`` or Local SFD98**:
   Queries SFD98 ``ebv`` values from ``dp2.Object`` via TAP (or local ``dustmaps`` / HEALPix
   maps) to build ``EBVProvider`` and ``dustmask`` ($E(B-V) > E(B-V)_{\\max}$, $|b| < b_{\\min}$).
3. **Stage 3 — ``starmask`` + ``boxmask`` via Gaia DR3 TAP -> Synthetic LSST ``radfunction``**:
   Queries public Gaia DR3 TAP (``https://gea.esac.esa.int/tap-server/tap``, chunked over
   occupied sky tiles for full-survey DP2 runs), transforms $(G, G_{\\mathrm{BP}}-G_{\\mathrm{RP}})$
   into synthetic apparent LSST ``ugrizy`` AB magnitudes, evaluates ``skykatana_lsst_radfunction``,
   and pixelizes circular star halos + saturated CCD bleed trails into ``SkyMaskPipe``.
4. **One-Command Production Pipeline (``DP2LocalTapMaskBuilder``)**:
   Drop-in local counterpart to ``DP2MaskBuilder`` producing identical ``MaskBuildSummary``,
   ``skykatana_pipe/``, ``.hs``, IVOA MOC ``.fits``, ``fracdet``, randoms ``.parquet``, and
   multi-scale QA figures (with automatic offline ``SyntheticDP2TapService`` fallback).
"""

from __future__ import annotations

import gc
import io
import logging
import math
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import healsparse as hsp
import hpgeom as hpg
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from astropy.io.votable import parse_single_table
from astropy.table import Table
from skykatana import SkyMaskPipe

from .config import BandCriteria, MaskPipelineConfig
from .dust import (
    DustMaskStats,
    EBVProvider,
    _icrs_to_galactic_lat_rad,
    build_dust_extinction_mask,
    compute_band_extinction_mag,
    get_rubin_band_extinction_rx,
)
from .memory import (
    MemoryGuard,
    configure_rsp_thread_env,
    estimate_working_set_gib,
)
from .monster_stars import (
    MonsterStarMaskStats,
    build_oriented_bleed_box_mask,
    make_skykatana_radfunction,
)
from .pipeline import FULL_SKY_AREA_DEG2, MaskBuildSummary, _render_qa_plot
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
    """Resolve an external RSP Bearer token from argument, file, or environment safely."""
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
    """Authenticated IVOA TAP 1.1 client using ``pyvo`` (if installed) or ``requests`` + Astropy VOTable."""

    def __init__(
        self,
        tap_url: str = DEFAULT_RSP_TAP_URL,
        *,
        token: str | None = None,
        timeout_sec: float = 180.0,
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
        """Execute an ADQL query synchronously or asynchronously and return a ``TapQueryResult``.

        Automatically promotes synchronous queries that hit a server-side synchronous
        execution timeout (`Query timeout exceeded` / HTTP 504 / ReadTimeout) to UWS `/async`.
        """
        if self.pyvo_service is not None:
            if async_query:
                job = self.pyvo_service.submit_job(adql, maxrec=maxrec)
                job.run()
                job.wait(phases=["COMPLETED", "ERROR", "ABORTED"])
                job.raise_if_error()
                return TapQueryResult(job.fetch_result().to_table())
            try:
                return TapQueryResult(self.pyvo_service.search(adql, maxrec=maxrec).to_table())
            except Exception as exc:
                msg = str(exc).lower()
                if "timeout" in msg or "/async" in msg or "504" in msg:
                    logger.info("Sync TAP query timed out; promoting automatically to UWS /async ...")
                    job = self.pyvo_service.submit_job(adql, maxrec=maxrec)
                    job.run()
                    job.wait(phases=["COMPLETED", "ERROR", "ABORTED"])
                    job.raise_if_error()
                    return TapQueryResult(job.fetch_result().to_table())
                raise

        if async_query:
            return TapQueryResult(self._run_http_async(adql, maxrec=maxrec))

        try:
            return TapQueryResult(self._run_http_sync(adql, maxrec=maxrec))
        except (requests.exceptions.Timeout, RuntimeError, requests.exceptions.HTTPError) as exc:
            msg = str(exc).lower()
            if (
                isinstance(exc, requests.exceptions.Timeout)
                or "timeout" in msg
                or "/async" in msg
                or "504" in msg
            ):
                logger.info("Sync TAP query hit timeout; promoting automatically to UWS /async ...")
                return TapQueryResult(self._run_http_async(adql, maxrec=maxrec))
            raise

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

    def _normalize_uws_url(self, raw_url: str, base_url: str) -> str:
        resolved = urllib.parse.urljoin(base_url, raw_url.strip()).rstrip("/")
        if self.tap_url.startswith("https://") and resolved.startswith("http://"):
            tap_host = urllib.parse.urlparse(self.tap_url).netloc
            res_host = urllib.parse.urlparse(resolved).netloc
            if res_host == tap_host:
                resolved = "https://" + resolved[len("http://") :]
        return resolved

    def _read_uws_phase(self, job_url: str) -> str:
        p_resp = self._session.get(f"{job_url}/phase", timeout=self.timeout_sec)
        p_resp.raise_for_status()
        raw = p_resp.text.strip()
        if raw.startswith("<"):
            root = ET.fromstring(p_resp.content)
            if root.tag.lower().endswith("phase") and root.text:
                return root.text.strip().upper()
            for elem in root.iter():
                if elem.tag.lower().endswith("phase") and elem.text:
                    return elem.text.strip().upper()
            return (root.text or "").strip().upper()
        return raw.upper()

    def _fetch_uws_result_bytes(self, job_url: str) -> bytes:
        tap_host = urllib.parse.urlparse(self.tap_url).netloc
        dl_timeout = max(self.timeout_sec, 300.0)
        target_url = f"{job_url}/results/result"

        r_resp = self._session.get(target_url, allow_redirects=False, timeout=dl_timeout)
        if r_resp.status_code == 404:
            res_xml = self._session.get(f"{job_url}/results", timeout=self.timeout_sec)
            res_xml.raise_for_status()
            root = ET.fromstring(res_xml.content)
            href: str | None = None
            for elem in root.iter():
                for attr_k, attr_v in elem.attrib.items():
                    if attr_k.lower().endswith("href") and attr_v:
                        href = attr_v
                        break
                if href:
                    break
            if not href:
                r_resp.raise_for_status()
            else:
                next_host = urllib.parse.urlparse(href).netloc
                if not next_host or next_host == tap_host:
                    u_norm = self._normalize_uws_url(href, f"{job_url}/results/")
                    r_resp = self._session.get(u_norm, allow_redirects=False, timeout=dl_timeout)
                else:
                    # External signed object-store URL (e.g. GCS on Rubin RSP): omit Bearer header!
                    r_resp = requests.get(href, allow_redirects=True, timeout=dl_timeout)
                    r_resp.raise_for_status()
                    return r_resp.content

        for _ in range(6):
            if r_resp.status_code not in (301, 302, 303, 307, 308):
                break
            loc = r_resp.headers.get("Location")
            if not loc:
                break
            next_url = urllib.parse.urljoin(r_resp.url, loc)
            next_host = urllib.parse.urlparse(next_url).netloc
            if next_host == tap_host:
                next_url = self._normalize_uws_url(next_url, r_resp.url)
                r_resp = self._session.get(next_url, allow_redirects=False, timeout=dl_timeout)
            else:
                # External cloud-storage signed URL (GCS / S3): never forward RSP Bearer header
                ext_resp = requests.get(next_url, allow_redirects=True, timeout=dl_timeout)
                ext_resp.raise_for_status()
                return ext_resp.content

        r_resp.raise_for_status()
        return r_resp.content

    def _run_http_async(self, adql: str, maxrec: int | None = None) -> Table:
        """Execute strict two-step IVOA UWS 1.1 `/async` job (compatible with Rubin CADC TAP & ESA Gaia TAP)."""
        create_data: dict[str, Any] = {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "votable",
            "QUERY": adql,
        }
        if maxrec is not None:
            create_data["MAXREC"] = int(maxrec)

        resp = self._session.post(
            f"{self.tap_url}/async",
            data=create_data,
            allow_redirects=False,
            timeout=self.timeout_sec,
        )
        if resp.status_code in (301, 302, 303, 307, 308) and "Location" in resp.headers:
            job_url = self._normalize_uws_url(resp.headers["Location"], f"{self.tap_url}/async/")
        else:
            resp.raise_for_status()
            job_url = self._normalize_uws_url(resp.url, f"{self.tap_url}/async/")
            if job_url.endswith("/async"):
                root = ET.fromstring(resp.content)
                job_id = None
                for elem in root.iter():
                    if elem.tag.lower().endswith("jobid") and elem.text:
                        job_id = elem.text.strip()
                        break
                if not job_id:
                    raise RuntimeError(f"Could not extract UWS jobId from /async response: {resp.text[:400]}")
                job_url = f"{self.tap_url}/async/{job_id}"

        for suffix in ("/phase", "/results"):
            if job_url.endswith(suffix):
                job_url = job_url[: -len(suffix)]

        try:
            phase = self._read_uws_phase(job_url)
            if phase in ("PENDING", ""):
                run_resp = self._session.post(
                    f"{job_url}/phase",
                    data={"PHASE": "RUN"},
                    allow_redirects=False,
                    timeout=self.timeout_sec,
                )
                if run_resp.status_code >= 400:
                    run_resp.raise_for_status()

            deadline = time.monotonic() + max(self.timeout_sec * 6.0, 1200.0)
            sleep_s = 0.4
            while time.monotonic() < deadline:
                phase = self._read_uws_phase(job_url)
                if phase == "COMPLETED":
                    payload = self._fetch_uws_result_bytes(job_url)
                    return self._parse_votable_bytes(payload, adql)
                if phase in ("ERROR", "ABORTED"):
                    err_txt = ""
                    try:
                        err_resp = self._session.get(f"{job_url}/error", timeout=30.0)
                        err_txt = err_resp.text[:800]
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"TAP async job {job_url} ended in {phase}: {err_txt} | ADQL={adql}"
                    )
                time.sleep(sleep_s)
                sleep_s = min(3.0, sleep_s * 1.35)

            raise TimeoutError(f"TAP async query timed out waiting for completion at {job_url}")
        finally:
            try:
                self._session.delete(job_url, timeout=10.0)
            except Exception:
                pass

    @staticmethod
    def _parse_votable_bytes(payload: bytes, adql: str) -> Table:
        head_txt = payload[:2500].decode("utf-8", errors="ignore")
        if 'QUERY_STATUS" value="ERROR"' in head_txt or " value='ERROR'" in head_txt:
            raise RuntimeError(f"TAP server returned VOTable ERROR for ADQL={adql!r}:\n{head_txt}")
        try:
            vot = parse_single_table(io.BytesIO(payload))
            return vot.to_table(use_names_over_ids=True)
        except IndexError:
            # Valid VOTable response with zero <TABLE> elements
            return Table()


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
    vd_short = short_tables.get("visitdetector") or short_tables.get("ccdvisit") or "VisitDetector"
    visit_short = short_tables.get("visit") or "Visit"
    obj_short = short_tables.get("object") or "Object"

    vd_full = f"{cat_schema}.{vd_short}"
    visit_full = f"{cat_schema}.{visit_short}"
    obj_full = f"{cat_schema}.{obj_short}"
    has_obscore = "ivoa.obscore" in all_tables_lower

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
    vd_visit_col = (
        "visitId"
        if ("visitId" in vd_cols or not vd_cols)
        else ("visit" if "visit" in vd_cols else "visitId")
    )
    visit_id_col = (
        "visitId"
        if ("visitId" in visit_cols or not visit_cols)
        else ("visit" if "visit" in visit_cols else "visitId")
    )

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

    if kind in ("POLYGON", "BOX") and len(nums) >= 4:
        if kind == "BOX" and len(nums) >= 4:
            rc, dc, w_deg, h_deg = nums[:4]
            cos_d = max(math.cos(math.radians(dc)), 0.1)
            hw = 0.5 * w_deg / cos_d
            hh = 0.5 * h_deg
            ra_v = np.array([rc - hw, rc + hw, rc + hw, rc - hw], dtype=np.float64) % 360.0
            dec_v = np.clip(
                np.array([dc - hh, dc - hh, dc + hh, dc + hh], dtype=np.float64), -89.9, 89.9
            )
            return _rasterize_convex_quad(nside, ra_v, dec_v)

        if len(nums) >= 6:
            ra_v = np.asarray(nums[0::2], dtype=np.float64) % 360.0
            dec_v = np.asarray(nums[1::2], dtype=np.float64)
            if (
                ra_v.size >= 4
                and abs(ra_v[0] - ra_v[-1]) < 1e-7
                and abs(dec_v[0] - dec_v[-1]) < 1e-7
            ):
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


def _adql_spatial_clause(
    cfg: MaskPipelineConfig, ra_col: str, dec_col: str, pad_deg: float = 0.25
) -> str:
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
    via RSP TAP on regional ``cone``/``box`` footprints (matching **DP2 Tutorial 311.1**),
    with automatic support for local ``dustmaps`` / ``rubin_scheduler`` SFD98 maps on
    full-survey runs.
    """
    if cfg.dust.ebv_map_path:
        return EBVProvider.from_file(cfg.dust.ebv_map_path)

    if cfg.region.mode == "full" and not getattr(tap_client, "is_synthetic", False):
        local_sfd = EBVProvider._try_from_dustmaps() or EBVProvider._try_from_rubin_sim_data()
        if local_sfd is not None:
            return local_sfd

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
        logger.warning("TAP Object.ebv query unavailable (%s); falling back to local/analytic model.", exc)

    return EBVProvider.from_config(cfg.dust)


def _order_quads_ccw_batch(
    ra_mat: np.ndarray, dec_mat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized counter-clockwise tangent-plane vertex ordering across ``(N, 4)`` CCD quads."""
    ra_r = np.deg2rad(ra_mat)
    dec_r = np.deg2rad(dec_mat)
    cos_d = np.cos(dec_r)
    vx = cos_d * np.cos(ra_r)
    vy = cos_d * np.sin(ra_r)
    vz = np.sin(dec_r)

    cx = vx.mean(axis=1)
    cy = vy.mean(axis=1)
    cz = vz.mean(axis=1)
    norm = np.maximum(np.sqrt(cx * cx + cy * cy + cz * cz), 1e-12)
    cx /= norm
    cy /= norm
    cz /= norm

    ra0 = np.arctan2(cy, cx)
    dec0 = np.arcsin(np.clip(cz, -1.0, 1.0))
    sin_ra0 = np.sin(ra0)
    cos_ra0 = np.cos(ra0)
    sin_dec0 = np.sin(dec0)
    cos_dec0 = np.cos(dec0)

    u_east = -vx * sin_ra0[:, None] + vy * cos_ra0[:, None]
    v_north = (
        -vx * (sin_dec0 * cos_ra0)[:, None]
        - vy * (sin_dec0 * sin_ra0)[:, None]
        + vz * cos_dec0[:, None]
    )
    order = np.argsort(np.arctan2(v_north, u_east), axis=1)
    rows = np.arange(ra_mat.shape[0])[:, None]
    return np.mod(ra_mat[rows, order], 360.0), np.clip(dec_mat[rows, order], -89.999, 89.999)


def _plan_full_dp2_ra_stripes(
    tap_client: IVOATapClient | "SyntheticDP2TapService",
    sinfo: TapSchemaInfo,
    bands: Sequence[str],
    target_visits_per_stripe: int = 1_200,
) -> list[tuple[float, float]]:
    """Query lightweight ``dp2.Visit.ra`` once to build visit-balanced RA stripes across ``[0, 360)``.

    Skips completely empty RA intervals and automatically subdivides ultra-dense Galactic
    Bulge / WFD zones (such as ``RA = [270, 300)``) into narrow stripes of ~1,200 visits
    (~225k CCD rows) so every UWS ``/async`` query completes quickly and well below ``MAXREC``.
    """
    default_stripes = [(float(lo), float(lo + 15.0)) for lo in range(0, 360, 15)]
    if getattr(tap_client, "is_synthetic", False):
        return [(0.0, 360.0)]

    bands_csv = ", ".join(f"'{b}'" for b in bands)
    try:
        v_adql = f"SELECT ra FROM {sinfo.visit_table} WHERE band IN ({bands_csv})"
        v_tbl = tap_client.search(v_adql, maxrec=500_000, async_query=True).to_table()
        if len(v_tbl) == 0:
            return default_stripes
        v_ra = np.sort(np.asarray(v_tbl["ra"], dtype=np.float64) % 360.0)
        v_ra = v_ra[np.isfinite(v_ra)]
        if v_ra.size == 0:
            return default_stripes
    except Exception as exc:
        logger.warning("Visit RA pre-scan unavailable (%s); using uniform 15-deg RA stripes.", exc)
        return default_stripes

    planned: list[tuple[float, float]] = []
    fov_pad = 3.0  # LSSTCam 3.5 deg FOV half-width + margin for high-|Dec| cos(dec)
    for base_lo, base_hi in default_stripes:
        # Check if any visit center lies within +/- fov_pad of [base_lo, base_hi) (mod 360)
        mid = 0.5 * (base_lo + base_hi)
        half_span = 0.5 * (base_hi - base_lo) + fov_pad
        d_ang = np.abs(((v_ra - mid + 180.0) % 360.0) - 180.0)
        if not np.any(d_ang <= half_span):
            continue

        in_bin = v_ra[(v_ra >= base_lo) & (v_ra < base_hi)]
        n_sub = max(1, int(math.ceil(in_bin.size / max(200, int(target_visits_per_stripe)))))
        if n_sub == 1 or in_bin.size < 2:
            planned.append((base_lo, base_hi))
        else:
            q_edges = np.quantile(in_bin, np.linspace(0.0, 1.0, n_sub + 1))
            q_edges[0] = base_lo
            q_edges[-1] = base_hi
            for k in range(n_sub):
                lo_k = round(float(q_edges[k]), 5)
                hi_k = float(base_hi) if k == n_sub - 1 else round(float(q_edges[k + 1]), 5)
                if hi_k > lo_k + 1e-4:
                    planned.append((lo_k, hi_k))

    logger.info(
        "Planned %d visit-balanced RA stripes across %d DP2 visits (target <= %d visits/stripe)",
        len(planned),
        int(v_ra.size),
        int(target_visits_per_stripe),
    )
    return planned if planned else default_stripes


def _iter_visit_detector_tables(
    tap_client: IVOATapClient | "SyntheticDP2TapService",
    cfg: MaskPipelineConfig,
    sinfo: TapSchemaInfo,
    bands: Sequence[str],
) -> Iterator[pd.DataFrame]:
    """Stream ``dp2.VisitDetector`` chunks: single query for ``cone``/``box``, or visit-balanced
    UWS ``/async`` RA stripes (with automatic bisection if ``MAXREC`` is ever reached) for
    ``region.mode == 'full'`` so multi-million-row DP2 surveys never time out or truncate.
    """
    bands_csv = ", ".join(f"'{b}'" for b in bands)
    band_expr = "vd.band" if sinfo.vd_has_band else "v.band"
    exptime_expr = "vd.expTime" if sinfo.vd_has_exptime else "v.expTime"
    join_clause = (
        ""
        if (sinfo.vd_has_band and sinfo.vd_has_exptime)
        else f"JOIN {sinfo.visit_table} AS v ON vd.{sinfo.vd_visit_col} = v.{sinfo.visit_id_col} "
    )

    select_cols = (
        f"SELECT {band_expr} AS band, {exptime_expr} AS expTime, "
        f"vd.seeing AS seeing, vd.magLim AS magLim, vd.ra AS ra, vd.dec AS dec, "
        f"vd.llcra, vd.llcdec, vd.ulcra, vd.ulcdec, vd.urcra, vd.urcdec, vd.lrcra, vd.lrcdec "
        f"FROM {sinfo.visit_detector_table} AS vd "
        f"{join_clause}"
    )

    if cfg.region.mode != "full" or getattr(tap_client, "is_synthetic", False):
        where_spatial = _adql_spatial_clause(cfg, "vd.ra", "vd.dec", pad_deg=0.30)
        adql = f"{select_cols} WHERE {band_expr} IN ({bands_csv}) AND ({where_spatial})"
        tbl = tap_client.search(adql, maxrec=1_000_000).to_table()
        if len(tbl) > 0:
            yield tbl.to_pandas()
        return

    stripe_maxrec = 1_000_000
    queue = list(_plan_full_dp2_ra_stripes(tap_client, sinfo, bands))
    total_initial = len(queue)
    done_idx = 0
    while queue:
        ra_lo, ra_hi = queue.pop(0)
        done_idx += 1
        adql = (
            f"{select_cols} "
            f"WHERE {band_expr} IN ({bands_csv}) "
            f"AND (vd.ra >= {ra_lo:.5f} AND vd.ra < {ra_hi:.5f})"
        )
        logger.info(
            "Querying full-DP2 TAP %s stripe %d/%d: RA=[%.2f, %.2f) deg (UWS /async) ...",
            sinfo.visit_detector_table,
            done_idx,
            max(done_idx + len(queue), total_initial),
            ra_lo,
            ra_hi,
        )
        tbl = tap_client.search(adql, maxrec=stripe_maxrec, async_query=True).to_table()
        if len(tbl) >= stripe_maxrec and (ra_hi - ra_lo) > 0.05:
            ra_mid = 0.5 * (ra_lo + ra_hi)
            logger.warning(
                "Stripe RA=[%.2f, %.2f) hit MAXREC=%d; bisecting into [%.2f, %.2f) and [%.2f, %.2f)!",
                ra_lo,
                ra_hi,
                stripe_maxrec,
                ra_lo,
                ra_mid,
                ra_mid,
                ra_hi,
            )
            queue.insert(0, (ra_mid, ra_hi))
            queue.insert(0, (ra_lo, ra_mid))
            continue
        if len(tbl) > 0:
            yield tbl.to_pandas()
        del tbl


def build_tap_property_mask(
    pipe: SkyMaskPipe,
    tap_client: IVOATapClient | "SyntheticDP2TapService",
    cfg: MaskPipelineConfig,
    guard: MemoryGuard,
    *,
    output_stage: str = "propmask",
    ebv_provider: EBVProvider | None = None,
    rx_map: Mapping[str, float] | None = None,
    schema_info: TapSchemaInfo | None = None,
    use_obscore_coadd_footprint: bool = True,
) -> tuple[hsp.HealSparseMap, PropertyMaskStats]:
    """Build ``propmask`` locally from RSP TAP ``VisitDetector`` polygons + ``ObsCore`` + Dust De-reddening!

    Matches the ``(propmask, prop_stats)`` return signature of ``build_dp2_property_mask``
    (storing diagnostic coadd depth maps on ``pipe._tap_coadd_maps`` for inspection).
    """
    pm = cfg.property_maps
    sinfo = schema_info or discover_dp2_tap_schema(tap_client)

    # On wide full-survey runs, LSSTCam CCDs (~0.22 deg = 800") span ~16x16 pixels at order 12
    # (51.5" pixels), saving 64x polygon-rasterization RAM/CPU while `combine` outputs `order_out` (15).
    raw_order_sparse = int(pm.order_sparse)
    order_sparse = min(raw_order_sparse, 12) if cfg.region.mode == "full" else raw_order_sparse
    order_cov = int(pm.order_cov)
    nside_sparse = 1 << order_sparse
    nside_cov = 1 << order_cov

    apply_dust = bool(cfg.dust.enabled and cfg.dust.apply_to_maglim)
    used_rx = (
        dict(rx_map)
        if (apply_dust and rx_map is not None)
        else (
            get_rubin_band_extinction_rx(r_v=cfg.dust.r_v, overrides=cfg.dust.r_x_overrides)
            if apply_dust
            else {}
        )
    )
    active_ebv = (
        ebv_provider
        if (apply_dust and ebv_provider is not None)
        else (build_tap_ebv_provider(tap_client, cfg, schema_info=sinfo) if apply_dust else None)
    )

    logger.info(
        "Streaming TAP %s detector pointings across bands=%s (order_sparse=%d) ...",
        sinfo.visit_detector_table,
        pm.bands,
        order_sparse,
    )

    # Optional IVOA ObsCore deep_coadd patch polygon constraint (calib_level=3) on regional runs
    obscore_valid_by_band: dict[str, np.ndarray] = {}
    if use_obscore_coadd_footprint and sinfo.has_obscore and cfg.region.mode != "full":
        bands_csv = ", ".join(f"'{b}'" for b in pm.bands)
        where_obs = _adql_spatial_clause(cfg, "s_ra", "s_dec", pad_deg=0.45)
        adql_obs = (
            f"SELECT lsst_band, s_region FROM ivoa.ObsCore "
            f"WHERE dataproduct_subtype = 'lsst.deep_coadd' AND calib_level = 3 "
            f"AND lsst_band IN ({bands_csv}) AND ({where_obs})"
        )
        try:
            obs_tbl = tap_client.search(adql_obs, maxrec=250_000).to_table()
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

    # Stream each TAP stripe directly into compact per-band HEALPix accumulators (22 B/pixel)
    # so multi-million-row DP2 VisitDetector tables are never concatenated together in RAM!
    band_accum: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {
        b: (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int16),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
        for b in pm.bands
    }
    total_vd_rows = 0

    for stripe_df in _iter_visit_detector_tables(tap_client, cfg, sinfo, pm.bands):
        total_vd_rows += len(stripe_df)
        if "band" in stripe_df.columns:
            stripe_df["band"] = stripe_df["band"].astype(str).str.strip()
        has_corners = all(
            c in stripe_df.columns
            for c in ("llcra", "llcdec", "ulcra", "ulcdec", "urcra", "urcdec", "lrcra", "lrcdec")
        )

        for band in pm.bands:
            sub = stripe_df[stripe_df["band"] == band]
            if len(sub) == 0:
                continue
            crit = pm.per_band.get(band, BandCriteria())
            exptimes = sub["expTime"].to_numpy(dtype=np.float64)
            maglims = sub["magLim"].to_numpy(dtype=np.float64)
            seeings = sub["seeing"].to_numpy(dtype=np.float64)
            ras = sub["ra"].to_numpy(dtype=np.float64)
            decs = sub["dec"].to_numpy(dtype=np.float64)

            if has_corners:
                c_ra_raw = np.column_stack(
                    [sub["llcra"], sub["ulcra"], sub["urcra"], sub["lrcra"]]
                ).astype(np.float64)
                c_dec_raw = np.column_stack(
                    [sub["llcdec"], sub["ulcdec"], sub["urcdec"], sub["lrcdec"]]
                ).astype(np.float64)
                finite_quad = np.all(np.isfinite(c_ra_raw), axis=1) & np.all(
                    np.isfinite(c_dec_raw), axis=1
                )
                c_ra_ord = np.empty_like(c_ra_raw)
                c_dec_ord = np.empty_like(c_dec_raw)
                if np.any(finite_quad):
                    c_ra_ord[finite_quad], c_dec_ord[finite_quad] = _order_quads_ccw_batch(
                        c_ra_raw[finite_quad], c_dec_raw[finite_quad]
                    )
            else:
                finite_quad = np.zeros(len(sub), dtype=bool)
                c_ra_ord = c_dec_ord = np.empty((0, 4))

            acc_pix, acc_nexp, acc_et, acc_w, acc_sw = band_accum[band]
            det_batch = 15_000
            for b_start in range(0, len(sub), det_batch):
                b_end = min(len(sub), b_start + det_batch)
                pix_chunks: list[np.ndarray] = []
                exp_chunks: list[np.ndarray] = []
                wt_chunks: list[np.ndarray] = []
                see_wt_chunks: list[np.ndarray] = []

                for i in range(b_start, b_end):
                    ml_i = maglims[i]
                    if not np.isfinite(ml_i):
                        continue
                    et_i = (
                        exptimes[i]
                        if np.isfinite(exptimes[i]) and exptimes[i] > 0
                        else float(crit.nominal_exptime_sec)
                    )
                    if finite_quad[i]:
                        try:
                            dpix = hpg.query_polygon(
                                nside_sparse, c_ra_ord[i], c_dec_ord[i], nest=True, degrees=True
                            )
                        except Exception:
                            dpix = _rasterize_convex_quad(nside_sparse, c_ra_ord[i], c_dec_ord[i])
                    elif np.isfinite(ras[i]) and np.isfinite(decs[i]):
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
                    w_i = 10.0 ** (0.8 * (ml_i - 25.0))
                    s_i = seeings[i] if np.isfinite(seeings[i]) else 1.0
                    pix_chunks.append(dpix)
                    exp_chunks.append(np.full(dpix.size, et_i, dtype=np.float64))
                    wt_chunks.append(np.full(dpix.size, w_i, dtype=np.float64))
                    see_wt_chunks.append(np.full(dpix.size, w_i * s_i, dtype=np.float64))

                if not pix_chunks:
                    continue

                cat_p = np.concatenate([acc_pix, *pix_chunks])
                cat_n = np.concatenate(
                    [acc_nexp.astype(np.int32), np.ones(sum(c.size for c in pix_chunks), dtype=np.int32)]
                )
                cat_et = np.concatenate([acc_et.astype(np.float64), *exp_chunks])
                cat_w = np.concatenate([acc_w.astype(np.float64), *wt_chunks])
                cat_sw = np.concatenate([acc_sw.astype(np.float64), *see_wt_chunks])
                del pix_chunks, exp_chunks, wt_chunks, see_wt_chunks

                acc_pix, inv_idx = np.unique(cat_p, return_inverse=True)
                acc_nexp = np.minimum(np.bincount(inv_idx, weights=cat_n), 32_000).astype(np.int16)
                acc_et = np.bincount(inv_idx, weights=cat_et).astype(np.float32)
                acc_w = np.bincount(inv_idx, weights=cat_w).astype(np.float32)
                acc_sw = np.bincount(inv_idx, weights=cat_sw).astype(np.float32)
                del cat_p, cat_n, cat_et, cat_w, cat_sw, inv_idx

            band_accum[band] = (acc_pix, acc_nexp, acc_et, acc_w, acc_sw)

        del stripe_df
        gc.collect()
        guard.check(f"TAP stripe accumulated ({total_vd_rows:,} VisitDetector rows processed)")

    logger.info("Accumulated %d total detector-visit rows from TAP %s", total_vd_rows, sinfo.visit_detector_table)

    per_band_area: dict[str, float] = {}
    coadd_maps: dict[str, hsp.HealSparseMap] = {}
    n_pass_map = hsp.HealSparseMap.make_empty(nside_cov, nside_sparse, dtype=np.int16, sentinel=0)
    req_bands = list(pm.required_bands)
    req_pass_maps: dict[str, hsp.HealSparseMap] = {}

    for band in pm.bands:
        crit = pm.per_band.get(band, BandCriteria())
        acc_pix, acc_nexp, acc_et, acc_w, acc_sw = band_accum.pop(band)

        if acc_pix.size == 0:
            per_band_area[band] = 0.0
            empty_b = hsp.HealSparseMap.make_empty(nside_cov, nside_sparse, dtype=np.bool_, bit_packed=True)
            if pm.keep_per_band_stages:
                setattr(pipe, f"prop_{band}_mask", empty_b)
            if band in req_bands:
                req_pass_maps[band] = empty_b
            continue

        keep_geom = np.ones(acc_pix.size, dtype=bool)
        if band in obscore_valid_by_band:
            keep_geom &= np.isin(acc_pix, obscore_valid_by_band[band], assume_unique=True)
        if region_clip_pix is not None:
            keep_geom &= np.isin(acc_pix, region_clip_pix, assume_unique=True)

        uniq_p = acc_pix[keep_geom]
        n_exp_pix = acc_nexp[keep_geom]
        sum_et = acc_et[keep_geom]
        sum_w = acc_w[keep_geom]
        sum_sw = acc_sw[keep_geom]
        del acc_pix, acc_nexp, acc_et, acc_w, acc_sw

        coadd_maglim_obs = 25.0 + 1.25 * np.log10(np.maximum(sum_w, 1e-30))
        mean_seeing_fwhm = sum_sw / np.maximum(sum_w, 1e-30)

        if apply_dust and active_ebv is not None and uniq_p.size > 0:
            ebv_pix = active_ebv.eval_nest_pixels(nside_sparse, uniq_p)
            a_band = compute_band_extinction_mag(ebv_pix, band=band, r_v=cfg.dust.r_v, rx_map=used_rx)
            coadd_maglim_eval = coadd_maglim_obs - a_band
        else:
            coadd_maglim_eval = coadd_maglim_obs

        if cfg.region.mode != "full" and band in ("g", "r", "i") and uniq_p.size > 0:
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
                    extra_meta={
                        "band": band,
                        "source": f"tap:{sinfo.visit_detector_table}",
                        "area_deg2": per_band_area[band],
                    },
                )
            else:
                setattr(pipe, f"prop_{band}_mask", band_map)
        if band in req_bands:
            req_pass_maps[band] = band_map

        guard.check(f"TAP band {band} stacked ({valid_b_pix.size:,} pixels)")

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
        "nside_sparse": nside_sparse,
        "nside_coverage": nside_cov,
        "order_sparse": order_sparse,
        "order_coverage": order_cov,
        "source": f"tap:{sinfo.visit_detector_table}",
        "obscore_clipped": bool(obscore_valid_by_band),
        "bands": list(pm.bands),
        "min_filters": int(pm.min_filters),
        "required_bands": req_bands,
        "dust_dereddened_maglim": apply_dust,
        "ebv_source_used": ebv_src,
        "dust_rx_by_band": {b: round(float(v), 4) for b, v in used_rx.items()} if apply_dust else None,
        "pixels": n_valid,
        "area_deg2": area_deg2,
        "per_band_area_deg2": per_band_area,
        "bit_packed": True,
    }
    if hasattr(pipe, "_finalize_stage"):
        pipe._finalize_stage(
            propmask, default_name="propmask", output_stage=output_stage, extra_meta=extra_meta
        )
    else:
        setattr(pipe, output_stage, propmask)

    setattr(pipe, "_tap_coadd_maps", coadd_maps)

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
        dust_rx_by_band={b: round(float(v), 4) for b, v in used_rx.items()} if apply_dust else {},
    )
    return propmask, stats


def gaia_to_synthetic_lsst_magnitudes(
    phot_g_mean_mag: np.ndarray | Sequence[float],
    bp_rp: np.ndarray | Sequence[float] | None = None,
) -> dict[str, np.ndarray]:
    """Transform Gaia DR3 $(G, G_{\\mathrm{BP}} - G_{\\mathrm{RP}})$ into synthetic apparent LSST
    ``ugrizy`` AB magnitudes along the stellar locus.
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


def _compute_star_query_clauses(
    footprint_mask: hsp.HealSparseMap,
    cfg: MaskPipelineConfig,
    buffer_deg: float,
) -> list[str]:
    """Build ADQL spatial clauses for bright-star TAP queries.

    - Regional ``cone`` / ``box``: single padded spatial clause.
    - Full-survey ``full``: groups occupied ``nside_cov=32`` footprint tiles (excluding
      Galactic plane tiles already vetoed by ``mwmask`` when ``galactic_mask.enabled``)
      into ``nside=4`` (~14.6 deg) sky cones so Gaia DR3 TAP only queries active DP2 sky!
    """
    if cfg.region.mode != "full":
        return [_adql_spatial_clause(cfg, "ra", "dec", pad_deg=buffer_deg)]

    cov_pix = np.where(footprint_mask.coverage_mask)[0]
    if cov_pix.size == 0:
        return []

    nside_cov = int(footprint_mask.nside_coverage)
    c_ra, c_dec = hpg.pixel_to_angle(nside_cov, cov_pix, nest=True, degrees=True)
    if cfg.galactic_mask.enabled:
        b_deg = np.rad2deg(np.abs(_icrs_to_galactic_lat_rad(c_ra, c_dec)))
        keep_gal = b_deg >= max(0.0, float(cfg.galactic_mask.b0_deg) - 2.0)
        c_ra, c_dec = c_ra[keep_gal], c_dec[keep_gal]
        if c_ra.size == 0:
            return []

    nside_cone = 8  # 768 sky cells across 4pi; max corner radius ~ 5.8 deg (~108 deg^2/cone)
    cone_ids = np.unique(hpg.angle_to_pixel(nside_cone, c_ra, c_dec, nest=True, degrees=True))
    q_ra, q_dec = hpg.pixel_to_angle(nside_cone, cone_ids, nest=True, degrees=True)
    cone_rad = float(hpg.max_pixel_radius(nside_cone, degrees=True)) + 0.35 + float(buffer_deg)
    return [
        f"CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {float(r0):.5f}, {float(d0):.5f}, {cone_rad:.5f})) = 1"
        for r0, d0 in zip(q_ra, q_dec)
    ]


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
    radfunction: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> MonsterStarMaskStats:
    """Query bright stars via Gaia DR3 TAP (streamed by sky cone on full-survey DP2 runs),
    compute synthetic LSST magnitudes, evaluate ``skykatana_lsst_radfunction``, and pixelize
    ``starmask`` + ``boxmask`` — returning a ``MonsterStarMaskStats`` summary matching Notebook 01.
    """
    ms_cfg = cfg.monster_stars
    work_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = work_dir / "monster_bright_stars_circles.parquet"
    boxes_fits_path = work_dir / "monster_bleed_boxes.fits"

    active_radfn = radfunction if radfunction is not None else make_skykatana_radfunction(ms_cfg)
    buffer_deg = float(ms_cfg.max_radius_arcsec) / 3600.0 + 0.05

    if gaia_tap_client is None:
        if isinstance(rsp_tap_client, SyntheticDP2TapService):
            gaia_tap_client = rsp_tap_client
        else:
            gaia_tap_client = IVOATapClient(DEFAULT_GAIA_TAP_URL)

    cov_pix = np.where(footprint_mask.coverage_mask)[0]
    nside_cov = int(footprint_mask.nside_coverage)
    dilated_128 = np.zeros(hpg.nside_to_npixel(128), dtype=bool)
    if cov_pix.size > 0:
        c_ra, c_dec = hpg.pixel_to_angle(nside_cov, cov_pix, nest=True, degrees=True)
        tile_rad = math.sqrt(4.0 * math.pi * (180.0 / math.pi) ** 2 / hpg.nside_to_npixel(nside_cov))
        for r0, d0 in zip(c_ra, c_dec):
            dp = hpg.query_circle(128, float(r0), float(d0), tile_rad + buffer_deg, nest=True, degrees=True)
            dilated_128[dp] = True

    clauses = _compute_star_query_clauses(footprint_mask, cfg, buffer_deg=buffer_deg)
    g_cut = float(ms_cfg.mag_max) + 0.6
    target_band = "r" if ms_cfg.profile_band in ("r_halo", "multi") else ms_cfg.profile_band

    arrow_schema = pa.schema(
        [
            ("ra", pa.float64()),
            ("dec", pa.float64()),
            ("mag", pa.float32()),
            ("radius", pa.float32()),
        ]
    )
    writer = pq.ParquetWriter(parquet_path, arrow_schema, compression="snappy")
    seen_keys: set[int] = set()
    total_stars = 0
    star_source = "gaia_dr3_tap"
    flux_cols_used = ["phot_g_mean_mag", "bp_rp -> synthetic_lsst_ugrizy"]

    try:
        for idx_c, where_clause in enumerate(clauses):
            adql_gaia = (
                f"SELECT ra, dec, phot_g_mean_mag, bp_rp "
                f"FROM gaiadr3.gaia_source "
                f"WHERE ({where_clause}) AND phot_g_mean_mag <= {g_cut:.3f}"
            )
            try:
                g_tbl = None
                last_gaia_err: Exception | None = None
                for attempt in range(3):
                    try:
                        g_tbl = gaia_tap_client.search(
                            adql_gaia,
                            maxrec=1_500_000,
                            async_query=(attempt >= 1 and not getattr(gaia_tap_client, "is_synthetic", False)),
                        ).to_table()
                        break
                    except Exception as g_exc:
                        last_gaia_err = g_exc
                        if getattr(gaia_tap_client, "is_synthetic", False):
                            raise
                        time.sleep(1.0 * (attempt + 1))
                if g_tbl is None:
                    assert last_gaia_err is not None
                    raise last_gaia_err
                ra_deg = np.asarray(g_tbl["ra"], dtype=np.float64) % 360.0
                dec_deg = np.asarray(g_tbl["dec"], dtype=np.float64)
                g_mag = np.asarray(g_tbl["phot_g_mean_mag"], dtype=np.float64)
                bp_rp = np.asarray(g_tbl["bp_rp"], dtype=np.float64)
                synth_lsst = gaia_to_synthetic_lsst_magnitudes(g_mag, bp_rp)
            except Exception as exc:
                if rsp_tap_client is None:
                    raise
                logger.warning(
                    "Public Gaia DR3 TAP unreachable on chunk %d/%d (%s); falling back to RSP TAP Object.",
                    idx_c + 1,
                    len(clauses),
                    exc,
                )
                sinfo = discover_dp2_tap_schema(rsp_tap_client)
                where_obj = where_clause.replace("ra,", "coord_ra,").replace("dec)", "coord_dec)")
                adql_obj = (
                    f"SELECT coord_ra AS ra, coord_dec AS dec, g_psfMag, r_psfMag, i_psfMag "
                    f"FROM {sinfo.object_table} "
                    f"WHERE ({where_obj}) AND refExtendedness = 0 AND r_psfMag <= {float(ms_cfg.mag_max):.3f}"
                )
                o_tbl = rsp_tap_client.search(adql_obj, maxrec=500_000).to_table()
                ra_deg = np.asarray(o_tbl["ra"], dtype=np.float64) % 360.0
                dec_deg = np.asarray(o_tbl["dec"], dtype=np.float64)
                synth_lsst = {
                    "lsst_g_mag": np.asarray(o_tbl["g_psfMag"], dtype=np.float64),
                    "lsst_r_mag": np.asarray(o_tbl["r_psfMag"], dtype=np.float64),
                    "lsst_i_mag": np.asarray(o_tbl["i_psfMag"], dtype=np.float64),
                }
                star_source = f"rsp_tap:{sinfo.object_table}"
                flux_cols_used = ["g_psfMag", "r_psfMag", "i_psfMag"]

            if ra_deg.size == 0:
                continue

            primary_mag = synth_lsst.get(f"lsst_{target_band}_mag", synth_lsst["lsst_r_mag"])
            keep = (
                np.isfinite(ra_deg)
                & np.isfinite(dec_deg)
                & np.isfinite(primary_mag)
                & (primary_mag >= float(ms_cfg.mag_min))
                & (primary_mag <= float(ms_cfg.mag_max))
            )
            if not np.any(keep):
                continue

            p128 = hpg.angle_to_pixel(128, ra_deg[keep], dec_deg[keep], nest=True, degrees=True)
            in_foot = dilated_128[p128]
            idx_sel = np.flatnonzero(keep)[in_foot]
            if idx_sel.size == 0:
                continue

            # Deduplicate overlapping cones via sub-arcsecond HEALPix nside=2^20 key
            if len(clauses) > 1:
                hp_keys = hpg.angle_to_pixel(
                    1 << 20, ra_deg[idx_sel], dec_deg[idx_sel], nest=True, degrees=True
                )
                uniq_mask = np.array([int(k) not in seen_keys for k in hp_keys], dtype=bool)
                for k in hp_keys[uniq_mask]:
                    seen_keys.add(int(k))
                idx_sel = idx_sel[uniq_mask]
                if idx_sel.size == 0:
                    continue

            cols_df: dict[str, np.ndarray] = {
                "ra": ra_deg[idx_sel],
                "dec": dec_deg[idx_sel],
                "mag": primary_mag[idx_sel],
            }
            for k_name, arr in synth_lsst.items():
                cols_df[k_name] = arr[idx_sel]
            chunk_df = pd.DataFrame(cols_df)
            active_radfn(chunk_df)

            writer.write_table(
                pa.Table.from_arrays(
                    [
                        pa.array(chunk_df["ra"].to_numpy(dtype=np.float64), type=pa.float64()),
                        pa.array(chunk_df["dec"].to_numpy(dtype=np.float64), type=pa.float64()),
                        pa.array(chunk_df["mag"].to_numpy(dtype=np.float32), type=pa.float32()),
                        pa.array(chunk_df["radius"].to_numpy(dtype=np.float32), type=pa.float32()),
                    ],
                    schema=arrow_schema,
                )
            )
            total_stars += len(chunk_df)
            del chunk_df, cols_df
    finally:
        writer.close()

    del seen_keys
    gc.collect()

    order_star = int(ms_cfg.order_sparse)
    order_cov = int(cfg.property_maps.order_cov)

    if total_stars > 0:
        circles_df = pd.read_parquet(parquet_path, columns=["ra", "dec", "radius"])
        starmask = pipe.build_circ_mask(
            data=circles_df,
            order_sparse=order_star,
            order_cov=order_cov,
            columns=["ra", "dec", "radius"],
            bit_packed=True,
            n_threads=int(cfg.memory.skykatana_threads),
            chunk_size=int(cfg.memory.circle_chunk_size),
            output_stage=star_stage,
        )
        del circles_df
    else:
        starmask = hsp.HealSparseMap.make_empty(
            1 << order_cov, 1 << order_star, dtype=np.bool_, bit_packed=True
        )
        if hasattr(pipe, "_finalize_stage"):
            pipe._finalize_stage(starmask, default_name=star_stage, output_stage=star_stage)
        else:
            setattr(pipe, star_stage, starmask)

    n_bleed = 0
    box_area = 0.0
    if ms_cfg.enable_bleed_boxes and total_stars > 0:
        bright = pd.read_parquet(parquet_path, columns=["ra", "dec", "mag", "radius"])
        bright = bright[bright["mag"] <= float(ms_cfg.bleed_mag_max)].reset_index(drop=True)
        n_bleed = len(bright)
        if n_bleed > 0:
            box_area = build_oriented_bleed_box_mask(
                pipe=pipe,
                bright_df=bright,
                ms_cfg=ms_cfg,
                order_sparse=order_star,
                order_cov=order_cov,
                n_threads=int(cfg.memory.skykatana_threads),
                boxes_fits_path=boxes_fits_path,
                output_stage=box_stage,
            )
            del bright
            if not ms_cfg.save_star_circles_parquet and boxes_fits_path.exists():
                boxes_fits_path.unlink()
        else:
            del bright

    saved_parquet_str: str | None = str(parquet_path)
    if not ms_cfg.save_star_circles_parquet and parquet_path.exists():
        parquet_path.unlink()
        saved_parquet_str = None

    guard.check("TAP bright-star & bleed stages pixelized")
    return MonsterStarMaskStats(
        dataset_type=star_source,
        flux_columns_used=flux_cols_used,
        radius_model=ms_cfg.radius_model,
        profile_band=ms_cfg.profile_band,
        completeness=float(ms_cfg.completeness),
        n_htm7_trixels_queried=len(clauses),
        n_htm7_shards_loaded=len(clauses),
        n_stars_selected=total_stars,
        n_bleed_boxes=n_bleed,
        starmask_valid_pixels=int(starmask.n_valid),
        starmask_area_deg2=float(starmask.get_valid_area(degrees=True)),
        boxmask_area_deg2=box_area,
        circles_parquet_path=saved_parquet_str,
    )


class DP2LocalTapMaskBuilder:
    """One-command local production builder mirroring ``DP2MaskBuilder`` via IVOA TAP services."""

    def __init__(
        self,
        cfg: MaskPipelineConfig,
        tap_client: IVOATapClient | "SyntheticDP2TapService" | None = None,
        gaia_tap_client: IVOATapClient | "SyntheticDP2TapService" | None = None,
        tap_url: str = DEFAULT_RSP_TAP_URL,
        allow_synthetic_fallback: bool = True,
    ) -> None:
        cfg.validate()
        self.cfg = cfg
        self._tap_client = tap_client
        self._gaia_tap_client = gaia_tap_client
        self._tap_url = tap_url
        self._allow_synthetic_fallback = bool(allow_synthetic_fallback)
        configure_rsp_thread_env(max_threads=cfg.memory.skykatana_threads)
        self.guard = MemoryGuard(
            max_ram_gib=cfg.memory.max_ram_gib,
            soft_warning_fraction=cfg.memory.soft_warning_fraction,
            hard_abort_fraction=cfg.memory.hard_abort_fraction,
        )
        self.pipe = SkyMaskPipe(
            order_out=cfg.output.order_out,
            order_cov=cfg.output.order_cov,
        )

    def estimate_memory(self, assumed_area_deg2: float = 6000.0) -> dict[str, float]:
        eff_prop_order = (
            min(int(self.cfg.property_maps.order_sparse), 12)
            if self.cfg.region.mode == "full"
            else int(self.cfg.property_maps.order_sparse)
        )
        return estimate_working_set_gib(
            area_deg2=assumed_area_deg2,
            order_prop=eff_prop_order,
            order_star=(
                self.cfg.monster_stars.order_sparse
                if self.cfg.monster_stars.enabled
                else eff_prop_order
            ),
            order_out=self.cfg.output.order_out,
            order_cov=self.cfg.output.order_cov,
            cov_batch_size=self.cfg.memory.cov_batch_size,
            circle_chunk_size=self.cfg.memory.circle_chunk_size,
            n_threads=self.cfg.memory.skykatana_threads,
        )

    def run(self) -> tuple[SkyMaskPipe, MaskBuildSummary]:
        t0 = time.perf_counter()
        out_dir = Path(self.cfg.output.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.guard.check("local TAP pipeline start")

        tap = (
            self._tap_client
            if self._tap_client is not None
            else open_rsp_tap_client(
                tap_url=self._tap_url,
                token_file=self.cfg.access_token_file,
                allow_synthetic_fallback=self._allow_synthetic_fallback,
                synthetic_kwargs=(
                    {
                        "center_ra_deg": self.cfg.region.ra_deg,
                        "center_dec_deg": self.cfg.region.dec_deg,
                        "radius_deg": self.cfg.region.radius_deg,
                    }
                    if self.cfg.region.mode == "cone" and self.cfg.region.ra_deg is not None
                    else ({"wide_dp2_survey": True} if self.cfg.region.mode == "full" else {})
                ),
            )
        )
        sinfo = discover_dp2_tap_schema(tap)

        ebv_provider: EBVProvider | None = None
        rx_map: dict[str, float] | None = None
        if self.cfg.dust.enabled:
            ebv_provider = build_tap_ebv_provider(tap, self.cfg, schema_info=sinfo)
            rx_map = get_rubin_band_extinction_rx(
                r_v=self.cfg.dust.r_v,
                overrides=self.cfg.dust.r_x_overrides,
            )

        logger.info("=== Stage 1 (Local TAP): Stacking %s -> propmask ===", sinfo.visit_detector_table)
        propmask, prop_stats = build_tap_property_mask(
            self.pipe,
            tap_client=tap,
            cfg=self.cfg,
            guard=self.guard,
            output_stage="propmask",
            ebv_provider=ebv_provider,
            rx_map=rx_map,
            schema_info=sinfo,
        )
        positive_stages = ["propmask"]
        negative_stages: list[str] = []

        dust_stats: DustMaskStats | None = None
        if self.cfg.dust.enabled and ebv_provider is not None:
            logger.info("=== Stage 1b (Local TAP): Evaluating Milky Way E(B-V) dust mask (dustmask) ===")
            dust_map, dust_stats = build_dust_extinction_mask(
                self.pipe,
                footprint_mask=propmask,
                cfg=self.cfg,
                ebv_provider=ebv_provider,
                guard=self.guard,
                rx_map=rx_map,
                output_stage="dustmask",
            )
            if dust_map is not None and dust_stats.dustmask_valid_pixels > 0:
                negative_stages.append("dustmask")

        mwmask_area_deg2 = 0.0
        if self.cfg.galactic_mask.enabled:
            logger.info("=== Stage 2a (Local TAP): Building Milky Way plane & bulge mask (mwmask) ===")
            mw = self.pipe.build_milkyway_mask(
                order_sparse=int(self.cfg.galactic_mask.order_sparse),
                order_cov=int(self.cfg.output.order_cov),
                b0_deg=float(self.cfg.galactic_mask.b0_deg),
                bulge_a_deg=float(self.cfg.galactic_mask.bulge_a_deg),
                bulge_b_deg=float(self.cfg.galactic_mask.bulge_b_deg),
                bit_packed=True,
                output_stage="mwmask",
            )
            mwmask_area_deg2 = float(mw.get_valid_area(degrees=True))
            if int(mw.n_valid) > 0:
                negative_stages.append("mwmask")
            self.guard.check("mwmask finalized", collect_garbage=True)

        star_stats: MonsterStarMaskStats | None = None
        if self.cfg.monster_stars.enabled:
            logger.info("=== Stage 2b (Local TAP): Building Gaia->LSST bright-star mask (starmask) ===")
            star_stats = build_local_tap_star_mask(
                self.pipe,
                footprint_mask=propmask,
                cfg=self.cfg,
                guard=self.guard,
                work_dir=out_dir,
                gaia_tap_client=self._gaia_tap_client,
                rsp_tap_client=tap,
                star_stage="starmask",
                box_stage="boxmask",
            )
            if star_stats.starmask_valid_pixels > 0:
                negative_stages.append("starmask")
            if self.cfg.monster_stars.enable_bleed_boxes and star_stats.n_bleed_boxes > 0:
                negative_stages.append("boxmask")

        logger.info(
            "=== Stage 4 (Local TAP): Combining positive=%s minus negative=%s -> order_out=%d ===",
            positive_stages,
            negative_stages,
            self.cfg.output.order_out,
        )
        combined_mask = self.pipe.combine(
            positive=positive_stages,
            negative=negative_stages if negative_stages else None,
            order_out=int(self.cfg.output.order_out),
            order_cov=int(self.cfg.output.order_cov),
            bit_packed=True,
            verbose=False,
            output_stage="mask",
        )
        gc.collect()
        self.guard.check("combined mask finalized")

        final_valid_pixels = int(combined_mask.n_valid)
        final_area_deg2 = float(combined_mask.get_valid_area(degrees=True))
        retained_frac = final_area_deg2 / prop_stats.area_deg2 if prop_stats.area_deg2 > 0 else 0.0

        output_files = self._export_outputs(out_dir)
        elapsed = time.perf_counter() - t0

        summary = MaskBuildSummary(
            butler_repo=f"tap:{tap.tap_url}",
            butler_collections=[sinfo.catalog_schema],
            region_mode=self.cfg.region.mode,
            order_out=int(self.cfg.output.order_out),
            order_cov=int(self.cfg.output.order_cov),
            nside_out=1 << int(self.cfg.output.order_out),
            property_mask=prop_stats,
            dust_mask=dust_stats,
            monster_stars=star_stats,
            mwmask_area_deg2=mwmask_area_deg2,
            positive_stages=positive_stages,
            negative_stages=negative_stages,
            final_valid_pixels=final_valid_pixels,
            final_area_deg2=final_area_deg2,
            footprint_retained_fraction=retained_frac,
            full_sky_fraction=final_area_deg2 / FULL_SKY_AREA_DEG2,
            peak_ram_gib=round(self.guard.peak_used_gib, 4),
            memory_limit_gib=round(self.guard.limit_gib, 2),
            memory_limit_source=self.guard.limit_source,
            elapsed_seconds=round(elapsed, 3),
            output_files=output_files,
        )

        if self.cfg.output.make_qa_plots:
            qa_path = out_dir / "dp2_mask_qa.png"
            _render_qa_plot(self.pipe, summary, qa_path)
            summary.output_files["qa_plot_png"] = str(qa_path)
            if self.pipe.mask is not None and int(self.pipe.mask.n_valid) > 0:
                import matplotlib.pyplot as plt

                from .visualization import plot_complete_mask_multiscale

                circ_p = star_stats.circles_parquet_path if star_stats is not None else None
                box_p = out_dir / "monster_bleed_boxes.fits"
                ms_path = out_dir / "dp2_complete_mask_multiscale.png"
                fig_ms, _ = plot_complete_mask_multiscale(
                    self.pipe,
                    circles_path=circ_p,
                    boxes_path=box_p if box_p.is_file() else None,
                    show=False,
                    save_path=ms_path,
                )
                plt.close(fig_ms)
                summary.output_files["complete_mask_multiscale_png"] = str(ms_path)

        summary_path = summary.write_json(out_dir / "mask_build_summary.json")
        summary.output_files["summary_json"] = str(summary_path)
        logger.info(
            "Finished Local TAP DP2 SkyKatana mask: valid_area=%.3f deg^2 (%.1f%% of propmask), "
            "peak_RAM=%.2f GiB, elapsed=%.1fs",
            summary.final_area_deg2,
            100.0 * summary.footprint_retained_fraction,
            summary.peak_ram_gib,
            summary.elapsed_seconds,
        )
        return self.pipe, summary

    def _export_outputs(self, out_dir: Path) -> dict[str, str]:
        out_cfg = self.cfg.output
        files: dict[str, str] = {}

        if out_cfg.write_skykatana_pipe:
            pipe_dir = out_dir / "skykatana_pipe"
            self.pipe.write(pipe_dir, overwrite=bool(out_cfg.overwrite))
            files["skykatana_pipe_dir"] = str(pipe_dir)

        if out_cfg.write_healsparse_fits and self.pipe.mask is not None:
            hs_path = out_dir / "dp2_combined_mask.hs"
            self.pipe.mask.write(str(hs_path), clobber=bool(out_cfg.overwrite))
            files["combined_healsparse_hs"] = str(hs_path)

        if out_cfg.write_moc_fits and self.pipe.mask is not None and int(self.pipe.mask.n_valid) > 0:
            eff_moc_order = (
                min(int(out_cfg.moc_max_order), int(out_cfg.order_out))
                if out_cfg.moc_max_order is not None
                else int(out_cfg.order_out)
            )
            moc_path = out_dir / "dp2_combined_mask_moc.fits"
            moc = SkyMaskPipe.moc_from_stage(
                self.pipe.mask,
                order_forced=eff_moc_order,
                progress=False,
            )
            moc.save(str(moc_path), format="fits", overwrite=bool(out_cfg.overwrite))
            files["moc_fits"] = str(moc_path)

        if out_cfg.write_fracdet_map and self.pipe.mask is not None and int(self.pipe.mask.n_valid) > 0:
            eff_frac_order = min(int(out_cfg.fracdet_order), int(out_cfg.order_out))
            frac_path = out_dir / f"dp2_fracdet_o{eff_frac_order}.hs"
            frac_map = SkyMaskPipe.frac_area_map(self.pipe.mask, order_frac=eff_frac_order)
            frac_map.write(str(frac_path), clobber=bool(out_cfg.overwrite))
            files["fracdet_healsparse_hs"] = str(frac_path)
            del frac_map

        if int(out_cfg.n_random_points) > 0 and self.pipe.mask is not None and int(self.pipe.mask.n_valid) > 0:
            rans_path = out_dir / "dp2_randoms.parquet"
            rng = np.random.RandomState(int(out_cfg.random_seed))
            rans_df = self.pipe.makerans(
                stage="mask",
                nr=int(out_cfg.n_random_points),
                file=None,
                rng=rng,
            )
            rans_df.to_parquet(rans_path, index=False)
            files["randoms_parquet"] = str(rans_path)
            del rans_df

        gc.collect()
        self.guard.check("outputs exported")
        return files


class SyntheticDP2TapService:
    """In-memory emulator of RSP DP2 TAP (``dp2.*``, ``ivoa.ObsCore``) + ESA Gaia DR3 TAP
    (``gaiadr3.gaia_source``) for local offline execution and automated testing.
    """

    def __init__(
        self,
        center_ra_deg: float = 53.1222,
        center_dec_deg: float = -28.0870,
        radius_deg: float = 1.5,
        n_stars: int = 240,
        n_objects: int = 18_000,
        seed: int = 2026,
        wide_dp2_survey: bool = False,
    ) -> None:
        self.tap_url = "synthetic://data.lsst.cloud/api/tap"
        self.is_synthetic = True
        self.center_ra_deg = float(center_ra_deg)
        self.center_dec_deg = float(center_dec_deg)
        self.radius_deg = float(radius_deg)
        self.wide_dp2_survey = bool(wide_dp2_survey)
        if self.wide_dp2_survey:
            self._populate_wide_dp2_tables(n_stars=max(n_stars, 1800), n_objects=max(n_objects, 30_000), seed=seed)
        else:
            self._populate_tables(n_stars=n_stars, n_objects=n_objects, seed=seed)

    @property
    def backend_name(self) -> str:
        return "synthetic_dp2_tap"

    def _populate_wide_dp2_tables(self, n_stars: int, n_objects: int, seed: int) -> None:
        """Populate ~2,100 deg^2 multi-band (ugrizy) DP2 WFD + DDF TAP tables for full-survey runs."""
        rng = np.random.default_rng(seed)
        nominal_visits = {"u": 4, "g": 5, "r": 6, "i": 6, "z": 5, "y": 4}
        single_visit_maglim = {
            "u": 23.42,
            "g": 24.38,
            "r": 24.02,
            "i": 23.62,
            "z": 22.98,
            "y": 21.86,
        }
        nominal_seeing = {"u": 1.12, "g": 1.02, "r": 0.94, "i": 0.89, "z": 0.87, "y": 0.86}

        # Coarse focal-plane super-blocks covering WFD (RA in [280, 360) U [0, 42], Dec in [-42, -20])
        # plus ECDFS (53.12, -28.09) and EDFS (60.50, -48.50) across all 6 bands in <1 second
        ra_centers_wfd = np.concatenate([np.arange(282.0, 360.0, 4.0), np.arange(2.0, 41.0, 4.0)])
        dec_centers_wfd = np.arange(-40.0, -21.0, 4.0)
        tile_centers: list[tuple[float, float, float]] = [
            (float(rc), float(dc), abs(float(dc) + 31.0) / 11.2)
            for rc in ra_centers_wfd
            for dc in dec_centers_wfd
        ]
        for r0, d0 in ((53.12, -28.09), (60.50, -48.50)):
            tile_centers.append((r0, d0, 0.15))

        vd_rows: list[dict[str, Any]] = []
        visit_rows: list[dict[str, Any]] = []
        visit_id = 2025090001
        hh_base = 2.08

        for b_idx, (band, n_vis) in enumerate(nominal_visits.items()):
            max_ef = 1.04 - 0.06 * (b_idx % 3)
            for v_k in range(n_vis):
                d_dec_jit = 0.28 * math.sin(2.0 * math.pi * v_k / n_vis)
                d_ra_jit = 0.28 * math.cos(2.0 * math.pi * v_k / n_vis)
                visit_rows.append(
                    {"visitId": visit_id, "band": band, "ra": 0.0, "dec": -31.0, "expTime": 30.0}
                )
                det_num = 0
                for rc0, dc0, ef in tile_centers:
                    if ef > max_ef and v_k >= 2:
                        continue
                    dc = dc0 + d_dec_jit
                    cos_dc = max(math.cos(math.radians(dc)), 0.15)
                    rc = (rc0 + d_ra_jit / cos_dc) % 360.0
                    # Avoid polygons straddling the 0/360 wrap seam directly
                    hw = min(hh_base / cos_dc, 2.8)
                    if rc - hw < 0.05 or rc + hw > 359.95:
                        rc = 2.9 if rc < 180.0 else 357.1
                    ml = float(single_visit_maglim[band] - 0.34 * (ef**2) + rng.normal(0.0, 0.03))
                    vd_rows.append(
                        {
                            "visitId": visit_id,
                            "detector": det_num,
                            "band": band,
                            "expTime": 30.0,
                            "seeing": nominal_seeing[band],
                            "magLim": ml,
                            "ra": rc,
                            "dec": dc,
                            "llcra": rc - hw,
                            "llcdec": dc - hh_base,
                            "ulcra": rc - hw,
                            "ulcdec": dc + hh_base,
                            "urcra": rc + hw,
                            "urcdec": dc + hh_base,
                            "lrcra": rc + hw,
                            "lrcdec": dc - hh_base,
                        }
                    )
                    det_num += 1
                visit_id += 1

        self._visit_tbl = Table(rows=visit_rows)
        self._vd_tbl = Table(rows=vd_rows)
        self._obscore_tbl = Table(
            names=(
                "obs_collection",
                "dataproduct_type",
                "dataproduct_subtype",
                "calib_level",
                "lsst_band",
                "s_ra",
                "s_dec",
                "s_region",
            ),
            dtype=(str, str, str, int, str, float, float, str),
        )

        # Sample objects across WFD + DDFs with SFD98 E(B-V) cirrus cloud at (304, -30)
        n_wfd = int(0.92 * n_objects)
        n_ddf = n_objects - n_wfd
        ra_wfd = (280.0 + rng.uniform(0.0, 122.0, size=n_wfd)) % 360.0
        dec_wfd = rng.uniform(-42.0, -20.0, size=n_wfd)
        ra_ddf = np.concatenate(
            [
                rng.normal(53.12, 0.9, size=n_ddf // 2),
                rng.normal(60.50, 0.9, size=n_ddf - n_ddf // 2),
            ]
        ) % 360.0
        dec_ddf = np.concatenate(
            [
                rng.normal(-28.09, 0.8, size=n_ddf // 2),
                rng.normal(-48.50, 0.8, size=n_ddf - n_ddf // 2),
            ]
        )
        obj_ra = np.concatenate([ra_wfd, ra_ddf])
        obj_dec = np.clip(np.concatenate([dec_wfd, dec_ddf]), -85.0, 85.0)
        dra_c = ((obj_ra - 304.0 + 180.0) % 360.0 - 180.0) * math.cos(math.radians(-30.0))
        ddec_c = obj_dec - (-30.0)
        ebv = 0.022 + 0.21 * np.exp(-((dra_c / 4.8) ** 2 + (ddec_c / 4.2) ** 2))
        rx = get_rubin_band_extinction_rx(r_v=3.1)
        g_intr = rng.uniform(20.2, 24.8, size=n_objects)
        r_intr = g_intr - rng.uniform(0.3, 0.9, size=n_objects)
        i_intr = r_intr - rng.uniform(0.15, 0.45, size=n_objects)
        self._object_tbl = Table(
            {
                "objectId": np.arange(8_000_000, 8_000_000 + n_objects, dtype=np.int64),
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

        s_ra = (280.0 + rng.uniform(0.0, 122.0, size=n_stars)) % 360.0
        s_dec = rng.uniform(-42.0, -20.0, size=n_stars)
        s_g = np.concatenate(
            [
                np.array([7.8, 8.6, 9.4, 10.2, 11.0, 11.6], dtype=np.float64),
                rng.uniform(12.0, 18.2, size=max(0, n_stars - 6)),
            ]
        )[:n_stars]
        self._gaia_tbl = Table(
            {
                "ra": s_ra,
                "dec": s_dec,
                "phot_g_mean_mag": s_g,
                "bp_rp": rng.uniform(0.35, 2.0, size=n_stars),
            }
        )

    def _populate_tables(self, n_stars: int, n_objects: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        ra0, dec0, rad = self.center_ra_deg, self.center_dec_deg, self.radius_deg
        cos_d = math.cos(math.radians(dec0))

        # Realistic LSSTCam 30s single-visit 5-sigma depths so coadded edge + dust cirrus
        # straddle `dp2_ecdfs_pilot.yaml` thresholds (`g=24.8, r=24.6, i=24.2`) realistically!
        nominal_visits = {"u": 7, "g": 12, "r": 16, "i": 16, "z": 11, "y": 9}
        single_visit_maglim = {
            "u": 23.18,
            "g": 24.08,
            "r": 23.72,
            "i": 23.32,
            "z": 22.72,
            "y": 21.62,
        }
        nominal_seeing = {"u": 1.12, "g": 1.02, "r": 0.94, "i": 0.89, "z": 0.87, "y": 0.86}

        vd_rows: list[dict[str, Any]] = []
        visit_rows: list[dict[str, Any]] = []
        visit_id = 2025050001

        det_half_deg = 0.34
        grid_offsets = (-0.88, -0.29, 0.29, 0.88)

        for b_idx, (band, n_vis) in enumerate(nominal_visits.items()):
            dither_scale = 0.44 * (1.0 - 0.04 * (b_idx % 3))
            for v_k in range(n_vis):
                ang = 2.0 * math.pi * v_k / max(1, n_vis)
                d_ra = (dither_scale * math.cos(ang)) / cos_d
                d_dec = dither_scale * math.sin(ang)
                v_ra = (ra0 + d_ra) % 360.0
                v_dec = dec0 + d_dec
                seeing_v = float(np.clip(nominal_seeing[band] + rng.normal(0.0, 0.06), 0.65, 1.38))
                visit_rows.append(
                    {"visitId": visit_id, "band": band, "ra": v_ra, "dec": v_dec, "expTime": 30.0}
                )
                det_num = 0
                for gx in grid_offsets:
                    for gy in grid_offsets:
                        if math.hypot(gx, gy) > 1.26:
                            continue
                        rc = (v_ra + gx / cos_d) % 360.0
                        dc = v_dec + gy
                        r_field_frac = math.hypot(((rc - ra0 + 180.0) % 360.0 - 180.0) * cos_d, dc - dec0) / max(
                            rad, 0.1
                        )
                        hw = det_half_deg / max(math.cos(math.radians(dc)), 0.1)
                        hh = det_half_deg
                        ml = float(
                            single_visit_maglim[band]
                            - 0.42 * (r_field_frac**2)
                            + rng.normal(0.0, 0.04)
                        )
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

        obs_rows: list[dict[str, Any]] = []
        patch_centers = np.linspace(-1.25, 1.25, 6)
        ph = 0.26
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
