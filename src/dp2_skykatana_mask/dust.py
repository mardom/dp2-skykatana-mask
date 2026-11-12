"""Milky Way foreground dust extinction correction and masking for Rubin DP2.

Implements the methodology and caveats of **DP2 Tutorial Notebook 311.1**
(*Milky Way Dust Extinction Correction*,
`https://dp2.lsst.io/tutorials/notebook/311/notebook-311-1.html`):

1. **Bandpass extinction coefficients** ($R_b = A_b / E(B-V)$):
   Uses ``rubin_sim.phot_utils.DustValues(r_v=3.1).r_x`` on the Rubin Science Platform
   (Cardelli, Clayton & Mathis 1989 [CCM89] curve integrated over the LSST ``ugrizy``
   passbands), with an exact zero-dependency CCM89 analytical fallback matching
   ``DustValues(r_v=3.1).r_x`` (`[4.81, 3.64, 2.70, 2.06, 1.58, 1.31]`) when running
   locally without ``rubin_sim``.

2. **De-reddening $5\\sigma$ coadd limiting magnitudes** (`psf_maglim`):
   Extraterrestrial/extragalactic sources behind Galactic dust column $E(B-V)$ suffer
   $A_b = R_b \\times E(B-V)$ magnitudes of attenuation. Enforcing uniform *intrinsic*
   extragalactic survey depth across the footprint therefore requires thresholding the
   dust-corrected limiting magnitude:

   $$m_{b,\\mathrm{lim}}^{\\mathrm{dered}}(\\alpha, \\delta) =
     m_{b,\\mathrm{lim}}^{\\mathrm{obs}}(\\alpha, \\delta) - R_b \\cdot E(B-V)(\\alpha, \\delta).$$

3. **High-extinction & low-Galactic-latitude exclusion stage (`dustmask`)**:
   As noted in Tutorial 311.1, the Schlegel, Finkbeiner & Davis (1998; SFD98) map can be
   contaminated by unsubtracted $100\\,\\mu\\mathrm{m}$ emission near $|b| < 5^\\circ$ and
   carries larger $R_V$ systematics in dense clouds. ``build_dust_extinction_mask`` creates
   a bit-packed negative ``SkyMaskPipe`` stage (``dustmask``) excluding pixels with
   $E(B-V) > E(B-V)_{\\max}$ (and optionally $|b| < b_{\\min}$).

4. **Note on stellar halo masking (`starmask`)**:
   Diffraction spikes, optical ghosts, and scattered-light halos on LSSTCam detectors scale
   with the *observed* photon flux reaching the telescope (and SFD98 integrates dust to
   infinity, which over-corrects foreground Galactic stars). Bright-star halo radii in
   ``monster_stars.py`` therefore intentionally use *observed* synthetic LSST magnitudes.
"""

from __future__ import annotations

import gc
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from astropy.coordinates import SkyCoord
import astropy.units as u
import healpy as hp
import healsparse as hsp
import hpgeom as hpg
import numpy as np
from skykatana import SkyMaskPipe

from .config import DustExtinctionConfig, MaskPipelineConfig
from .memory import MemoryGuard

logger = logging.getLogger(__name__)

# Standard rubin_sim.phot_utils.DustValues(r_v=3.1).r_x coefficients for LSST ugrizy
# (CCM89 extinction curve with R_V = 3.1, flat SED reference; see DP2 Tutorial 311.1).
DEFAULT_RUBIN_RX_RV31: dict[str, float] = {
    "u": 4.81,
    "g": 3.64,
    "r": 2.70,
    "i": 2.06,
    "z": 1.58,
    "y": 1.31,
}

# Effective passband wavelengths (nm) calibrated so CCM89 reproduces DEFAULT_RUBIN_RX_RV31 at R_V=3.1
_RUBIN_EFF_WAVELENGTH_NM: dict[str, float] = {
    "u": 368.5,
    "g": 480.2,
    "r": 623.1,
    "i": 754.2,
    "z": 869.0,
    "y": 973.6,
}


@dataclass
class DustMaskStats:
    """Summary statistics for Milky Way dust extinction correction & `dustmask` stage."""

    enabled: bool
    ebv_source_used: str
    r_v: float
    r_x_by_band: dict[str, float]
    applied_to_maglim: bool
    max_ebv: float | None
    min_galactic_lat_deg: float | None
    order_sparse: int
    median_ebv_in_footprint: float | None
    p95_ebv_in_footprint: float | None
    max_ebv_in_footprint: float | None
    dustmask_valid_pixels: int
    dustmask_area_deg2: float


def ccm89_ab_ratio(wavelength_nm: float, r_v: float = 3.1) -> float:
    """Cardelli, Clayton & Mathis (1989) $A(\\lambda) / E(B-V) = R_V [a(x) + b(x)/R_V]$."""
    if r_v <= 0:
        raise ValueError(f"r_v must be positive, got {r_v}")
    # Wavenumber x = 1 / lambda_um in inverse microns
    x = 1000.0 / float(wavelength_nm)
    if 0.3 <= x < 1.1:
        # Infrared regime
        a = 0.574 * (x**1.61)
        b = -0.527 * (x**1.61)
    elif 1.1 <= x <= 3.3:
        # Optical / NIR regime (covers all LSST ugrizy bands: x ~ 1.03 .. 2.71 um^-1)
        y = x - 1.82
        a = (
            1.0
            + 0.17699 * y
            - 0.50447 * (y**2)
            - 0.02427 * (y**3)
            + 0.72085 * (y**4)
            + 0.01979 * (y**5)
            - 0.77530 * (y**6)
            + 0.32999 * (y**7)
        )
        b = (
            1.41338 * y
            + 2.28305 * (y**2)
            + 1.07233 * (y**3)
            - 5.38434 * (y**4)
            - 0.62251 * (y**5)
            + 5.30260 * (y**6)
            - 2.09002 * (y**7)
        )
    else:
        raise ValueError(f"Wavelength {wavelength_nm} nm outside optical/NIR CCM89 range")
    return float(r_v) * a + b


def get_rubin_band_extinction_rx(
    r_v: float = 3.1,
    overrides: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Return $R_b = A_b / E(B-V)$ for Rubin ``ugrizy`` bands (Tutorial 311.1).

    Attempts ``from rubin_sim.phot_utils import DustValues`` first (native on the RSP),
    falling back to ``DEFAULT_RUBIN_RX_RV31`` (at $R_V = 3.1$) or analytical CCM89
    scaling when ``rubin_sim`` is not installed in a lightweight local environment.
    """
    rx: dict[str, float] | None = None
    try:
        from rubin_sim.phot_utils import DustValues  # type: ignore[import-not-found]

        dv = DustValues(r_v=float(r_v))
        raw_rx = getattr(dv, "r_x", None) or getattr(dv, "ax1", None)
        if raw_rx is not None:
            rx = {b: float(raw_rx[b]) for b in ("u", "g", "r", "i", "z", "y") if b in raw_rx}
    except Exception:
        rx = None

    if not rx or len(rx) < 6:
        if abs(float(r_v) - 3.1) < 1e-6:
            rx = dict(DEFAULT_RUBIN_RX_RV31)
        else:
            # Scale relative to exact R_V=3.1 passband values via CCM89 ratio
            rx = {}
            for band, wl_nm in _RUBIN_EFF_WAVELENGTH_NM.items():
                scale = ccm89_ab_ratio(wl_nm, r_v=float(r_v)) / ccm89_ab_ratio(wl_nm, r_v=3.1)
                rx[band] = round(DEFAULT_RUBIN_RX_RV31[band] * scale, 4)

    if overrides:
        for b, val in overrides.items():
            rx[str(b)] = float(val)

    return rx


def compute_band_extinction_mag(
    ebv: np.ndarray | float,
    band: str,
    r_v: float = 3.1,
    rx_map: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Compute band extinction $A_b = R_b \\times E(B-V)$ in magnitudes."""
    coeffs = rx_map if rx_map is not None else get_rubin_band_extinction_rx(r_v=r_v)
    if band not in coeffs:
        raise KeyError(f"Unknown LSST band {band!r}; available: {sorted(coeffs)}")
    ebv_arr = np.asarray(ebv, dtype=np.float64)
    return float(coeffs[band]) * np.maximum(0.0, ebv_arr)


def deredden_magnitude(
    mag_obs: np.ndarray | float,
    ebv: np.ndarray | float,
    band: str,
    r_v: float = 3.1,
    rx_map: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Correct observed AB magnitudes (or $5\\sigma$ ``psf_maglim``) for Milky Way dust:

    $$m_{b}^{\\mathrm{dered}} = m_{b}^{\\mathrm{obs}} - R_b \\times E(B-V).$$
    """
    a_band = compute_band_extinction_mag(ebv, band=band, r_v=r_v, rx_map=rx_map)
    return np.asarray(mag_obs, dtype=np.float64) - a_band


class EBVProvider:
    """Memory-bounded $E(B-V)$ evaluator across sky coordinates and HEALPix NESTED pixels.

    Supports multiple RSP and local SFD98 backends:
      - ``healsparse``: Pre-loaded or file-backed ``HealSparseMap``
      - ``healpix_icrs`` / ``healpix_galactic``: In-memory 1D HEALPix RING array (e.g. from
        ``rubin_scheduler`` / ``$RUBIN_SIM_DATA_DIR/maps/DustMaps/dust_nside_*.npz``, ~12 MB)
      - ``butler``: Consolidated ``HealSparseMap`` stored in Butler (including ``SyntheticDP2Butler``)
      - ``dustmaps_sfd``: ``dustmaps.sfd.SFDQuery`` callable
    """

    def __init__(
        self,
        source_name: str,
        *,
        hsp_map: hsp.HealSparseMap | None = None,
        hp_ring_map: np.ndarray | None = None,
        hp_is_galactic: bool = False,
        butler_info: tuple[Any, str, str] | None = None,
        sfd_query: Any | None = None,
        analytic_model: bool = False,
    ) -> None:
        self.source_name = source_name
        self._hsp_map = hsp_map
        self._hp_ring_map = hp_ring_map
        self._hp_nside = hp.npix2nside(len(hp_ring_map)) if hp_ring_map is not None else None
        self._hp_is_galactic = bool(hp_is_galactic)
        self._butler_info = butler_info  # (butler, dataset_type, skymap)
        self._sfd_query = sfd_query
        self._analytic_model = bool(analytic_model)

    @classmethod
    def from_config(
        cls,
        cfg: MaskPipelineConfig | DustExtinctionConfig,
        butler: Any | None = None,
    ) -> "EBVProvider":
        """Auto-detect or load the configured $E(B-V)$ dust map source."""
        if isinstance(cfg, DustExtinctionConfig):
            dcfg = cfg
            skymap = "lsst_cells_v2"
            full_cfg: MaskPipelineConfig | None = None
        else:
            dcfg = cfg.dust
            skymap = cfg.property_maps.skymap
            full_cfg = cfg
        mode = dcfg.ebv_source.lower().strip()

        # 1. Explicit file path
        if dcfg.ebv_map_path or mode in ("healsparse_file", "healpix_file", "npz_file"):
            if not dcfg.ebv_map_path:
                raise ValueError(f"dust.ebv_map_path is required when ebv_source={mode!r}")
            return cls.from_file(dcfg.ebv_map_path)

        # 2. Butler EBV map (if present in Butler or SyntheticDP2Butler)
        if mode in ("auto", "butler") and butler is not None:
            prov = cls._try_from_butler(butler, dcfg.ebv_dataset, skymap)
            if prov is not None:
                return prov
            if mode == "butler":
                raise RuntimeError(
                    f"Dust dataset {dcfg.ebv_dataset!r} not found in Butler repository."
                )

        # 3. Rubin Science Platform / rubin_scheduler / RUBIN_SIM_DATA_DIR SFD98 HEALPix map
        if mode in ("auto", "rubin_sim_sfd", "rubin_scheduler"):
            prov = cls._try_from_rubin_sim_data()
            if prov is not None:
                return prov
            if mode in ("rubin_sim_sfd", "rubin_scheduler"):
                raise RuntimeError(
                    "Could not locate Rubin SFD dust map via rubin_scheduler or $RUBIN_SIM_DATA_DIR."
                )

        # 4. dustmaps.sfd package
        if mode in ("auto", "dustmaps_sfd"):
            prov = cls._try_from_dustmaps()
            if prov is not None:
                return prov
            if mode == "dustmaps_sfd":
                raise RuntimeError("`dustmaps.sfd` is not installed or SFD data dir is missing.")

        # 5. RSP TAP dp2_prep.Object `ebv` aggregation (Tutorial 311.1 direct TAP column)
        if mode == "tap_object":
            if full_cfg is None:
                raise ValueError("ebv_source='tap_object' requires a full MaskPipelineConfig.")
            return cls._from_rsp_tap_object(full_cfg)

        # 6. Smooth Galactic cosecant model (explicit or final offline fallback)
        if mode in ("auto", "analytic_galactic"):
            if mode == "auto":
                logger.warning(
                    "No local SFD98 map (Butler / RUBIN_SIM_DATA_DIR / dustmaps) detected; "
                    "using smooth Galactic latitude E(B-V) fallback. On RSP, rubin_sim/rubin_scheduler "
                    "SFD maps or `dust.ebv_map_path` will be used automatically."
                )
            return cls(source_name="analytic_galactic_cosecant", analytic_model=True)

        raise ValueError(f"Unsupported dust.ebv_source: {dcfg.ebv_source!r}")

    @classmethod
    def from_file(cls, path: str | Path) -> "EBVProvider":
        """Load an $E(B-V)$ map from a `.hs`, `.fits`, or `.npz` file."""
        fpath = Path(path).expanduser()
        if not fpath.exists():
            raise FileNotFoundError(f"Dust E(B-V) map file not found: {fpath}")

        suffix = fpath.suffix.lower()
        if suffix == ".npz":
            data = np.load(fpath)
            for key in ("ebvMap", "ebv", "dust", "map"):
                if key in data:
                    arr = np.asarray(data[key], dtype=np.float32).ravel()
                    is_gal = _detect_galactic_frame(arr)
                    return cls(
                        source_name=f"npz:{fpath.name}",
                        hp_ring_map=arr,
                        hp_is_galactic=is_gal,
                    )
            raise KeyError(f"Could not find ebvMap/ebv array in {fpath}; keys={list(data.keys())}")

        # Try HealSparse first, then standard HEALPix FITS
        try:
            m = hsp.HealSparseMap.read(str(fpath))
            return cls(source_name=f"healsparse:{fpath.name}", hsp_map=m)
        except Exception:
            arr = np.asarray(hp.read_map(str(fpath), dtype=np.float32), dtype=np.float32)
            is_gal = _detect_galactic_frame(arr)
            return cls(
                source_name=f"healpix:{fpath.name}",
                hp_ring_map=arr,
                hp_is_galactic=is_gal,
            )

    @classmethod
    def _try_from_butler(
        cls,
        butler: Any,
        dataset_name: str,
        skymap: str,
    ) -> "EBVProvider | None":
        try:
            if hasattr(butler, "query_dataset_types"):
                dts = [
                    dt.name if hasattr(dt, "name") else str(dt)
                    for dt in butler.query_dataset_types("*ebv*")
                ]
            else:
                dts = []
            target = dataset_name if dataset_name in dts else (dts[0] if dts else None)
            if target is None:
                return None
            # Verify coverage can be fetched
            _ = butler.get(f"{target}.coverage", skymap=skymap)
            logger.info("Using Butler E(B-V) map dataset: %s", target)
            return cls(
                source_name=f"butler:{target}",
                butler_info=(butler, target, skymap),
            )
        except Exception:
            return None

    @classmethod
    def _try_from_rubin_sim_data(cls) -> "EBVProvider | None":
        # 1. Try rubin_scheduler CurrentAreaMap / read_dust_map (standard on RSP LSST kernel)
        try:
            from rubin_scheduler.scheduler.utils import CurrentAreaMap  # type: ignore[import-not-found]

            sag = CurrentAreaMap()
            dust_raw = getattr(sag, "dustmap", getattr(sag, "dust_map", None))
            if dust_raw is not None:
                arr = np.asarray(dust_raw, dtype=np.float32).ravel()
                is_gal = _detect_galactic_frame(arr)
                logger.info(
                    "Loaded SFD98 E(B-V) map from rubin_scheduler.CurrentAreaMap (nside=%d, galactic=%s)",
                    hp.npix2nside(len(arr)),
                    is_gal,
                )
                return cls(
                    source_name="rubin_scheduler:CurrentAreaMap",
                    hp_ring_map=arr,
                    hp_is_galactic=is_gal,
                )
        except Exception:
            pass

        # 2. Check $RUBIN_SIM_DATA_DIR/maps/DustMaps/dust_nside_*.npz directly
        candidate_dirs: list[Path] = []
        env_dir = os.environ.get("RUBIN_SIM_DATA_DIR")
        if env_dir:
            candidate_dirs.append(Path(env_dir) / "maps" / "DustMaps")
        # Common shared RSP / USDF data mounts
        for shared in (
            "/sdf/data/rubin/shared/rubin_sim_data/maps/DustMaps",
            "/rubin/rubin_sim_data/maps/DustMaps",
            str(Path.home() / "rubin_sim_data" / "maps" / "DustMaps"),
        ):
            candidate_dirs.append(Path(shared))

        for ddir in candidate_dirs:
            if ddir.is_dir():
                npz_files = sorted(ddir.glob("dust_nside_*.npz"), reverse=True)
                if npz_files:
                    return cls.from_file(npz_files[0])
        return None

    @classmethod
    def _try_from_dustmaps(cls) -> "EBVProvider | None":
        try:
            import dustmaps.sfd  # type: ignore[import-not-found]

            sfd = dustmaps.sfd.SFDQuery()
            # Probe a single coordinate to confirm SFD FITS files are downloaded
            _ = sfd(SkyCoord(ra=53.0 * u.deg, dec=-28.0 * u.deg, frame="icrs"))
            logger.info("Loaded SFD98 E(B-V) provider via dustmaps.sfd.SFDQuery")
            return cls(source_name="dustmaps:SFDQuery", sfd_query=sfd)
        except Exception:
            return None

    @classmethod
    def _from_rsp_tap_object(cls, cfg: MaskPipelineConfig) -> "EBVProvider":
        """Build a coarse HEALPix E(B-V) map from DP2 ``Object.ebv`` via RSP TAP (Tutorial 311.1)."""
        try:
            from lsst.rsp import get_tap_service  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "lsst.rsp is required for ebv_source='tap_object'. Run inside the RSP LSST kernel."
            ) from exc

        service = get_tap_service("tap")
        if cfg.region.mode != "cone":
            raise ValueError(
                "ebv_source='tap_object' is intended for cone pilot regions; "
                "use 'auto' or 'rubin_sim_sfd' for full-footprint DP2 runs."
            )
        adql = (
            "SELECT coord_ra, coord_dec, ebv "
            "FROM dp2_prep.Object "
            f"WHERE CONTAINS(POINT('ICRS', coord_ra, coord_dec), "
            f"CIRCLE('ICRS', {cfg.region.ra_deg:.6f}, {cfg.region.dec_deg:.6f}, {cfg.region.radius_deg:.6f})) = 1 "
            "AND ebv IS NOT NULL"
        )
        df = service.search(adql, maxrec=500_000).to_table().to_pandas()
        nside = 512
        m = hsp.HealSparseMap.make_empty(32, nside, dtype=np.float32)
        if len(df) > 0:
            pix = hpg.angle_to_pixel(
                nside,
                df["coord_ra"].to_numpy(dtype=np.float64),
                df["coord_dec"].to_numpy(dtype=np.float64),
                nest=True,
                degrees=True,
            )
            df_pix = np.unique(pix)
            # Fast mean per pixel via bincount
            uniq, inv = np.unique(pix, return_inverse=True)
            sums = np.bincount(inv, weights=df["ebv"].to_numpy(dtype=np.float64))
            cnts = np.bincount(inv)
            m.update_values_pix(uniq, (sums / np.maximum(1, cnts)).astype(np.float32))
            del df_pix
        return cls(source_name="rsp_tap:dp2_prep.Object.ebv", hsp_map=m)

    def eval_radec(self, ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
        """Evaluate $E(B-V)$ [mag] at ICRS ``(ra_deg, dec_deg)`` arrays."""
        ra = np.asarray(ra_deg, dtype=np.float64)
        dec = np.asarray(dec_deg, dtype=np.float64)
        if ra.size == 0:
            return np.empty(0, dtype=np.float64)

        if self._hsp_map is not None:
            vals = np.asarray(self._hsp_map.get_values_pos(ra, dec, lonlat=True), dtype=np.float64)
            bad = (~np.isfinite(vals)) | (vals < 0.0)
            vals[bad] = 0.0
            return vals

        if self._butler_info is not None:
            butler, ds_name, skymap = self._butler_info
            # Determine needed nside_cov=32 pixels for this batch only
            cov_pix = np.unique(hpg.angle_to_pixel(32, ra, dec, nest=True, degrees=True))
            chunk = butler.get(
                ds_name,
                skymap=skymap,
                parameters={"pixels": [int(p) for p in cov_pix]},
            )
            vals = np.asarray(chunk.get_values_pos(ra, dec, lonlat=True), dtype=np.float64)
            del chunk
            bad = (~np.isfinite(vals)) | (vals < 0.0)
            vals[bad] = 0.0
            return vals

        if self._hp_ring_map is not None and self._hp_nside is not None:
            if self._hp_is_galactic:
                sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic
                lon = sc.l.deg
                lat = sc.b.deg
            else:
                lon = ra
                lat = dec
            ipix = hp.ang2pix(self._hp_nside, lon, lat, nest=False, lonlat=True)
            vals = np.asarray(self._hp_ring_map[ipix], dtype=np.float64)
            bad = (~np.isfinite(vals)) | (vals < 0.0) | (vals == hp.UNSEEN)
            vals[bad] = 0.0
            return vals

        if self._sfd_query is not None:
            sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
            vals = np.asarray(self._sfd_query(sc), dtype=np.float64)
            vals[~np.isfinite(vals) | (vals < 0.0)] = 0.0
            return vals

        # Smooth Galactic latitude cosecant model: E(B-V) ~ 0.018 / sin(|b|)
        b_rad = _icrs_to_galactic_lat_rad(ra, dec)
        sin_b = np.maximum(np.sin(np.abs(b_rad)), np.sin(math.radians(2.5)))
        return 0.018 / sin_b

    def eval_nest_pixels(self, nside: int, nest_pixels: np.ndarray) -> np.ndarray:
        """Evaluate $E(B-V)$ [mag] at HEALPix NESTED pixel indices at ``nside``."""
        pix = np.asarray(nest_pixels, dtype=np.int64)
        if pix.size == 0:
            return np.empty(0, dtype=np.float64)
        ra, dec = hpg.pixel_to_angle(int(nside), pix, nest=True, degrees=True)
        return self.eval_radec(ra, dec)


def _icrs_to_galactic_lat_rad(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Fast vectorized ICRS -> Galactic latitude $b$ [radians] without SkyCoord overhead."""
    # IAU 1958 / Hipparcos North Galactic Pole in ICRS: (alpha_NGP, delta_NGP)
    ra_ngp = math.radians(192.85948)
    dec_ngp = math.radians(27.12825)
    ra_r = np.deg2rad(ra_deg)
    dec_r = np.deg2rad(dec_deg)
    sin_b = math.sin(dec_ngp) * np.sin(dec_r) + math.cos(dec_ngp) * np.cos(dec_r) * np.cos(
        ra_r - ra_ngp
    )
    return np.arcsin(np.clip(sin_b, -1.0, 1.0))


def _detect_galactic_frame(hp_ring_map: np.ndarray) -> bool:
    """Heuristic from Rubin WFD scripts: check whether (0, 0) has peak dust in Galactic vs ICRS."""
    nside = hp.npix2nside(len(hp_ring_map))
    pix_origin = hp.ang2pix(nside, 0.0, 0.0, nest=False, lonlat=True)
    # Galactic center (l=0, b=0) in ICRS is (266.405, -28.936)
    pix_gc_icrs = hp.ang2pix(nside, 266.405, -28.936, nest=False, lonlat=True)
    v_orig = float(hp_ring_map[pix_origin])
    v_icrs = float(hp_ring_map[pix_gc_icrs])
    return v_orig > v_icrs


def build_dust_extinction_mask(
    pipe: SkyMaskPipe,
    footprint_mask: hsp.HealSparseMap,
    cfg: MaskPipelineConfig,
    ebv_provider: EBVProvider,
    guard: MemoryGuard,
    rx_map: Mapping[str, float] | None = None,
    output_stage: str = "dustmask",
) -> tuple[hsp.HealSparseMap | None, DustMaskStats]:
    """Build a bit-packed ``SkyMaskPipe`` negative stage (``dustmask``) for dusty sightlines.

    Marks pixels inside ``footprint_mask`` where:
      - $E(B-V) > \\text{cfg.dust.max\\_ebv}$ (high dust attenuation / $R_V$ uncertainty), or
      - $|b| < \\text{cfg.dust.min\\_galactic\\_lat\\_deg}$ (SFD98 thermal source contamination zone
        highlighted in DP2 Tutorial 311.1).
    """
    dcfg = cfg.dust
    rx_used = dict(rx_map) if rx_map is not None else get_rubin_band_extinction_rx(
        r_v=dcfg.r_v, overrides=dcfg.r_x_overrides
    )

    order_dust = int(dcfg.order_sparse)
    nside_dust = 1 << order_dust
    order_cov = int(cfg.output.order_cov)
    nside_cov = 1 << order_cov

    order_foot = int(round(math.log2(int(footprint_mask.nside_sparse))))
    # Project active footprint pixels onto `order_dust` without materializing dense sky arrays
    dust_cand_chunks: list[np.ndarray] = []
    for fchunk in footprint_mask.iter_valid_pixels_by_covpix():
        arr = np.asarray(fchunk, dtype=np.int64)
        if arr.size == 0:
            continue
        if order_foot >= order_dust:
            shift = 2 * (order_foot - order_dust)
            dust_cand_chunks.append(np.unique(arr >> shift))
        else:
            shift = 2 * (order_dust - order_foot)
            n_sub = 1 << shift
            expanded = (arr[:, None] << shift) + np.arange(n_sub, dtype=np.int64)[None, :]
            dust_cand_chunks.append(expanded.ravel())

    cand_pix = (
        np.unique(np.concatenate(dust_cand_chunks))
        if dust_cand_chunks
        else np.empty(0, dtype=np.int64)
    )
    del dust_cand_chunks

    dustmask = hsp.HealSparseMap.make_empty(
        nside_cov, nside_dust, dtype=np.bool_, bit_packed=True
    )

    ebv_samples: list[np.ndarray] = []
    chunk_len = 250_000
    make_stage = (dcfg.max_ebv is not None) or (dcfg.min_galactic_lat_deg is not None)

    for start in range(0, cand_pix.size, chunk_len):
        sub_pix = cand_pix[start : start + chunk_len]
        ra, dec = hpg.pixel_to_angle(nside_dust, sub_pix, nest=True, degrees=True)
        ebv_vals = ebv_provider.eval_radec(ra, dec)
        # Keep decimated sample for quantile summary without storing millions of floats
        step = max(1, ebv_vals.size // 20_000)
        ebv_samples.append(ebv_vals[::step].astype(np.float32))

        if make_stage:
            bad = np.zeros(sub_pix.size, dtype=bool)
            if dcfg.max_ebv is not None:
                bad |= ebv_vals > float(dcfg.max_ebv)
            if dcfg.min_galactic_lat_deg is not None:
                b_deg = np.rad2deg(np.abs(_icrs_to_galactic_lat_rad(ra, dec)))
                bad |= b_deg < float(dcfg.min_galactic_lat_deg)
            if np.any(bad):
                dustmask.update_values_pix(sub_pix[bad], True)

        del ra, dec, ebv_vals

    if ebv_samples:
        cat_ebv = np.concatenate(ebv_samples)
        med_ebv: float | None = round(float(np.nanmedian(cat_ebv)), 5)
        p95_ebv: float | None = round(float(np.nanpercentile(cat_ebv, 95.0)), 5)
        max_ebv_val: float | None = round(float(np.nanmax(cat_ebv)), 5)
        del cat_ebv
    else:
        med_ebv = p95_ebv = max_ebv_val = None
    del ebv_samples, cand_pix

    n_valid = int(dustmask.n_valid)
    area_deg2 = float(dustmask.get_valid_area(degrees=True))

    if make_stage:
        extra_meta = {
            "ebv_source": ebv_provider.source_name,
            "r_v": float(dcfg.r_v),
            "max_ebv": dcfg.max_ebv,
            "min_galactic_lat_deg": dcfg.min_galactic_lat_deg,
            "median_ebv": med_ebv,
            "p95_ebv": p95_ebv,
            "pixels": n_valid,
            "area_deg2": area_deg2,
            "bit_packed": True,
        }
        if hasattr(pipe, "_finalize_stage"):
            pipe._finalize_stage(
                dustmask,
                default_name=output_stage,
                output_stage=output_stage,
                extra_meta=extra_meta,
            )
        else:
            setattr(pipe, output_stage, dustmask)

    gc.collect()
    guard.check("dustmask evaluated", collect_garbage=False)

    stats = DustMaskStats(
        enabled=True,
        ebv_source_used=ebv_provider.source_name,
        r_v=float(dcfg.r_v),
        r_x_by_band={b: round(float(v), 4) for b, v in rx_used.items()},
        applied_to_maglim=bool(dcfg.apply_to_maglim),
        max_ebv=float(dcfg.max_ebv) if dcfg.max_ebv is not None else None,
        min_galactic_lat_deg=(
            float(dcfg.min_galactic_lat_deg) if dcfg.min_galactic_lat_deg is not None else None
        ),
        order_sparse=order_dust,
        median_ebv_in_footprint=med_ebv,
        p95_ebv_in_footprint=p95_ebv,
        max_ebv_in_footprint=max_ebv_val,
        dustmask_valid_pixels=n_valid,
        dustmask_area_deg2=area_deg2,
    )
    return (dustmask if make_stage else None), stats
