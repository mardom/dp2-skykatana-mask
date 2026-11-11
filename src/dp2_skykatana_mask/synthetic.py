"""Synthetic DP2 Gen3 Butler emulator for local offline verification and CI tests.

Generates realistic multi-band ``HealSparseMap`` survey property maps (matching DP2
`sp_maps` schema at ``nside_coverage=32``) and ``htm7``-sharded ``the_monster_20250219``
catalogs (storing ``coord_ra``/``coord_dec`` in radians and ``monster_ComCam_{band}_flux``
in ``nJy``, matching LSST ``SimpleCatalog`` conventions on the RSP).
"""

from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass
from typing import Any, Sequence

import healsparse as hsp
import hpgeom as hpg
import numpy as np

from .monster_stars import AB_MAG_ZEROPOINT_NJY


@dataclass(frozen=True)
class _MockDatasetType:
    name: str


@dataclass(frozen=True)
class _MockDatasetRef:
    datasetType: _MockDatasetType
    dataId: dict[str, Any]


class _MockSchema:
    def __init__(self, names: Sequence[str]) -> None:
        self._names = list(names)

    def getNames(self) -> list[str]:
        return list(self._names)


class MockSimpleCatalog:
    """Emulates ``lsst.afw.table.SimpleCatalog`` (radians coords, ``nJy`` fluxes)."""

    def __init__(self, columns: dict[str, np.ndarray]) -> None:
        self._columns = {k: np.asarray(v) for k, v in columns.items()}
        self.schema = _MockSchema(list(self._columns.keys()))

    def __getitem__(self, key: str) -> np.ndarray:
        return self._columns[key]

    def __len__(self) -> int:
        if not self._columns:
            return 0
        first = next(iter(self._columns.values()))
        return int(first.shape[0])


def abmag_to_flux_njy(mag_ab: np.ndarray | float) -> np.ndarray:
    m = np.asarray(mag_ab, dtype=np.float64)
    return 10.0 ** (-0.4 * (m - AB_MAG_ZEROPOINT_NJY))


class SyntheticDP2Butler:
    """In-memory Gen3 Butler double exposing DP2 `sp_maps` and `the_monster_20250219`."""

    CONSOLIDATED_MAPS = (
        "deepCoadd_exposure_time_consolidated_map_sum",
        "deepCoadd_psf_maglim_consolidated_map_weighted_mean",
        "deepCoadd_psf_size_consolidated_map_weighted_mean",
        "deepCoadd_sky_noise_consolidated_map_weighted_mean",
        "deepCoadd_sky_background_consolidated_map_weighted_mean",
    )
    MONSTER_DATASET = "the_monster_20250219"

    def __init__(
        self,
        center_ra_deg: float = 53.1222,
        center_dec_deg: float = -28.0870,
        radius_deg: float = 1.6,
        nside_cov: int = 32,
        nside_sparse: int = 4096,  # order=12 for fast unit tests, or 16384/32768 for demos
        n_stars: int = 180,
        seed: int = 12345,
    ) -> None:
        self.collections = ["dp2"]
        self.center_ra_deg = float(center_ra_deg)
        self.center_dec_deg = float(center_dec_deg)
        self.radius_deg = float(radius_deg)
        self.nside_cov = int(nside_cov)
        self.nside_sparse = int(nside_sparse)
        self._maps: dict[tuple[str, str], hsp.HealSparseMap] = {}
        self._monster_shards: dict[int, MockSimpleCatalog] = {}

        self._populate_sp_maps(seed=seed)
        self._populate_monster_shards(n_stars=n_stars, seed=seed + 99)

    def _populate_sp_maps(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        bands = ("u", "g", "r", "i", "z", "y")
        nominal_maglim = {"u": 24.4, "g": 25.6, "r": 25.3, "i": 24.9, "z": 24.2, "y": 23.0}
        nominal_visits = {"u": 8.0, "g": 14.0, "r": 18.0, "i": 18.0, "z": 12.0, "y": 10.0}

        all_pix = hpg.query_circle(
            self.nside_sparse,
            self.center_ra_deg,
            self.center_dec_deg,
            self.radius_deg,
            nest=True,
            degrees=True,
        )
        ra, dec = hpg.pixel_to_angle(self.nside_sparse, all_pix, nest=True, degrees=True)
        dra = (ra - self.center_ra_deg) * math.cos(math.radians(self.center_dec_deg))
        ddec = dec - self.center_dec_deg
        radial_frac = np.hypot(dra, ddec) / max(1e-6, self.radius_deg)

        for b_idx, band in enumerate(bands):
            # Vary coverage slightly by band so multi-band K-of-N (`min_filters`) is exercised!
            band_max_frac = 1.0 - 0.05 * (b_idx % 3)
            keep = radial_frac <= band_max_frac
            bpix = all_pix[keep]
            r_frac = radial_frac[keep]

            # Exposure time drops near field boundary (simulating dithered visit roll-off)
            n_exp = np.maximum(
                1.0,
                np.round(nominal_visits[band] * (1.0 - 0.72 * (r_frac**2))),
            )
            exptime = (n_exp * 30.0).astype(np.float32)

            # Limiting magnitude scales as +1.25 * log10(exptime / 30s) plus small spatial noise
            maglim = (
                nominal_maglim[band]
                - 0.8 * (r_frac**2)
                + rng.normal(0.0, 0.05, size=bpix.size)
            ).astype(np.float32)

            # PSF determinant radius in 0.2" pixels (~1.8 - 2.6 pix -> FWHM ~ 0.85" - 1.22")
            psf_size = (1.85 + 0.55 * (r_frac**2)).astype(np.float32)
            sky_noise = np.full(bpix.size, 8.5 + b_idx, dtype=np.float32)
            sky_bg = np.full(bpix.size, 150.0 + 20.0 * b_idx, dtype=np.float32)

            for ds_name, values in (
                ("deepCoadd_exposure_time_consolidated_map_sum", exptime),
                ("deepCoadd_psf_maglim_consolidated_map_weighted_mean", maglim),
                ("deepCoadd_psf_size_consolidated_map_weighted_mean", psf_size),
                ("deepCoadd_sky_noise_consolidated_map_weighted_mean", sky_noise),
                ("deepCoadd_sky_background_consolidated_map_weighted_mean", sky_bg),
            ):
                m = hsp.HealSparseMap.make_empty(
                    self.nside_cov, self.nside_sparse, dtype=np.float32
                )
                m.update_values_pix(bpix, values)
                self._maps[(ds_name, band)] = m

    def _populate_monster_shards(self, n_stars: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        # Sample stars across and slightly outside the circular field (testing buffer stars!)
        u_rad = np.sqrt(rng.uniform(0.0, 1.0, size=n_stars)) * (self.radius_deg * 1.15)
        theta = rng.uniform(0.0, 2.0 * math.pi, size=n_stars)
        dec_deg = self.center_dec_deg + u_rad * np.sin(theta)
        ra_deg = (
            self.center_ra_deg
            + (u_rad * np.cos(theta)) / math.cos(math.radians(self.center_dec_deg))
        ) % 360.0

        # Include bright stars (9..17.5 mag) and faint stars (18.5..21 mag that shouldn't be masked)
        mags_r = np.concatenate(
            [
                np.array([9.2, 10.4, 11.2], dtype=np.float64),
                rng.uniform(12.0, 20.5, size=max(0, n_stars - 3)),
            ]
        )[:n_stars]

        # Partition into 4 mock htm7 shards
        n_shards = 4
        shard_ids = [140001 + i for i in range(n_shards)]
        for s_idx, htm_id in enumerate(shard_ids):
            sl = slice(s_idx, n_stars, n_shards)
            ra_rad = np.deg2rad(ra_deg[sl])
            dec_rad = np.deg2rad(dec_deg[sl])
            mr = mags_r[sl]
            cols: dict[str, np.ndarray] = {
                "id": np.arange(1_000_000 + s_idx * 10_000, 1_000_000 + s_idx * 10_000 + mr.size),
                "coord_ra": ra_rad,
                "coord_dec": dec_rad,
            }
            for b_off, band in enumerate(("u", "g", "r", "i", "z", "y")):
                color_shift = 0.25 * (2 - b_off)
                cols[f"monster_ComCam_{band}_flux"] = abmag_to_flux_njy(mr + color_shift)
            self._monster_shards[htm_id] = MockSimpleCatalog(cols)

    def query_dataset_types(self, pattern: str = "*") -> list[_MockDatasetType]:
        all_names = [*self.CONSOLIDATED_MAPS, self.MONSTER_DATASET]
        return [_MockDatasetType(n) for n in all_names if fnmatch.fnmatch(n, pattern)]

    def query_datasets(
        self,
        dataset_type: str,
        collections: Sequence[str] | None = None,
        where: str | None = None,
        find_first: bool = True,
    ) -> list[_MockDatasetRef]:
        del collections, find_first
        if dataset_type != self.MONSTER_DATASET:
            return []
        selected_ids = sorted(self._monster_shards.keys())
        if where and "htm7 IN" in where:
            inside = where.split("(", 1)[1].rsplit(")", 1)[0]
            wanted = {int(tok.strip()) for tok in inside.split(",") if tok.strip()}
            match = [hid for hid in selected_ids if hid in wanted]
            # If real lsst.sphgeom computed real HTM7 IDs vs mock IDs, return all mock shards
            if match:
                selected_ids = match
        return [
            _MockDatasetRef(_MockDatasetType(self.MONSTER_DATASET), {"htm7": hid})
            for hid in selected_ids
        ]

    def get(
        self,
        dataset_or_ref: Any,
        band: str | None = None,
        skymap: str | None = None,
        parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        del skymap, kwargs
        if isinstance(dataset_or_ref, _MockDatasetRef):
            htm_id = int(dataset_or_ref.dataId["htm7"])
            return self._monster_shards[htm_id]

        ds_name = str(dataset_or_ref)
        if ds_name.endswith(".coverage"):
            base_ds = ds_name[: -len(".coverage")]
            assert band is not None
            return self._maps[(base_ds, band)]._cov_map

        assert band is not None
        full_map = self._maps[(ds_name, band)]
        if not parameters:
            return full_map

        sub = full_map
        if "pixels" in parameters:
            cov_pix = np.asarray(parameters["pixels"], dtype=np.int64)
            out = hsp.HealSparseMap.make_empty(
                full_map.nside_coverage, full_map.nside_sparse, dtype=full_map.dtype
            )
            shift = 2 * int(round(math.log2(full_map.nside_sparse // full_map.nside_coverage)))
            vpix = full_map.valid_pixels
            in_cov = np.isin(vpix >> shift, cov_pix)
            sel = vpix[in_cov]
            if sel.size > 0:
                out.update_values_pix(sel, full_map.get_values_pix(sel))
            sub = out

        if "degrade_nside" in parameters and int(parameters["degrade_nside"]) < sub.nside_sparse:
            sub = sub.degrade(nside_out=int(parameters["degrade_nside"]), reduction="mean")
        return sub
