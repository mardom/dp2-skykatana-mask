"""Butler access helpers for DP2 survey property maps and The Monster catalog.

Compatible with both ``DirectButler`` (USDF / NERSC) and ``RemoteButler`` on the
Rubin Science Platform (`data.lsst.cloud`), using modern non-deprecated
``butler.query_dataset_types`` / ``butler.query_datasets`` APIs first with
automatic fallback to ``butler.registry.*`` for older stacks.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from .config import MaskPipelineConfig

logger = logging.getLogger(__name__)


_RSP_STACK_LOADERS = (
    Path("/opt/lsst/software/stack/loadLSST.bash"),
    Path("/opt/lsst/software/stack/loadLSST.sh"),
)


def find_rsp_stack_loader() -> Path | None:
    """Return the RSP LSST stack loader script if running inside an RSP/Rubin container."""
    candidates = list(_RSP_STACK_LOADERS)
    lsst_home = os.environ.get("LSST_HOME", "").strip()
    if lsst_home:
        candidates.append(Path(lsst_home) / "loadLSST.bash")
        candidates.append(Path(lsst_home) / "loadLSST.sh")
    for path in candidates:
        if path.is_file():
            return path
    return None


def open_dp2_butler(cfg: MaskPipelineConfig) -> Any:
    """Instantiate an LSST Gen3 ``Butler`` (inside RSP or remotely via token).

    Parameters
    ----------
    cfg : MaskPipelineConfig
        Pipeline configuration specifying repository, collections, and optional
        external Gafaelfawr token file.
    """
    if cfg.butler_Repository_index:
        os.environ["DAF_BUTLER_REPOSITORY_INDEX"] = str(cfg.butler_Repository_index)

    try:
        from lsst.daf.butler import Butler
    except ImportError as exc:
        rsp_loader = find_rsp_stack_loader()
        if rsp_loader is not None:
            raise ImportError(
                f"lsst.daf.butler is not loaded in this RSP terminal shell "
                f"(found stack loader at '{rsp_loader}', active Python is '{sys.executable}').\n"
                f"In RSP terminals (unlike the Jupyter LSST kernel), activate EUPS first:\n"
                f"  source {rsp_loader}\n"
                f"  setup lsst_distrib\n"
                f"And if you ran ./scripts/setup_rsp.sh before activating lsst_distrib, "
                f"re-run ./scripts/setup_rsp.sh once so ~/.local/bin/dp2-mask binds to rubin-env."
            ) from exc
        raise ImportError(
            "lsst.daf.butler is not installed in the active Python environment. "
            "On the RSP terminal, run `source /opt/lsst/software/stack/loadLSST.bash && setup lsst_distrib` "
            "(or `./scripts/setup_rsp.sh`), use the RSP `LSST` notebook kernel, "
            "or run `dp2-mask demo-synthetic` for offline local testing."
        ) from exc

    token = _resolve_external_token(cfg.access_token_file)
    if token is not None:
        from lsst.daf.butler import LabeledButlerFactory

        logger.info(
            "Opening RemoteButler for repo=%r collections=%r using external token",
            cfg.butler_repo,
            cfg.butler_collections,
        )
        factory = LabeledButlerFactory()
        butler = factory.create_butler(label=cfg.butler_repo, access_token=token)
        # Attach default collections via clone if supported
        if hasattr(butler, "clone") and cfg.butler_collections:
            butler = butler.clone(collections=cfg.butler_collections)
        return butler

    logger.info(
        "Opening Butler(repo=%r, collections=%r)",
        cfg.butler_repo,
        cfg.butler_collections,
    )
    return Butler(cfg.butler_repo, collections=cfg.butler_collections)


def _resolve_external_token(token_file: str | None) -> str | None:
    """Read an optional external RSP token without ever logging its value."""
    if token_file:
        path = Path(token_file).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Configured RSP access_token_file not found: {path}")
        tok = path.read_text(encoding="utf-8").strip()
        if not tok:
            raise ValueError(f"Configured RSP access_token_file is empty: {path}")
        return tok

    env_tok = os.environ.get("RSP_ACCESS_TOKEN", "").strip()
    if env_tok:
        return env_tok
    return None


def list_dataset_type_names(butler: Any, pattern: str = "*") -> list[str]:
    """List dataset type names matching ``pattern`` across RemoteButler and DirectButler."""
    if hasattr(butler, "query_dataset_types"):
        dts = butler.query_dataset_types(pattern)
        return sorted(dt.name if hasattr(dt, "name") else str(dt) for dt in dts)
    dts = butler.registry.queryDatasetTypes(expression=pattern)
    return sorted(dt.name if hasattr(dt, "name") else str(dt) for dt in dts)


def list_consolidated_maps(butler: Any) -> list[str]:
    """Return all DP2 consolidated survey property map dataset types in ``butler``."""
    return list_dataset_type_names(butler, "*consolidated_map*")


def get_sp_map_coverage(butler: Any, dataset_type: str, band: str, skymap: str) -> Any:
    """Fetch the lightweight ``HealSparseCoverage`` (~12 KB) for a DP2 property map."""
    return butler.get(f"{dataset_type}.coverage", band=band, skymap=skymap)


def get_sp_map_chunk(
    butler: Any,
    dataset_type: str,
    band: str,
    skymap: str,
    cov_pixels: Sequence[int],
    nside_sparse: int,
    native_nside_sparse: int = 32768,
) -> Any:
    """Fetch a subset of coverage pixels from a DP2 ``HealSparseMap``."""
    params: dict[str, Any] = {"pixels": [int(p) for p in cov_pixels]}
    if int(nside_sparse) < int(native_nside_sparse):
        params["degrade_nside"] = int(nside_sparse)
    return butler.get(dataset_type, band=band, skymap=skymap, parameters=params)


def discover_monster_dataset(butler: Any, configured_name: str = "auto") -> str:
    """Resolve the versioned ``the_monster_*`` reference catalog dataset type in DP2."""
    if configured_name and configured_name != "auto":
        return configured_name

    candidates = [
        name
        for name in list_dataset_type_names(butler, "*monster*")
        if name.startswith("the_monster")
    ]
    if not candidates:
        raise RuntimeError(
            "Could not discover any `the_monster*` dataset type in Butler. "
            "Specify `monster_stars.dataset_type` explicitly (e.g. 'the_monster_20250219')."
        )
    chosen = sorted(candidates)[-1]
    logger.info("Discovered Monster reference catalog dataset type: %s", chosen)
    return chosen


def query_monster_refs_for_htm7(
    butler: Any,
    dataset_type: str,
    htm7_ids: Sequence[int] | None,
    collections: Sequence[str] | None = None,
    batch_size: int = 200,
) -> list[Any]:
    """Query Butler ``DatasetRef`` shards for ``the_monster`` across ``htm7`` trixels."""
    colls = list(collections) if collections else None

    def _query_where(where_expr: str | None) -> list[Any]:
        if hasattr(butler, "query_datasets"):
            kwargs: dict[str, Any] = {"find_first": True}
            if colls is not None:
                kwargs["collections"] = colls
            if where_expr:
                kwargs["where"] = where_expr
            return list(butler.query_datasets(dataset_type, **kwargs))

        # Fallback for older registry API
        reg_kwargs: dict[str, Any] = {}
        if colls is not None:
            reg_kwargs["collections"] = colls
        else:
            reg_kwargs["collections"] = butler.collections
        if where_expr:
            reg_kwargs["where"] = where_expr
        refs = butler.registry.queryDatasets(dataset_type, **reg_kwargs)
        if hasattr(refs, "findFirst"):
            refs = refs.findFirst()
        return list(refs)

    if htm7_ids is None:
        refs = _query_where(None)
        if not refs and colls is None:
            refs = _retry_with_refcats_collection(butler, dataset_type, None)
        return _deduplicate_refs_by_htm7(refs)

    unique_ids = sorted({int(i) for i in htm7_ids})
    if not unique_ids:
        return []

    all_refs: list[Any] = []
    used_refcats_fallback = False

    for start in range(0, len(unique_ids), max(1, int(batch_size))):
        chunk = unique_ids[start : start + int(batch_size)]
        id_csv = ", ".join(str(x) for x in chunk)
        where = f"htm7 IN ({id_csv})"
        try:
            batch_refs = _query_where(where)
        except Exception:
            batch_refs = []
        if not batch_refs and colls is None and not used_refcats_fallback and start == 0:
            fallback_refs = _retry_with_refcats_collection(butler, dataset_type, where)
            if fallback_refs:
                used_refcats_fallback = True
                colls = ["refcats"]
                batch_refs = fallback_refs
        all_refs.extend(batch_refs)

    return _deduplicate_refs_by_htm7(all_refs)


def _retry_with_refcats_collection(
    butler: Any,
    dataset_type: str,
    where_expr: str | None,
) -> list[Any]:
    """Fallback search in `refcats` collections if user passed narrow run collections."""
    candidate_colls = ["refcats", "dp2"]
    for c in candidate_colls:
        try:
            if hasattr(butler, "query_datasets"):
                kw: dict[str, Any] = {"collections": [c], "find_first": True}
                if where_expr:
                    kw["where"] = where_expr
                res = list(butler.query_datasets(dataset_type, **kw))
                if res:
                    logger.info("Found %s shards in collection %r", dataset_type, c)
                    return res
        except Exception:
            continue
    return []


def _deduplicate_refs_by_htm7(refs: Sequence[Any]) -> list[Any]:
    seen: set[Any] = set()
    unique: list[Any] = []
    for ref in refs:
        data_id = getattr(ref, "dataId", None)
        key = data_id.get("htm7") if hasattr(data_id, "get") else id(ref)
        if key not in seen:
            seen.add(key)
            unique.append(ref)
    return unique


def resolve_monster_flux_columns(
    available_columns: Sequence[str],
    reference_bands: Sequence[str],
    explicit_flux_column: str | None = None,
) -> list[str]:
    """Resolve synthetic flux column(s) in ``the_monster`` (DMTN-277 schema).

    In ``the_monster_20250219``, LSSTCam bands map to ``monster_ComCam_{band}_flux``.
    Future Monster releases may expose ``monster_LSST_{band}_flux`` or
    ``monster_LSSTCam_{band}_flux``. This resolver checks them in priority order,
    with final fallback to Gaia ``phot_g_mean_flux``.
    """
    col_set = set(available_columns)

    if explicit_flux_column:
        if explicit_flux_column not in col_set:
            raise KeyError(
                f"Configured explicit_flux_column={explicit_flux_column!r} not in Monster "
                f"columns (sample: {sorted(col_set)[:15]})"
            )
        return [explicit_flux_column]

    resolved: list[str] = []
    for band in reference_bands:
        prefixes = (
            f"monster_LSST_{band}_flux",
            f"monster_LSSTCam_{band}_flux",
            f"monster_ComCam_{band}_flux",
            f"monster_DES_{band}_flux",
        )
        for cand in prefixes:
            if cand in col_set:
                if cand not in resolved:
                    resolved.append(cand)
                break

    if resolved:
        return resolved

    for gaia_fallback in ("phot_g_mean_flux", "monster_Gaia_G_flux", "gaia_G_flux"):
        if gaia_fallback in col_set:
            logger.warning(
                "No band-specific monster_*_flux columns matched %r; falling back to %s",
                reference_bands,
                gaia_fallback,
            )
            return [gaia_fallback]

    raise KeyError(
        f"Could not find any Monster synthetic flux columns for bands {list(reference_bands)} "
        f"in schema columns: {sorted(col_set)}"
    )
