# dp2-skykatana-mask

Memory-bounded (**32 GiB RAM RSP-safe**) spatial footprint, Milky Way dust, and bright-star mask builder for **Vera C. Rubin Observatory Data Preview 2 (DP2)** powered by [SkyKatana](https://skykatana.readthedocs.io/en/latest/).

Combines:
1. **DP2 Survey Property Maps (`sp_maps`)** ([dp2.lsst.io/products/maps/sp_maps.html](https://dp2.lsst.io/products/maps/sp_maps.html)):
   - Per-pixel minimal number of exposures per band (via `deepCoadd_exposure_time_consolidated_map_sum / nominal_exptime_sec`),
   - Per-pixel multi-band filter multiplicity (`min_filters`: at least $K$ of $N$ bands passing) plus mandatory `required_bands` (e.g. `["r", "i"]`),
   - Per-pixel **dust-dereddened** $5\sigma$ PSF limiting magnitudes (`deepCoadd_psf_maglim_consolidated_map_weighted_mean`),
   - Optional PSF seeing (`max_psf_fwhm_arcsec`), sky noise, sky background, or arbitrary custom `HealSparseMap` thresholds.
2. **Milky Way Dust Extinction Correction (`dustmask` & `psf_maglim` de-reddening)** ([DP2 Tutorial 311.1](https://dp2.lsst.io/tutorials/notebook/311/notebook-311-1.html)):
   - Uses Rubin $R_V = 3.1$ bandpass extinction coefficients $R_b = A_b / E(B-V)$ (`[4.81, 3.64, 2.70, 2.06, 1.58, 1.31]` for `u, g, r, i, z, y` via `rubin_sim.phot_utils.DustValues(r_v=3.1).r_x`),
   - De-reddens extragalactic coadd depth per pixel ($m_{b,\mathrm{lim}}^{\mathrm{dered}} = m_{b,\mathrm{lim}}^{\mathrm{obs}} - R_b E(B-V)_{\mathrm{SFD98}}$) during `propmask` streaming,
   - Builds a bit-packed negative **`dustmask`** stage excluding high-extinction sightlines ($E(B-V)$ > `max_ebv`) and low Galactic latitudes ($|b|$ < `min_galactic_lat_deg`) where 2D SFD98 maps lose reliability, while keeping foreground bright-star halo magnitudes in *observed* apparent LSST magnitudes.
3. **The Monster Reference Catalog (`the_monster`) + SkyKatana `radfunction`** ([DP2 Tutorial 204.2](https://dp2.lsst.io/tutorials/notebook/204/notebook-204-2.html)):
   - Dynamically discovers the latest `the_monster_YYYYMMDD` dataset in Gen3 Butler and queries only `htm7` (`HtmPixelization(7)`) trixels overlapping the valid footprint + maximum star-halo buffer,
   - Converts `SimpleCatalog` coordinates (`radians` -> `degrees`) and synthetic `monster_*_{band}_flux` (`nJy` -> observed LSST `ABmag`, $m_{\mathrm{AB}} = -2.5\log_{10}(f_{\mathrm{nJy}}) + 31.4000656$),
   - Computes per-star halo radii in degrees (`df['radius'] = R_arcsec / 3600`) via SkyKatana's **`radfunction(df, **kwargs)`** contract (`skykatana_lsst_radfunction` / `make_skykatana_radfunction`), calibrated for Rubin LSST bands (`u, g, r, i, z, y`, 3-piece ghost+wing+core `r_halo`, or multi-band envelope `multi` at `completeness` `0.80 | 0.85 | 0.90`), plus `build_monster_hats_star_mask_online` for HATS / `lsdb` Monster catalogs,
   - Pixelizes circular halos (`starmask`) and optional saturated CCD bleed-trail boxes (`boxmask`) into bit-packed `SkyMaskPipe` stages.
4. **Optional Galactic Plane & Bulge Exclusion (`mwmask`)**, streaming multi-order boolean combination (`SkyMaskPipe.combine`), IVOA MOC export, `fracdet` maps, and uniform random catalogs (`SkyMaskPipe.makerans`).

---

## Quickstart on the Rubin Science Platform (RSP)

1. Open a terminal in the RSP Notebook Aspect (`data.lsst.cloud`, up to **32 GiB RAM** pod) and clone this repository:
   ```bash
   git clone git@github.com:mardom/dp2-skykatana-mask.git
   cd dp2-skykatana-mask
   ```
2. Install `skykatana` (latest `main`) + `dp2-skykatana-mask` safely on top of `rubin-env` (`--user --no-deps`, preserving `lsst.daf.butler`, `rubin_sim`, `numpy`, `pyarrow`, and `healsparse`):
   ```bash
   ./scripts/setup_rsp.sh
   export PATH="$HOME/.local/bin:$PATH"
   ```
3. Inspect DP2 property maps and `the_monster` dataset in Butler:
   ```bash
   dp2-mask inspect-butler
   ```
4. Run the fast **ECDFS pilot** (~30–60 s) to validate thresholds:
   ```bash
   ./scripts/run_rsp_mask.sh configs/dp2_ecdfs_pilot.yaml
   ```
5. Run the **full DP2 footprint mask** (recommended inside `tmux` in an RSP terminal):
   ```bash
   tmux new -s dp2mask
   ./scripts/run_rsp_mask.sh configs/dp2_wfd_default.yaml
   ```
   Or interactively walk through every `SkyMaskPipe` stage (`propmask`, `dustmask`, `starmask`, `combine`, `plot`, `plot_moc`, `plot_fracmap`, `makerans`, `apply`) in `notebooks/01_DP2_Skykatana_Mask_RSP.ipynb` using the `LSST` kernel.

---

## Quickstart Locally (`lsdb_env-py313`)

```bash
# Install latest SkyKatana + editable dp2-skykatana-mask into lsdb_env-py313
/home/mardom/miniconda3/envs/lsdb_env-py313/bin/pip install --upgrade --no-deps git+https://github.com/samotracio/skykatana.git@main
/home/mardom/miniconda3/envs/lsdb_env-py313/bin/pip install --no-deps -e .

# Run full offline synthetic DP2 verification (includes synthetic Galactic cirrus cloud + Monster stars)
/home/mardom/miniconda3/envs/lsdb_env-py313/bin/dp2-mask demo-synthetic --output-dir outputs/demo_synthetic
```

---

## Why Standard `HealSparse` / `SkyKatana` Workflows OOM on 32 GiB RSP Pods (and How This Repo Solves It)

On the RSP, user Nublado containers run on large 64-core / 256–512 GiB Kubernetes nodes, **cgroup-capped at 32 GiB RAM**. Standard tools often query host `sysconf` RAM (seeing 512 GiB) or load all 6 bands $\times$ multiple float64 `nside=32768` maps at once, triggering an immediate Kubernetes `OOMKilled` (`SIGKILL`, exit 137).

| Stage | Naive Peak RAM (6,500 deg², `order=15`) | `dp2-skykatana-mask` Design | Peak Working Set |
| :--- | :---: | :--- | :---: |
| **Container Detection** | Reads host 256–512 GiB (`sysconf`) | Reads Nublado `MEM_LIMIT` & `/sys/fs/cgroup/memory.max` + caps OpenMP/BLAS threads to `2` | — |
| **Stage 1: `sp_maps` + Dust** | 18–35 GiB (all maps + `valid_pixels` in RAM) | Queries 12 KB `.coverage` first, pre-prunes tiles with `< min_filters` bands, streams `cov_batch_size=64` tiles with inline $A_b = R_b E(B-V)$ de-reddening directly into bit-packed `HealSparseMap` | **~0.9–1.5 GiB** |
| **Stage 2: `dustmask`** | 4–8 GiB float64 full-sky dust map | Evaluates $E(B-V)$ & $|b|$ per coverage batch into bit-packed `order_sparse=13` `dustmask` | **~0.2–0.4 GiB** |
| **Stage 3: `the_monster`** | 15–28 GiB (concatenating `SimpleCatalog` shards) | Queries buffered `htm7` shards one-by-one, applies `skykatana_lsst_radfunction` per shard, streams `(ra, dec, mag, radius)` to Snappy Parquet, runs `build_circ_mask(chunk_size=100_000, n_threads=2, bit_packed=True)` | **~3.5–5.5 GiB** |
| **Stage 4: `combine` + `makerans`** | 12+ GiB if unpacked | Row-by-row coverage streaming (`SkyMaskPipe.combine(..., bit_packed=True)`) + two-pass streaming `makerans` | **~1.5–2.2 GiB** |

Check any YAML config's analytic memory budget before running:
```bash
dp2-mask estimate-ram --config configs/dp2_wfd_default.yaml --area-deg2 6500
```

---

## Configuration Presets (`configs/`)

- `configs/dp2_ecdfs_pilot.yaml` — $1.5^\circ$ cone around ECDFS (`order_sparse=15`, keeps per-band `prop_{band}_mask` stages, `apply_to_maglim: true`, `max_ebv: 0.20`, `skykatana_rubin` `r`-band `completeness=0.85` + bleed boxes, 100k randoms).
- `configs/dp2_wfd_default.yaml` — Full DP2 survey mask (`min_filters: 4`, `required_bands: ["r", "i"]`, dust-dereddened `psf_maglim` + `max_ebv: 0.12` + `min_galactic_lat_deg: 5.0`, `the_monster` bright stars via `skykatana_rubin` at `order=15`, Galactic mask $|b| < 15^\circ$).
- `configs/dp2_deep_uniform.yaml` — Strict 6-band cosmology/LSS uniformity preset (`min_filters: 6`, PSF seeing FWHM cuts $< 1.2\text{--}1.3''$, strict dust cut `max_ebv: 0.08` & $|b| \ge 10^\circ$, 3-piece `r_halo` at `completeness: 0.90`).

### Filtering `lsdb` / HATS Catalogs & De-reddening Photometry

```python
import lsdb
from mocpy import MOC
from skykatana import SkyMaskPipe
from dp2_skykatana_mask import EBVProvider, DustExtinctionConfig, deredden_magnitude

# 1. Coarse server-side partition pruning with the exported IVOA MOC:
moc = MOC.from_fits("outputs/dp2_wfd_default/dp2_combined_mask_moc.fits")
cat = lsdb.open_catalog("path_or_url_to_dp2_hats", search_filter=lsdb.MOCSearch(moc))

# 2. Exact sub-arcsecond pixel filtering with SkyMaskPipe / HealSparse:
pipe = SkyMaskPipe.read("outputs/dp2_wfd_default/skykatana_pipe", verbose=False)
df_clean = pipe.apply(stage="mask", cat=df_partition, columns=("coord_ra", "coord_dec"))

# 3. Apply DP2 Tutorial 311.1 Milky Way dust correction to surviving extragalactic sources:
ebv_prov = EBVProvider.from_config(DustExtinctionConfig())
ebv = ebv_prov.eval_radec(df_clean["coord_ra"].to_numpy(), df_clean["coord_dec"].to_numpy())
df_clean["r_mag_dered"] = deredden_magnitude(df_clean["r_psfMag"].to_numpy(), ebv, band="r")
```

---

## Remote RSP & Kubernetes ("Help / Helm") Notes

- **Kubernetes / Helm on RSP**: Rubin's RSP is deployed via **Phalanx** (`lsst-sqre/phalanx`, Argo CD + Helm charts managed by SQuaRE). Science users inside Nublado pods intentionally do not have cluster `helm` or `kubectl` permissions and cannot alter the pod's 32 GiB cgroup limit; `./scripts/setup_rsp.sh` (`--user --no-deps`) + `./scripts/run_rsp_mask.sh` serve as the reproducible user-space equivalent.
- **Running / Assisting Remotely from Outside RSP**:
  - `dp2_skykatana_mask.butler_io.open_dp2_butler` natively supports external RSP **Gafaelfawr tokens** (`LabeledButlerFactory` / `RemoteButler`) via `--token-file ~/.rsp-tap.token` or `RSP_ACCESS_TOKEN` (never commit tokens to Git).
  - Long full-DP2 builds on RSP should be launched inside `tmux` in an RSP terminal so browser tab disconnects never interrupt execution.
