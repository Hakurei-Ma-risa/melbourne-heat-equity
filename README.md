# Summer surface temperature and socio-economic inequity in Greater Melbourne

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20712253.svg)](https://doi.org/10.5281/zenodo.20712253)

Reproducible Google Earth Engine (GEE) workflow and processed data for an analysis of
**summer land surface temperature (LST) and socio-economic equity** across the **359
Statistical Area Level 2 (SA2)** units of Greater Melbourne, Australia, over **2000–2024**.

This repository accompanies a manuscript submitted to the *International Journal of Digital
Earth* (Taylor & Francis). All inputs are open data.

## Summary of findings

- Over 2000–2024 the metropolis **greened markedly** (NDVI rose significantly in ~86 % of SA2)
  with **no significant surface warming** — a post-drought, climate-driven trend rather than an
  urbanisation signal.
- Spatially, recent (2015–2024) summer surface temperature is **strongly inequitable**: the most
  disadvantaged communities are on average **1.69 °C hotter** than the most advantaged
  (Pearson *r* = −0.30 with socio-economic advantage, *p* < 0.001).
- The disparity is **robust to topography** (controlling for elevation strengthens it). In a
  heteroskedasticity-robust **spatial error model** controlling for vegetation, built-up density,
  elevation and spatial autocorrelation, the socio-economic effect **remains significant**
  (standardised β = −0.17, *p* = 0.009).
- A bootstrap **mediation analysis** shows the disparity is **partially (~42 %) mediated** by
  unequal green/grey infrastructure while retaining a significant **independent** direct effect
  (~58 %). Random Forest skill falls from R² = 0.88 (in-sample) to **0.31 under spatial-block
  cross-validation** (the honest, reported value).

## Repository structure

```
00_get_boundaries.py     Clip ABS SA2 shapefile to Greater Melbourne -> GeoJSON for GEE
01_gee_extract.py        GEE: cloud mask, multi-sensor harmonisation, indices, austral-summer
                         composites, SA2 zonal stats -> melb_sa2_annual_indices.csv (via Drive)
01b_smoketest.py         Optional GEE connectivity / auth smoke test
prep_abs.py              Merge ABS Census (GCP) + SEIFA -> abs_sa2_socioeconomic.csv
02_trend_analysis.py     Background temporal trends (Mann-Kendall + Sen's slope) -> sa2_trends.csv
03_heat_equity.py        Core: heat-equity statistics, maps -> heat_equity_results.csv, figures
04_spatial.py            Spatial autocorrelation (Moran's I) and spatial regression
05a_elevation.py         Per-SA2 mean SRTM elevation -> sa2_elevation.csv
05b_revision_stats.py    Elevation control, VIF + spatial LM/Robust-LM diagnostics, bootstrap mediation, spatial-block CV

requirements.txt         Python dependencies
CITATION.cff             How to cite this repository
```

### Processed data included (lets you reproduce 03–05 without GEE or raw ABS downloads)

| File | Description |
|---|---|
| `melb_sa2_annual_indices.csv` | GEE output: annual austral-summer NDVI/NDBI/NDWI/LST per SA2, 2000–2024 |
| `abs_sa2_socioeconomic.csv` | ABS Census 2021 (G01/G02) + SEIFA 2021 (IRSD/IRSAD) per SA2 |
| `sa2_elevation.csv` | Mean SRTM elevation per SA2 |
| `heat_equity_results.csv` | Core analysis table (recent-mean LST, indices, SEIFA, area) |
| `sa2_trends.csv` | Per-SA2 Mann-Kendall/Sen's-slope trend results |
| `SA2_GreaterMelbourne.geojson`, `GreaterMelbourne_GCCSA.geojson` | Derived study-area boundaries |
| `fig_heat_equity.png`, `fig_maps_heat_seifa.png` | Main figures |

## Data sources (raw inputs, all open)

The large raw inputs are **not committed** (see `.gitignore`); download them as follows:

- **ABS ASGS 2021 SA2 boundaries** (shapefile) → unzip into `sa2_shp/`:
  https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3
- **ABS 2021 Census General Community Profile (GCP), VIC, SA2** DataPack → `gcp_vic/`:
  https://www.abs.gov.au/census/find-census-data/datapacks
- **ABS SEIFA 2021** (IRSD / IRSAD, SA2): https://www.abs.gov.au/statistics/people/people-and-communities/socio-economic-indexes-areas-seifa-australia
- **Landsat Collection 2 Level-2** surface reflectance & surface temperature — accessed via Google
  Earth Engine (no manual download).
- **SRTM** elevation — accessed via Google Earth Engine (`USGS/SRTMGL1_003`).

## Reproduce

```bash
pip install -r requirements.txt
earthengine authenticate          # one-time browser login (requires a GEE Cloud project)

# --- Full pipeline (from raw data) ---
python 00_get_boundaries.py        # -> SA2_GreaterMelbourne.geojson  (upload as a GEE asset)
python 01_gee_extract.py           # triggers GEE export -> CSV lands in your Google Drive
#   wait for the task at https://code.earthengine.google.com (Tasks tab), download the CSV here
python prep_abs.py                 # -> abs_sa2_socioeconomic.csv
python 02_trend_analysis.py        # background trends
python 03_heat_equity.py           # core heat-equity analysis + figures
python 04_spatial.py               # spatial autocorrelation + regression
python 05a_elevation.py            # -> sa2_elevation.csv
python 05b_revision_stats.py       # elevation control + VIF/LM diagnostics + mediation + spatial CV

# --- Shortcut ---
# The processed CSVs above are already included, so steps 03–05 run directly.
```

**Configuration:** set your own GEE Cloud project id (replace `journal-34649077` in
`01_gee_extract.py`, `01b_smoketest.py`, `05a_elevation.py`) and the matching SA2 asset path.

### Note on IPv6

On some Linux hosts with a broken/advertised-but-dead IPv6 route, GEE/google-auth calls can hang
on TCP connect. `05a_elevation.py` includes a force-IPv4 `getaddrinfo` shim; if other scripts
hang during auth, prefer IPv4 at the OS level (e.g. `precedence ::ffff:0:0/96 100` in `/etc/gai.conf`).

## Citation

If you use this code or data, please cite the repository (see `CITATION.cff`) and the accompanying
article once published. This release is archived on Zenodo:
**[10.5281/zenodo.20712253](https://doi.org/10.5281/zenodo.20712253)** (concept DOI — always resolves
to the latest version).

## License

- **Code:** MIT (see [`LICENSE`](LICENSE)).
- **Processed data products** derived from ABS inputs inherit the **ABS Creative Commons
  Attribution 4.0 (CC BY 4.0)** terms; please attribute the Australian Bureau of Statistics.

## Acknowledgements

Australian Bureau of Statistics (Census, SEIFA, ASGS boundaries), USGS/NASA (Landsat, SRTM), and
Google Earth Engine for open data and computing infrastructure.
