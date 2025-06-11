# Write README.md file with content
content = """# README – Cirrus Figures Notebook

This repository contains the notebook and supporting data sets used to generate all figures for the manuscript on Amazonian cirrus clouds.

---

## 1. Files & folders

| Path | Purpose |
|------|---------|
| **main.ipynb** | Jupyter notebook that ingests the data sets below and exports publication‑ready figures. |
| **df_cirrus_metadata.txt** | df_cirrus dictionary |
| **df_cirrus_part1.csv** | Cirrus‑layer properties (≈ first third of the record). |
| **df_cirrus_part2.csv** | Second third of the cirrus data set. |
| **df_cirrus_part3.csv** | Final third of the cirrus data set. |
| **gouveia_freq_wet.csv** | Monthly cirrus-frequency climatology for the wet season. |
| **gouveia_freq_dry.csv** | As above, dry season. |
| **gouveia_freq_trans.csv** | Transition season. |
| **gouveia_freq_total.csv** | Full-year climatology (wet + dry + transition). |
| **precip_2011_2017_cmorph.csv** | 30‑min CMORPH precipitation time series (Jul 2011 – Dec 2017). |
| **precip_2011_2017_cmorph_daily.csv** | Same data aggregated to daily totals. |

---

## 2. Quick‑start

```bash
# 1. Create environment (example with conda)
conda create -n cirrus_figures python=3.11 pandas numpy matplotlib seaborn

# 2. Activate
conda activate cirrus_figures

# 3. Clone repo / place all CSVs alongside main.ipynb, then
jupyter lab
