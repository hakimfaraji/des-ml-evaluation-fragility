# des-ml-evaluation-fragility

Reproducibility package for the manuscript:

**When High Accuracy Misleads in Literature-Derived Machine Learning for Deep Eutectic Solvent Recommendation**

This repository contains the curated dataset, leakage-clean modeling workflow, audit outputs, evaluation outputs, feature-importance results, statistical-comparison outputs, and revised figures used to support the manuscript and Supporting Information.

## Repository contents

```text
data/
  Dataset_S1.csv

code/
  Code_S1.ipynb
  revision_feature_importance_and_effect_sizes.py

outputs/
  audits/              # dataset schema, split audit, feature filtering, train/test DOI audit
  pairwise/            # train-only pairwise dataset summaries and dependency audit
  ranking/             # baseline/hybrid ranking outputs and revised Table 4 support
  statistics/          # exact sign tests, effect sizes, classification metrics
  uncertainty/         # group-level uncertainty estimates
  sanity_baselines/    # random and polarity-mismatch heuristic baselines
  feature_importance/  # permutation feature importance outputs

tables_for_si/         # CSV exports of revised SI tables S7-S25
figures/               # revised Figure 5 and simplified graphical abstract
manuscript/            # place final manuscript/SI files here before release
```

## Data

- `data/Dataset_S1.csv`: curated experiment-level modeling dataset used in the final leakage-clean pipeline.

## Code

- `code/Code_S1.ipynb`: reproducible leakage-clean modeling workflow.
- `code/revision_feature_importance_and_effect_sizes.py`: standalone reference script documenting revision addendum analyses.

The workflow includes DOI-grouped splitting, train-only preprocessing, train-only pairwise construction, test-only within-group inference, sanity-baseline evaluation, uncertainty estimation, exact sign tests, feature importance, and paired effect-size analysis.

## Reproducing the analysis

The main notebook was originally run in Google Colab. To reproduce the analysis:

1. Open `code/Code_S1.ipynb`.
2. Upload or point the notebook to `data/Dataset_S1.csv`.
3. Run the notebook sequentially.
4. The notebook writes outputs to the configured `OUTPUT_DIR`.
5. The revision addendum cells generate the feature-importance and paired-effect-size outputs used in the revised manuscript and SI.

## Key revised outputs

- `outputs/feature_importance/feature_importance_top20.csv` supports Table S19.
- `outputs/statistics/paired_ranking_effect_sizes.csv` supports Table S20.
- `outputs/ranking/group_ranking_metrics_all_and_comparable.csv` supports Table S7 and the revised main Table 4.
- `outputs/sanity_baselines/sanity_baseline_comparison.csv` supports Table S17.
- `tables_for_si/` contains CSV versions of revised tables introduced or updated during peer-review revision.

## Version history

### v1.2 — author-metadata corrected peer-review revision package

- Updated package metadata to match the current revised manuscript author list.
- Removed the author no longer included in the revised manuscript from `CITATION.cff` and `zenodo_metadata.md`.
- Included the revised peer-review reproducibility contents: dataset, code, audit outputs, ranking/classification outputs, sanity-baseline results, permutation feature-importance outputs, paired effect-size outputs, revised tables, revised Figure 5, and simplified graphical abstract.
- Added dataset-flow and validation-audit table exports.
- Added representative corrected-entry table for extraction validation.
- Added aligned Table S7/Table S17 group-ranking metrics and corrected uncertainty-table values.
- Added descriptor-rationale, alternative-target, and AI-ready reporting checklist tables.

### v1.0 — original reproducibility package

- Initial public dataset, code, and core leakage-clean audit/evaluation outputs.

## License

Code is released under the MIT License. The curated dataset and tabular outputs are intended for scholarly reuse with attribution to the manuscript and Zenodo record.

## Citation

Please cite the associated manuscript and the Zenodo DOI assigned to the released version of this repository.
