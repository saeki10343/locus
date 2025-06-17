# Locus Reproduction

This project aims to reproduce the evaluation of **Locus: Locating Bugs from Software Changes** on the Apache Tomcat repository. The included data is intentionally small and only contains a handful of bug reports and commits. To obtain results comparable to the paper, you need to build a dataset from the full Tomcat history and corresponding Bugzilla reports.

## Preparing the Dataset

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

1. Clone the full Tomcat repository:
   ```bash
   git clone https://github.com/apache/tomcat.git tomcat
   ```

2. Extract commit information (message, diffs) from the repository:
   ```bash
   python src/extract_commits.py tomcat data/commits.json --branch main
   ```
   Remove `--max-count` to process the entire history or set a limit if needed.

3. Generate bug reports linked to commits:
   ```bash
   python tools/collect_dataset.py
   ```
   The script scans commit messages for patterns like `Bug 12345` and fetches the corresponding report from Bugzilla. The resulting file `data/bug_reports.json` will contain the mapping used for evaluation.

Once the dataset is generated, rebuild the TF-IDF matrix and run evaluation:

```bash
python src/build_corpus.py
python src/evaluate_ranking.py
```

If you regenerate bug reports or commit data, make sure to rebuild the TF-IDF matrix before running `evaluate_ranking.py` so that the indices stay consistent.

Using the complete dataset and richer features should yield results closer to those reported in the Locus paper.
