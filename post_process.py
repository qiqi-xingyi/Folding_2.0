# --*-- conding:utf-8 --*--
# @time:10/21/25 14:23
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:post_process.py

from qsadpp.orchestrator import OrchestratorConfig, PipelineOrchestrator
from qsadpp.io_reader import ReaderOptions
from qsadpp.feature_calculator import FeatureConfig

cfg = OrchestratorConfig(
    pdb_dir="./quantum_data/1m7y",
    reader_options=ReaderOptions(
        chunksize=100_000,
        strict=True,
        categorize_strings=True,
        include_all_csv=False,
    ),
    fifth_bit=False,
    out_dir="e_results",

    # ===== NEW: turn on features =====
    compute_features=True,
    feature_from="decoded",
    combined_feature_name="features.jsonl",
    feature_config=FeatureConfig(
        output_format="jsonl",

    ),
)

if __name__ == "__main__":
    runner = PipelineOrchestrator(cfg)
    summary = runner.run()
    print(summary)

