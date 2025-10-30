# --*-- conding:utf-8 --*--
# @time:10/21/25 14:23
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:post_process.py

from qsadpp.orchestrator import OrchestratorConfig, PipelineOrchestrator
from qsadpp.io_reader import ReaderOptions

cfg = OrchestratorConfig(
    pdb_dir="./quantum_data/1m7y",
    reader_options=ReaderOptions(
        chunksize=100000,
        strict=True,
        categorize_strings=True,
        include_all_csv=False,
    ),
    fifth_bit=False,
    out_dir="e_results",
    decoder_output_format="jsonl",
    aggregate_only=True,
)


if __name__ == "__main__":
    runner = PipelineOrchestrator(cfg)
    summary = runner.run()
    print(summary)

