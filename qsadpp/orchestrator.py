# --*-- conding:utf-8 --*--
# @time:10/23/25 17:20
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:orchestrator.py

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import pandas as pd

from qsadpp.io_reader import SampleReader, ReaderOptions
from qsadpp.coordinate_decoder import CoordinateBatchDecoder, CoordinateDecoderConfig
from qsadpp.energy_calculator import LatticeEnergyCalculator, EnergyConfig


_LOG = logging.getLogger(__name__)
if not _LOG.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass
class OrchestratorConfig:
    """
    End-to-end pipeline configuration for:
      read -> per-group decode -> per-group energy -> optional global aggregation.

    Notes:
    - For main-chain only, side_chain_hot_vector will be inferred as [False] * len(sequence)
      by default via `side_chain_hot_builder`. You can override it with a custom builder.
    - fifth_bit default is False; set to True if your problem encoding requires it.
    """
    # Reader
    pdb_dir: str
    reader_options: ReaderOptions = field(default_factory=ReaderOptions)

    # Side chain & encoding options (main-chain only default)
    fifth_bit: bool = False
    side_chain_hot_builder: Optional[Callable[[str], list[bool]]] = None  # input: sequence -> list of bool

    # Output directories and templates
    out_dir: str = "results"
    # Per-group file templates
    decoded_tpl: str = "{protein}_{label}_g{gid}.decoded.jsonl"
    energy_tpl: str = "{protein}_{label}_g{gid}.energy.jsonl"
    # Optional global append-to files
    decoded_all_path: Optional[str] = "decoded_all.jsonl"
    energy_all_path: Optional[str] = "energies_all.jsonl"

    # Energy defaults (reasonable for tetrahedral lattice)
    energy_config: EnergyConfig = field(default_factory=lambda: EnergyConfig(
        r_min=0.5,
        r_contact=1.0,
        d0=0.57735,
        lambda_overlap=1000.0,
        weights={"steric": 1.0, "geom": 0.5, "bond": 0.2, "mj": 1.0},
        normalize=True,
        output_path="__unused__.jsonl",  # will be overridden per group
    ))

    # Decode defaults (bitstring->turns->coords)
    decoder_output_format: str = "jsonl"  # "jsonl" | "parquet"
    # Safety toggles
    strict_decode: bool = False
    strict_energy: bool = False  # currently only affects behavior inside calc if you add strict later


class PipelineOrchestrator:
    """
    Orchestrates the end-to-end flow:
      - stream groups from SampleReader
      - decode each group's bitstrings to coordinates
      - evaluate energies on decoded coordinates
      - write per-group artifacts
      - optionally append into two global files (decoded_all.jsonl, energies_all.jsonl)

    Assumptions:
      - All CSVs in `pdb_dir` share the same sequence alphabet and consistent schema.
      - Only main-chain is present; side_chain_hot_vector defaults to all-False of length N.
    """

    def __init__(self, cfg: OrchestratorConfig):
        self.cfg = cfg
        _ensure_dir(self.cfg.out_dir)

        # Initialize reader
        self.reader = SampleReader(self.cfg.pdb_dir, self.cfg.reader_options)

        # Global append-to paths (optional)
        self.decoded_all_path = (
            os.path.join(self.cfg.out_dir, self.cfg.decoded_all_path)
            if self.cfg.decoded_all_path
            else None
        )
        self.energy_all_path = (
            os.path.join(self.cfg.out_dir, self.cfg.energy_all_path)
            if self.cfg.energy_all_path
            else None
        )
        # Prepare empty/clean files if needed
        if self.decoded_all_path:
            # ensure file exists and is empty (overwrite)
            open(self.decoded_all_path, "w", encoding="utf-8").close()
        if self.energy_all_path:
            open(self.energy_all_path, "w", encoding="utf-8").close()

    # ---------- helpers ----------

    def _infer_side_chain_hot(self, sequence: str) -> list[bool]:
        """Default builder: main-chain only => all False, length=len(sequence)."""
        if self.cfg.side_chain_hot_builder is not None:
            return list(self.cfg.side_chain_hot_builder(sequence))
        return [False] * len(sequence)

    def _build_decoder(self, sequence: str) -> CoordinateBatchDecoder:
        """Create a decoder with a side_chain_hot_vector matched to this group's sequence length."""
        hot_vec = self._infer_side_chain_hot(sequence)
        dec_cfg = CoordinateDecoderConfig(
            side_chain_hot_vector=hot_vec,
            fifth_bit=self.cfg.fifth_bit,
            output_format=self.cfg.decoder_output_format,
            output_path="__per_group_set_later__.jsonl",  # will be set per group
            bitstring_col="bitstring",
            sequence_col="sequence",
            strict=self.cfg.strict_decode,
        )
        return CoordinateBatchDecoder(dec_cfg)

    def _build_energy_calc(self) -> LatticeEnergyCalculator:
        """Create an energy calculator with configured MJ path and weights."""
        return LatticeEnergyCalculator(self.cfg.energy_config)

    def _group_paths(self, protein: str, label: str, gid: int) -> Tuple[str, str]:
        """Resolve per-group decoded and energy file paths."""
        decoded_name = self.cfg.decoded_tpl.format(protein=protein, label=label, gid=gid)
        energy_name = self.cfg.energy_tpl.format(protein=protein, label=label, gid=gid)
        return os.path.join(self.cfg.out_dir, decoded_name), os.path.join(self.cfg.out_dir, energy_name)

    def _append_file(self, from_path: str, to_path: Optional[str]) -> None:
        """Append a JSONL file into a global JSONL (line by line)."""
        if not to_path:
            return
        with open(from_path, "r", encoding="utf-8") as fin, open(to_path, "a", encoding="utf-8") as fout:
            for line in fin:
                fout.write(line)

    # ---------- main API ----------

    def run(self) -> Dict[str, int]:
        """
        Execute the full pipeline over all groups yielded by SampleReader.
        Returns a small summary with counts of groups processed and records written.
        """
        groups = 0
        decoded_rows_total = 0
        energy_rows_total = 0

        for (protein, label, gid), df, meta in self.reader.iter_groups():
            groups += 1
            if len(df) == 0:
                _LOG.warning("Group (%s, %s, %s) is empty; skipping.", protein, label, gid)
                continue

            # Ensure sequence column exists for traceability and MJ computation
            if "sequence" not in df.columns:
                _LOG.warning("Group (%s, %s, %s) has no 'sequence' column; skipping.", protein, label, gid)
                continue

            # Use the first row's sequence to set side_chain_hot_vector
            seq0 = str(df["sequence"].iloc[0])
            decoder = self._build_decoder(sequence=seq0)
            decoded_path, energy_path = self._group_paths(protein, label, gid)

            # ---- Decode per group ----
            # Only keep minimum columns required by decoder (bitstring, sequence)
            in_df = df[[decoder.cfg.bitstring_col, decoder.cfg.sequence_col]].copy()
            decoder.cfg.output_path = decoded_path  # set per-group output
            decoder.cfg.output_format = self.cfg.decoder_output_format

            _ensure_dir(os.path.dirname(decoded_path))
            _ensure_dir(os.path.dirname(energy_path))

            _LOG.info("Decoding group (%s, %s, %s) -> %s", protein, label, gid, decoded_path)
            dec_summary = decoder.decode_and_save(in_df)
            decoded_rows_total += int(dec_summary.get("written", 0))

            # Optionally append decoded into a global file
            if self.decoded_all_path and os.path.exists(decoded_path):
                self._append_file(decoded_path, self.decoded_all_path)

            # ---- Energy per group ----
            calc = self._build_energy_calc()
            calc.cfg.output_path = energy_path  # set per-group output
            _LOG.info("Computing energies for group (%s, %s, %s) -> %s", protein, label, gid, energy_path)
            en_summary = calc.evaluate_jsonl(decoded_path, energy_path)
            energy_rows_total += int(en_summary.get("written", 0))

            # Optionally append energy into a global file
            if self.energy_all_path and os.path.exists(energy_path):
                self._append_file(energy_path, self.energy_all_path)

        _LOG.info("Pipeline finished: %d groups, %d decoded rows, %d energy rows.",
                  groups, decoded_rows_total, energy_rows_total)
        return {
            "groups": groups,
            "decoded_rows": decoded_rows_total,
            "energy_rows": energy_rows_total,
            "decoded_all": self.decoded_all_path or "",
            "energies_all": self.energy_all_path or "",
        }
