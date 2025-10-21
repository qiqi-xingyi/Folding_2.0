# --*-- conding:utf-8 --*--
# @time:10/21/25 14:13
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:reverse_decoder.py

"""
Unified reverse-decoding helpers for QSAD post-processing.

This module wires together:
- The original, unmodified ProteinShapeDecoder (imported from qsadpp.decoder)
- Optional hooks to obtain Cα coordinates via your existing ProteinFoldingProblem/Result
  (if you want structures, not just turn vectors)

Design goals:
- Keep algorithmic logic simple and dependency-light
- Let callers choose either:
    A) "Vectors-only" decoding (pure bitstring -> main/side turn vectors)
    B) "Full" decoding (bitstring -> vectors -> ProteinFoldingResult -> Cα coords)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .decoder import ProteinShapeDecoder  # unmodified original logic


def safe_get_calpha_coords(pf_result) -> np.ndarray:
    """
    Try several common access patterns to fetch Cα coordinates from a ProteinFoldingResult.
    Adjust this if your class exposes different APIs.

    Order of attempts:
      1) pf_result.get_calpha_coords()
      2) pf_result.get_ca_coords()
      3) pf_result.get_coordinates()  # expects dict/tuple; try extracting Cα
      4) pf_result.coordinates        # attribute form; try extracting Cα
    Returns:
      (L, 3) float32 numpy array

    Raises:
      AttributeError if none of the patterns work.
    """
    # 1) Direct getters
    for name in ("get_calpha_coords", "get_ca_coords"):
        if hasattr(pf_result, name):
            arr = getattr(pf_result, name)()
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] == 3:
                return arr
            raise ValueError(f"{name}() did not return (L,3) array, got {arr.shape}")

    # 2) Generic getters/attributes that might return multiple atom types
    def _extract_ca(obj) -> Optional[np.ndarray]:
        keys = ("CA", "Calpha", "Cα", "C-A", "Ca", "ca")
        if isinstance(obj, dict):
            for k in keys:
                if k in obj:
                    arr = np.asarray(obj[k], dtype=np.float32)
                    if arr.ndim == 2 and arr.shape[1] == 3:
                        return arr
        if isinstance(obj, (list, tuple)) and obj:
            arr = np.asarray(obj[0], dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] == 3:
                return arr
        return None

    if hasattr(pf_result, "get_coordinates"):
        obj = pf_result.get_coordinates()
        ca = _extract_ca(obj)
        if ca is not None:
            return ca

    if hasattr(pf_result, "coordinates"):
        obj = getattr(pf_result, "coordinates")
        ca = _extract_ca(obj)
        if ca is not None:
            return ca

    raise AttributeError(
        "Unable to obtain Cα coordinates from ProteinFoldingResult. "
        "Please provide a custom calpha_getter."
    )


@dataclass
class ReverseDecoder:
    """
    Reverse-decoding façade.

    Parameters
    ----------
    calpha_getter : Optional[Callable[[object], np.ndarray]]
        A function that maps a ProteinFoldingResult -> (L,3) Cα numpy array.
        If None, decode_coords_via_problem will call `safe_get_calpha_coords`.
    """
    calpha_getter: Optional[Callable[[object], np.ndarray]] = None

    def decode_vectors(
        self,
        bitstring: str,
        side_chain_hot_vector: Sequence[bool],
        fifth_bit: bool,
    ) -> Dict[str, object]:
        """
        Decode main and side-chain turn vectors from a compact bitstring.

        Returns
        -------
        dict with:
          - main_vectors: List[int]   (length N-1)
          - side_vectors: List[Optional[int]] (length N; None means no side chain)
        """
        dec = ProteinShapeDecoder(
            vector_sequence=str(bitstring),
            side_chain_hot_vector=list(side_chain_hot_vector),
            fifth_bit=bool(fifth_bit),
        )
        return {
            "main_vectors": dec.main_vectors,
            "side_vectors": dec.side_vectors,
        }

    def decode_coords_via_problem(
        self,
        problem,  # your ProteinFoldingProblem
        bitstring: str,
        side_chain_hot_vector: Sequence[bool],
        fifth_bit: bool,
        calpha_getter: Optional[Callable[[object], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        """
        Full pipeline for one bitstring:
          bitstring --(ProteinShapeDecoder)--> turn vectors
          bitstring --(ProteinFoldingProblem.interpret)--> ProteinFoldingResult --(getter)--> Cα coords

        Returns
        -------
        (ca_coords, vectors_dict)
          - ca_coords: (L,3) float32 numpy array
          - vectors_dict: same dict as decode_vectors()
        """
        vectors = self.decode_vectors(bitstring, side_chain_hot_vector, fifth_bit)

        if not hasattr(problem, "interpret"):
            raise AttributeError("Problem object must provide an .interpret(binary_probs) method.")
        pf_result = problem.interpret({bitstring: 1.0})

        getter = calpha_getter or self.calpha_getter or safe_get_calpha_coords
        ca = getter(pf_result)
        return ca, vectors


def batch_decode_vectors(
    items: Iterable[Tuple[str, float]],  # (bitstring, prob)
    side_chain_hot_vector: Sequence[bool],
    fifth_bit: bool,
) -> List[Dict[str, object]]:
    """
    Decode vectors for a list of bitstrings in one group.

    Returns list of dicts, each dict containing:
      - bitstring
      - q_prob
      - main_vectors
      - side_vectors
    """
    dec = ReverseDecoder()
    out: List[Dict[str, object]] = []
    for bitstring, q in items:
        vecs = dec.decode_vectors(bitstring, side_chain_hot_vector, fifth_bit)
        row = {"bitstring": bitstring, "q_prob": float(q), **vecs}
        out.append(row)
    return out


def batch_decode_coords_via_problem(
    problem,
    items: Iterable[Tuple[str, float]],  # (bitstring, prob)
    side_chain_hot_vector: Sequence[bool],
    fifth_bit: bool,
    calpha_getter: Optional[Callable[[object], np.ndarray]] = None,
) -> List[Dict[str, object]]:
    """
    Decode coordinates (via ProteinFoldingProblem) for a list of bitstrings.

    Returns list of dicts, each dict containing:
      - bitstring
      - q_prob
      - ca_coords  (np.ndarray, shape (L,3))
      - main_vectors
      - side_vectors
    """
    dec = ReverseDecoder(calpha_getter=calpha_getter)
    out: List[Dict[str, object]] = []
    for bitstring, q in items:
        ca, vecs = dec.decode_coords_via_problem(
            problem=problem,
            bitstring=bitstring,
            side_chain_hot_vector=side_chain_hot_vector,
            fifth_bit=fifth_bit,
            calpha_getter=calpha_getter,
        )
        row = {"bitstring": bitstring, "q_prob": float(q), "ca_coords": ca, **vecs}
        out.append(row)
    return out




