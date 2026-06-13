"""Segment 4 — evaluation harness."""

from grounded.eval.factscore import HttpClaimVerifier, MockClaimVerifier, compute_factscore
from grounded.eval.verifier_client import load_claim_verifier
from grounded.eval.prompts_build import build_eval_prompts

__all__ = [
    "HttpClaimVerifier",
    "MockClaimVerifier",
    "build_eval_prompts",
    "compute_factscore",
    "load_claim_verifier",
]
