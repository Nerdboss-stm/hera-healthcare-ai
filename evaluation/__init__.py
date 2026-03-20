"""LLM-as-Judge Clinical Evaluation Framework.

Evaluates clinical AI outputs for factual consistency, hallucination
detection, medical accuracy, and clinical safety — going beyond
traditional ROUGE/BLEU metrics.
"""

from evaluation.evaluator import ClinicalEvaluator

__all__ = ["ClinicalEvaluator"]
