"""Depth-free code-as-policy manipulation benchmark — runtime + pure-primitive tool substrate.

Governing invariant (spec 2026-07-01-depthfree-codeaspolicy-benchmark-design.md §0.1): the harness may
expose evidence, execute bounded actions, and report consequences — it must NOT choose perception
targets, scale sources, recovery strategies, or task decomposition for the model.
"""
