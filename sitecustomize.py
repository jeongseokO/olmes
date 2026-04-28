"""Small import guard for local Capsule OLMES launches.

lm_eval 0.4.3 imports every registered model backend at package import time.
When an editable vLLM checkout is installed in the same environment, that means
plain HF evals can abort while importing vLLM native CUDA modules before OLMES
has a chance to select the HF model path.
"""

from __future__ import annotations

import importlib.abc
import os
import sys


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
        "y",
    }


class _BlockedVLLMImportFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None):
        if fullname == "vllm" or fullname.startswith("vllm."):
            raise ModuleNotFoundError(
                "vLLM imports are disabled for this HF OLMES run "
                "(CAPSULE_BLOCK_VLLM_IMPORT=1)"
            )
        return None


if _env_truthy("CAPSULE_BLOCK_VLLM_IMPORT") and not any(
    isinstance(finder, _BlockedVLLMImportFinder) for finder in sys.meta_path
):
    sys.meta_path.insert(0, _BlockedVLLMImportFinder())
