import copy
import importlib.util
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, List, Optional, Tuple, cast

import importlib_metadata
import torch
import transformers
from lm_eval.models.utils import Collator
from lm_eval.models.vllm_causallms import VLLM
from packaging import version
from tqdm import tqdm

from oe_eval.components.instances import RequestInstance
from oe_eval.components.requests import (
    GenerateUntilAndLoglikelihoodRequest,
    GenerateUntilRequest,
    LoglikelihoodRequest,
)
from oe_eval.utilities.model_results_collation import collate_results
from oe_eval.utils import cut_at_stop_sequence

logger = logging.getLogger(__name__)
STRUCTURED_PROMPT_KWARGS = (
    "prompt_messages",
    "prompt_add_generation_prompt",
    "structured_prompt_segments",
)


def _supports_structured_prompt_generate(model, generation_kwargs=None) -> bool:
    if generation_kwargs:
        if bool(generation_kwargs.get("unified_llopa_generate")):
            return True
        if bool(generation_kwargs.get("direct_llopa_generate")):
            return True
        if bool(generation_kwargs.get("llopa_v2_vllm_generate")):
            return True
    if model is None:
        return False
    if bool(getattr(model, "_structured_llopa_generate_attached", False)):
        return True
    if bool(getattr(model, "_unified_llopa_generate_default", False)):
        return True
    if bool(getattr(model, "_llopa_v2_vllm_generate_default", False)):
        return True
    if bool(getattr(model, "_direct_llopa_generate_default", False)):
        return True
    return False


def _strip_structured_prompt_kwargs(model, generation_kwargs):
    cleaned = {} if generation_kwargs is None else generation_kwargs.copy()
    if _supports_structured_prompt_generate(model, cleaned):
        return cleaned, []
    removed = [key for key in STRUCTURED_PROMPT_KWARGS if key in cleaned]
    for key in removed:
        cleaned.pop(key, None)
    return cleaned, removed


LLOPA_V2_VLLM_KEYS = (
    "llopa_v2_vllm_generate",
    "llopa_v2_batch_generate",
    "llopa_v2_generate",
    "llopa_v2_layers",
    "llopa_v2_attn",
    "llopa_v2_system_prefill",
    "llopa_v2_user_prefill",
    "llopa_v2_replay_module",
    "llopa_v2_last_layer_module",
    "llopa_v2_replay_per_layers",
    "llopa_v2_no_upper_attn",
    "llopa_v2_see_past_assistant",
    "llopa_v2_seed_mode",
    "force_custom_modeling",
    "lopa_modeling_path",
    "modeling_family",
    "attn_impl",
    "attn_implementation",
    "_attn_implementation",
)


def _repo_capsule_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "Capsule"


def _load_capsule_module(module_name: str, filename: str):
    capsule_dir = _repo_capsule_dir()
    capsule_dir_s = str(capsule_dir)
    if capsule_dir_s not in sys.path:
        sys.path.insert(0, capsule_dir_s)
    cur_py_path = os.environ.get("PYTHONPATH", "")
    py_parts = [p for p in cur_py_path.split(":") if p]
    if capsule_dir_s not in py_parts:
        os.environ["PYTHONPATH"] = f"{capsule_dir_s}:{cur_py_path}" if cur_py_path else capsule_dir_s

    try:
        return __import__(module_name, fromlist=["*"])
    except Exception:
        module_path = capsule_dir / filename
        spec = importlib.util.spec_from_file_location(module_name, str(module_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load {module_name} from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _configure_llopa_v2_vllm(kwargs: dict) -> dict[str, Any] | None:
    if not _normalize_bool(kwargs.get("llopa_v2_vllm_generate", False)):
        return None

    llopa_config = {
        "layers": int(kwargs.get("llopa_v2_layers") or 0),
        "attn": str(kwargs.get("llopa_v2_attn") or "causal").strip().lower(),
        "system_prefill": str(kwargs.get("llopa_v2_system_prefill") or "full").strip().lower(),
        "user_prefill": str(kwargs.get("llopa_v2_user_prefill") or "full").strip().lower(),
        "see_past_assistant": _normalize_bool(kwargs.get("llopa_v2_see_past_assistant", False)),
        "modeling_family": str(kwargs.get("modeling_family") or "llama").strip().lower(),
    }
    if llopa_config["attn"] == "prefix_full":
        llopa_config["attn"] = "full"
    if llopa_config["attn"] != "causal":
        raise ValueError("llopa_v2_vllm_generate currently supports only llopa_v2_attn='causal'.")
    if llopa_config["user_prefill"] != "full":
        raise ValueError("llopa_v2_vllm_generate currently supports only llopa_v2_user_prefill='full'.")
    if _normalize_bool(kwargs.get("llopa_v2_no_upper_attn", False)):
        raise ValueError("llopa_v2_vllm_generate does not support llopa_v2_no_upper_attn=True.")
    replay_module = str(kwargs.get("llopa_v2_replay_module") or "none").strip().lower()
    if replay_module and replay_module != "none":
        raise ValueError("llopa_v2_vllm_generate does not support replay modules.")
    if llopa_config["layers"] <= 0:
        raise ValueError("llopa_v2_vllm_generate requires llopa_v2_layers > 0.")

    plugin = _load_capsule_module("vllm_llopa_plugin", "vllm_llopa_plugin.py")
    plugin.register()
    if hasattr(plugin, "patch_runtime"):
        plugin.patch_runtime()
    arch = plugin.select_architecture(llopa_config["modeling_family"], "auto")
    if not arch:
        arch = "LLOPALlamaForCausalLM"
    kwargs.setdefault("hf_overrides", {"architectures": [arch]})
    return llopa_config


def _load_structured_prompt_builder():
    mod = _load_capsule_module(
        "llopa_utils.structured_prompt",
        "llopa_utils/structured_prompt.py",
    )
    return mod.build_structured_prompt_segments


def _tensor_len(value: Any) -> int:
    if isinstance(value, torch.Tensor) and value.dim() >= 2:
        return int(value.size(1))
    return 0


def _build_llopa_ctx_from_prompt_metadata(
    tokenizer,
    *,
    prompt_messages,
    prompt_add_generation_prompt,
    structured_prompt_segments,
    llopa_config: dict[str, Any],
) -> dict[str, Any] | None:
    segments = structured_prompt_segments if isinstance(structured_prompt_segments, dict) else None
    if segments is None:
        if prompt_messages is None:
            return None
        builder = _load_structured_prompt_builder()
        segments = builder(
            tokenizer,
            prompt_messages,
            prompt_add_generation_prompt=bool(prompt_add_generation_prompt),
            device="cpu",
        )

    system_len = _tensor_len(segments.get("system_ids"))
    prefix_len = _tensor_len(segments.get("prefix_ids"))
    prompt_len = _tensor_len(segments.get("prompt_ids"))
    if prompt_len <= 0:
        prompt_len = prefix_len + _tensor_len(segments.get("assistant_prefill_ids"))
    if prefix_len <= 0:
        prefix_len = max(0, prompt_len - _tensor_len(segments.get("assistant_prefill_ids")))

    system_prefill = str(llopa_config.get("system_prefill") or "full").strip().lower()
    if system_prefill == "full":
        keep_prefix_len = system_len
    elif system_prefill == "no_system":
        keep_prefix_len = min(1, system_len)
    else:
        keep_prefix_len = 0

    keep_ranges: list[list[int]] = []
    if keep_prefix_len > 0:
        keep_ranges.append([0, int(keep_prefix_len)])

    if bool(llopa_config.get("see_past_assistant")):
        starts = segments.get("assistant_header_starts")
        ends = segments.get("assistant_turn_ends")
        mask = segments.get("assistant_header_start_mask")
        if isinstance(starts, torch.Tensor) and isinstance(ends, torch.Tensor):
            starts = starts.reshape(1, -1) if starts.dim() == 1 else starts
            ends = ends.reshape(1, -1) if ends.dim() == 1 else ends
            if not isinstance(mask, torch.Tensor) or mask.shape != starts.shape:
                mask = starts >= 0
            else:
                mask = mask.to(dtype=torch.bool)
            turns = min(int(starts.size(1)), int(ends.size(1)))
            for col in range(turns):
                if not bool(mask[0, col].item()):
                    continue
                start = int(starts[0, col].item())
                end = int(ends[0, col].item())
                if start >= prefix_len:
                    continue
                start = max(start, keep_prefix_len)
                end = min(end, prefix_len)
                if end > start:
                    keep_ranges.append([start, end])

    keep_ranges.sort()
    merged: list[list[int]] = []
    for start, end in keep_ranges:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    return {
        "lower_k": int(llopa_config["layers"]),
        "keep_prefix_len": int(keep_prefix_len),
        "drop_len": max(0, int(prefix_len) - int(keep_prefix_len)),
        "tail_start": int(prefix_len),
        "prompt_len": int(prompt_len),
        "keep_ranges": merged,
    }

# Verbose versions of key methods in VLLM from lm_eval.models.vllm_causallms


class VLLM_Verbose(VLLM):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM  # for encode_pairs to work

    def __init__(
        self,
        pretrained="gpt2",
        device: Optional[str] = None,
        # dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto",
        # revision: Optional[str] = None,
        # trust_remote_code: Optional[bool] = False,
        # tokenizer: Optional[str] = None,
        # tokenizer_mode: Literal["auto", "slow"] = "auto",
        # tokenizer_revision: Optional[str] = None,
        # add_bos_token: Optional[bool] = False,
        # tensor_parallel_size: int = 1,
        # quantization: Optional[str] = None,
        # max_gen_toks: int = 256,
        # swap_space: int = 4,
        # batch_size: Union[str, int] = 1,
        # max_batch_size=None,
        # max_length: int = None,
        # max_model_len: int = None,
        # seed: int = 1234,
        # gpu_memory_utilization: float = 0.9,
        # data_parallel_size: int = 1,
        **kwargs,
    ):
        try:
            self.vllm_version = importlib_metadata.version("vllm")
        except importlib_metadata.PackageNotFoundError:
            raise ModuleNotFoundError(
                "model-type is `vllm` but `vllm` is not installed! Please install vllm via `pip install -e .[gpu]`"
            )

        self._llopa_v2_vllm_generate_default = _normalize_bool(
            kwargs.get("llopa_v2_vllm_generate", False)
        )
        self._llopa_v2_vllm_config = _configure_llopa_v2_vllm(kwargs)

        self.vllm_logit_bias = kwargs.pop("vllm_logit_bias", None)
        if version.parse(self.vllm_version) < version.parse("0.6.3") and self.vllm_logit_bias:
            raise RuntimeError(
                "vllm version 0.6.3 is required for using logit_bias in vllm sampling. "
                "Try 'pip install -e .[gpu]' again to update vllm."
            )

        if device is None:
            device = "cuda" if torch.cuda.device_count() > 0 else "cpu"
        if torch.cuda.device_count() > 1:
            kwargs["tensor_parallel_size"] = torch.cuda.device_count()

        if torch.cuda.device_count() == 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # if "revision" in kwargs and kwargs["revision"] is None:
        # Hack to deal with Eleuther using "main" as a default for "revision", not needed for VLLM
        #    kwargs = kwargs.copy()
        #    del kwargs["revision"]

        self.vllm_for_mc = kwargs.pop("vllm_for_mc", False)
        if self._llopa_v2_vllm_generate_default:
            kwargs["enable_prefix_caching"] = _normalize_bool(
                kwargs.get("enable_prefix_caching", False)
            )
        for key in LLOPA_V2_VLLM_KEYS:
            kwargs.pop(key, None)

        super().__init__(pretrained, device=device, **kwargs)

    def unload_model(self):
        # Free model from GPU memory, following advice in https://github.com/vllm-project/vllm/issues/1908
        import contextlib
        import gc

        import ray
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        # Delete the llm object and free the memory
        destroy_model_parallel()
        destroy_distributed_environment()
        with contextlib.suppress(AttributeError):
            del self.model.llm_engine.model_executor  # type: ignore
        del self.model  # type: ignore
        self.model = None
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()
        try:
            ray.shutdown()
        except Exception:
            pass

    def _model_generate_with_llopa_ctx(
        self,
        requests: List[List[int]],
        llopa_contexts: List[dict[str, Any] | None],
        *,
        max_tokens: int,
        stop: Optional[List[str]],
        kwargs: dict[str, Any],
    ):
        if self.data_parallel_size > 1:
            raise ValueError("llopa_v2_vllm_generate does not support data_parallel_size > 1.")

        from vllm import SamplingParams

        base_kwargs = self.modify_gen_kwargs(copy.deepcopy(kwargs))
        prompts = [{"prompt_token_ids": list(token_ids)} for token_ids in requests]
        sampling_params = []
        for ctx in llopa_contexts:
            per_kwargs = copy.deepcopy(base_kwargs)
            extra_args = per_kwargs.pop("extra_args", None)
            if not isinstance(extra_args, dict):
                extra_args = {}
            if ctx is not None:
                extra_args["llopa_ctx"] = ctx
            if extra_args:
                per_kwargs["extra_args"] = extra_args
            sampling_params.append(
                SamplingParams(max_tokens=max_tokens, stop=stop, **per_kwargs)
            )

        generate_kwargs = {
            "sampling_params": sampling_params,
            "use_tqdm": True if self.batch_size == "auto" else False,
        }
        if self.lora_request is not None:
            generate_kwargs["lora_request"] = self.lora_request
        return self.model.generate(prompts, **generate_kwargs)

    def generate_until_verbose(
        self, requests: List[GenerateUntilRequest], disable_tqdm: bool = False
    ) -> List[dict]:
        # oe-eval: Convert compute context and all_gen_kwargs in a custom way, then minimal changes needed below
        list_context = []
        list_gen_kwargs = []
        list_prompt_messages = []
        list_prompt_add_generation_prompt = []
        list_structured_prompt_segments = []
        for request in requests:
            kwargs = request.generation_kwargs
            kwargs, removed_prompt_kwargs = _strip_structured_prompt_kwargs(self, kwargs)
            if removed_prompt_kwargs and not getattr(
                self, "_warned_ignored_structured_prompt_kwargs", False
            ):
                logger.warning(
                    "Ignoring structured prompt generation kwargs for a model without structured LLoPA generate support; using the rendered chat context instead."
                )
                self._warned_ignored_structured_prompt_kwargs = True
            kwargs["until"] = request.stop_sequences
            prompt_messages = kwargs.pop("prompt_messages", getattr(request, "prompt_messages", None))
            prompt_add_generation_prompt = kwargs.pop(
                "prompt_add_generation_prompt",
                getattr(request, "prompt_add_generation_prompt", None),
            )
            structured_prompt_segments = kwargs.pop("structured_prompt_segments", None)
            list_context.append(cast(str, request.context))
            list_gen_kwargs.append(kwargs)
            list_prompt_messages.append(prompt_messages)
            list_prompt_add_generation_prompt.append(prompt_add_generation_prompt)
            list_structured_prompt_segments.append(structured_prompt_segments)
        res = []

        # batch tokenize contexts
        # context, all_gen_kwargs = zip(*(req.arg for req in requests))
        list_context_encoding = self.tokenizer(
            list_context, add_special_tokens=self.add_bos_token
        ).input_ids
        requests_tup = [
            ((a, b, pm, pagp, seg), c)
            for a, b, pm, pagp, seg, c in zip(
                list_context,
                list_context_encoding,
                list_prompt_messages,
                list_prompt_add_generation_prompt,
                list_structured_prompt_segments,
                list_gen_kwargs,
            )
        ]

        def _collate_gen(_requests):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            return -len(_requests[0][1]), _requests[0][0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = Collator(requests_tup, _collate_gen, group_by="gen_kwargs")
        chunks = re_ords.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0, batch_fn=None
        )

        pbar = tqdm(
            total=len(requests_tup),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        # for each different set of kwargs, we execute all requests, by batch.
        context_length_warning = False
        for chunk in chunks:
            context_and_encoding, all_gen_kwargs = zip(*chunk)
            (
                context,
                context_encoding,
                all_prompt_messages,
                all_prompt_add_generation_prompt,
                all_structured_prompt_segments,
            ) = zip(*context_and_encoding)
            context_lengths = [len(x) for x in context_encoding]
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            until = None
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                if "until" in kwargs.keys():
                    until = kwargs.pop("until")
                    if isinstance(until, str):
                        until = [until]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                        )
            else:
                raise ValueError(f"Expected `kwargs` to be of type `dict` but got {gen_kwargs}")
            # add EOS token to stop sequences
            eos = self.tokenizer.decode(self.eot_token_id)
            if not until:
                until = [eos]
            else:
                until.append(eos)
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            truncate_context = kwargs.pop("truncate_context", True)
            min_acceptable_gen_toks = kwargs.pop("min_acceptable_gen_toks", 0)

            if truncate_context:
                # set the max length in tokens of inputs ("context_enc")
                # max len for inputs = max length, minus room to generate the max new tokens
                max_ctx_len = self.max_length - max_gen_toks
                if max_ctx_len <= 0:
                    raise ValueError(
                        f"max_gen_toks ({max_gen_toks}) is greater than max_length ({self.max_length}), "
                        + '"consider using "truncate_context": False'
                    )
            else:
                # If truncate_context is False we modify max_gen_toks instead, to keep full context
                max_ctx_len = self.max_length
                # The first context is the longest because of ordering above
                max_batch_toks = len(context_encoding[0])
                new_max_gen_toks = min(max_gen_toks, self.max_length - max_batch_toks)
                if new_max_gen_toks < max_gen_toks and not context_length_warning:
                    logger.warning(
                        f"Running with effective max_gen_toks of {new_max_gen_toks} instead of {max_gen_toks}!"
                    )
                    context_length_warning = True
                max_gen_toks = max(new_max_gen_toks, 0)
                if max_gen_toks == 0:
                    if min_acceptable_gen_toks > 0:
                        # We allow desperation context truncation here, after all
                        max_ctx_len -= min_acceptable_gen_toks
                        max_gen_toks = min_acceptable_gen_toks
                    else:
                        raise ValueError(
                            f"Effective max_gen_toks is 0 (max_batch_toks = {max_batch_toks}, max_ctx_len = {max_ctx_len}) "
                            + '"consider using "truncate_context": True'
                        )

            needs_llopa_v2_vllm = bool(self._llopa_v2_vllm_generate_default)
            if needs_llopa_v2_vllm and any(len(x) > max_ctx_len for x in context_encoding):
                raise ValueError(
                    "llopa_v2_vllm_generate cannot use left-truncated contexts because "
                    "LLoPA visibility metadata is position-based. Increase OE_EVAL_MAX_LENGTH "
                    "or lower max_gen_toks."
                )
            context_encoding_trunc = [x[-max_ctx_len:] for x in context_encoding]

            kwargs["logprobs"] = 1  # oe-eval: always return logprobs
            if self.vllm_logit_bias:
                logger.warning(
                    f"""
                    Sampling generations with logit_bias of {self.vllm_logit_bias}.
                    Logprobs in the output will be modified by the underlying logit_bias!
                    """
                )
                kwargs["logit_bias"] = self.vllm_logit_bias
            llopa_contexts = None
            if needs_llopa_v2_vllm:
                llopa_contexts = [
                    _build_llopa_ctx_from_prompt_metadata(
                        self.tokenizer,
                        prompt_messages=prompt_messages,
                        prompt_add_generation_prompt=prompt_add_generation_prompt,
                        structured_prompt_segments=structured_prompt_segments,
                        llopa_config=self._llopa_v2_vllm_config or {},
                    )
                    for prompt_messages, prompt_add_generation_prompt, structured_prompt_segments in zip(
                        all_prompt_messages,
                        all_prompt_add_generation_prompt,
                        all_structured_prompt_segments,
                    )
                ]

            # perform batched generation
            if llopa_contexts is not None and any(ctx is not None for ctx in llopa_contexts):
                cont = self._model_generate_with_llopa_ctx(
                    context_encoding_trunc,
                    llopa_contexts,
                    max_tokens=max_gen_toks,
                    stop=until,
                    kwargs=kwargs,
                )
            else:
                cont = self._model_generate(
                    requests=context_encoding_trunc,
                    generate=True,
                    max_tokens=max_gen_toks,
                    stop=until,
                    **kwargs,
                )

            # cache generations
            for output, context, context_length in zip(cont, context, context_lengths):
                completion_output = output.outputs[0]
                generated_text_raw = completion_output.text
                generated_text = cut_at_stop_sequence(
                    generated_text_raw, until
                )  # oe_eval: catch stop sequence
                num_tokens = len(completion_output.token_ids)
                cumulative_logprob = completion_output.cumulative_logprob
                res1 = {
                    "continuation": generated_text,
                    "num_tokens": num_tokens,
                    "context_tokens": context_length,
                    # "tokens": completion_output.token_ids,
                }
                if cumulative_logprob is not None:
                    res1["sum_logits"] = cumulative_logprob
                    # log_probs = [list(p.values())[0].logprob for p in completion_output.logprobs]
                    # res1["logits"] = log_probs
                if generated_text_raw != generated_text:
                    res1["continuation_raw"] = generated_text_raw
                res.append(res1)  # oe_eval: change to dict here
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), generated_text)
                pbar.update(1)

        pbar.close()
        # reorder all group of results back to original unsorted form
        return re_ords.get_original(res)

    def loglikelihood_verbose(
        self,
        requests: List[LoglikelihoodRequest],
        override_bs: Optional[int] = None,
        disable_tqdm: bool = False,
    ) -> List[dict]:
        # This method is just copied from eleuther_huggingface for now
        new_reqs = []
        st_reqs = []
        use_single_token = self.vllm_for_mc
        for req in requests:
            context = cast(str, req.context)
            continuation = req.continuation
            if context == "":
                # BOS or EOS as context
                context_enc, continuation_enc = (
                    [self.prefix_token_id],
                    self.tok_encode(continuation),
                )
                use_single_token = False  # Cannot use single-token mode with empty context
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)
                if use_single_token:
                    # we use simple encoding for token continuation length check to avoiding picking up special tokens
                    lean_continuation_enc = self.tokenizer(
                        continuation, add_special_tokens=False
                    ).input_ids
                    if len(lean_continuation_enc) != 1:
                        use_single_token = False
                    else:
                        st_reqs.append((context, lean_continuation_enc))

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        if use_single_token:
            assert len(st_reqs) == len(
                requests
            ), "We send ALL requests to single_token_generate() or none!"

            logger.info(
                """
            *** requests routing ***
            The loglikelihood requests all have single-token continuations, and `vllm-for-mc` is enabled.
            Processing requests with the vLLM single_token_generate method.
            Check 'no_answer' metric in the output: if > 2% then the score may be suppressed by using `vllm-for-mc`.
            """
            )
            # Grouping requests by the same context (doc)
            req_groups: DefaultDict[str, List[dict]] = defaultdict(list)
            for res_id, streq in enumerate(st_reqs):
                req_groups[streq[0]].append({"res_id": res_id, "cont_enc": streq[1][0]})

            grouped_st_reqs = list(req_groups.items())

            return self.single_token_generate(grouped_st_reqs, disable_tqdm=disable_tqdm)

        return self._loglikelihood_tokens_verbose(new_reqs, disable_tqdm=disable_tqdm)

    def _loglikelihood_tokens_verbose(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        verbose_token_logits: bool = False,
    ) -> List[dict]:
        res = []

        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        # Reorder requests by length and batch
        re_ord = Collator(requests, sort_fn=_collate)
        chunks = re_ord.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0, batch_fn=None
        )

        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inputs = []
            ctxlens = []
            for cache_key, context_enc, continuation_enc in chunk:
                inp = (context_enc + continuation_enc)[-(self.max_length) :]
                ctxlen = len(context_enc) - max(
                    0, len(context_enc) + len(continuation_enc) - (self.max_length)
                )

                inputs.append(inp)
                ctxlens.append(ctxlen)

            outputs = self._model_generate(requests=inputs, generate=False)

            for output, ctxlen, (cache_key, _, _), inp in zip(outputs, ctxlens, chunk, inputs):
                answer = self._parse_logprobs_verbose(
                    tokens=inp,
                    outputs=output,
                    ctxlen=ctxlen,
                    verbose_token_logits=verbose_token_logits,
                )

                res.append(answer)

                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                pbar.update(1)
        pbar.close()
        return re_ord.get_original(res)

    def _parse_logprobs_verbose(
        self, tokens: List, outputs, ctxlen: int, verbose_token_logits=False
    ) -> dict:
        """Process logprobs and tokens.

        :param tokens: list
            Input tokens (potentially left-truncated)
        :param outputs: RequestOutput
            Contains prompt_logprobs
        :param ctxlen: int
            Length of context (so we can slice them away and only keep the predictions)
        :return:
            verbose answer dict
        """

        # The first entry of prompt_logprobs is None because the model has no previous tokens to condition on.
        continuation_logprobs_dicts = outputs.prompt_logprobs

        def coerce_logprob_to_num(logprob):
            # vLLM changed the return type of logprobs from float
            # to a Logprob object storing the float value + extra data
            # (https://github.com/vllm-project/vllm/pull/3065).
            # If we are dealing with vllm's Logprob object, return
            # the logprob value stored as an attribute. Otherwise,
            # return the object itself (which should be a float
            # for older versions of vLLM).
            return getattr(logprob, "logprob", logprob)

        continuation_logprobs_dicts = [
            (
                {token: coerce_logprob_to_num(logprob) for token, logprob in logprob_dict.items()}
                if logprob_dict is not None
                else None
            )
            for logprob_dict in continuation_logprobs_dicts
        ]

        # Calculate continuation_logprobs
        # assume ctxlen always >= 1
        logits = [
            logprob_dict.get(token)
            for token, logprob_dict in zip(tokens[ctxlen:], continuation_logprobs_dicts[ctxlen:])
        ]
        continuation_logprobs = sum(logits)

        # Determine if is_greedy
        is_greedy = True
        for token, logprob_dict in zip(tokens[ctxlen:], continuation_logprobs_dicts[ctxlen:]):
            # Get the token with the maximum log probability from the logprob_dict
            if logprob_dict:  # Ensure the logprob_dict is not None
                top_token = max(logprob_dict, key=logprob_dict.get)
                if top_token != token:
                    is_greedy = False
                    break

        verbose_answer: dict = {
            "sum_logits": continuation_logprobs,
            "num_tokens": len(logits),
            "num_tokens_all": len(tokens),
            "is_greedy": is_greedy,
        }
        if verbose_token_logits:
            instance_tokens = [
                self.tok_decode(x, skip_special_tokens=False) for x in tokens[ctxlen:]
            ]
            verbose_answer["tokens"] = instance_tokens
            verbose_answer["logits"] = logits

        return verbose_answer

    def single_token_generate(
        self,
        requests: List[Tuple[str, list]],
        disable_tqdm: bool = False,
    ) -> List[dict]:
        # batch requests by distinct continuation assortments
        grouped_reqs: DefaultDict[tuple, List[tuple]] = defaultdict(list)
        for context, multi_conts_enc in requests:
            context_enc: List[List[int]] = self.tok_encode(
                context, add_special_tokens=self.add_bos_token
            )
            multi_conts_set = {continuation["cont_enc"] for continuation in multi_conts_enc}
            grouped_reqs[tuple(sorted(multi_conts_set))].append((context_enc, multi_conts_enc))

        res = []

        if self.batch_size != "auto":
            logger.warning(
                "vLLM single_token_generate() always uses 'auto' batch_size, set model_type to 'hf' to enforce a fixed batch_size"
            )

        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running requests through single_token_generate",
        )

        for multi_conts_key, chunk in grouped_reqs.items():
            chunk_context_enc, chunk_multi_conts_enc = zip(*chunk)

            context_enc_trunc = [x[-self.max_length + 1 :] for x in chunk_context_enc]

            context_enc_lengths = [len(context) for context in context_enc_trunc]

            # perform batched generation
            if not self.vllm_logit_bias:
                outputs = self._model_generate(
                    requests=context_enc_trunc,
                    generate=True,
                    max_tokens=1,
                    stop=None,
                    logprobs=20,
                    temperature=0.0,
                )
            else:
                logger.warning(
                    f"""
                    Sampling generations with logit_bias of {self.vllm_logit_bias}.
                    Logprobs in the output will be modified by the underlying logit_bias!
                    """
                )
                outputs = self._model_generate(
                    requests=context_enc_trunc,
                    generate=True,
                    max_tokens=1,
                    stop=None,
                    logprobs=min(len(multi_conts_key) + 5, 20),
                    temperature=0.0,
                    logit_bias=self.vllm_logit_bias,
                )

            for context_len, out, multi_conts_enc in zip(
                context_enc_lengths, outputs, chunk_multi_conts_enc
            ):
                out_logprobs = out.outputs[0].logprobs[0]

                for continuation in multi_conts_enc:
                    cont_logprob = out_logprobs.get(continuation["cont_enc"])
                    if cont_logprob is not None:
                        res.append(
                            {
                                "sum_logits": cont_logprob.logprob,
                                "num_tokens": 1,
                                "num_tokens_all": context_len + 1,
                                "is_greedy": cont_logprob.rank == 1,
                                "rank": cont_logprob.rank,
                                "res_id": continuation["res_id"],
                            }
                        )
                    else:
                        res.append(
                            {
                                "sum_logits": -999,
                                "num_tokens": 1,
                                "num_tokens_all": context_len + 1,
                                "is_greedy": False,
                                "rank": None,
                                "res_id": continuation["res_id"],
                            }
                        )

                pbar.update(1)

        pbar.close()
        sorted_res = sorted(res, key=lambda x: x["res_id"])
        for d in sorted_res:
            del d["res_id"]
        return sorted_res

    def generate_until_and_loglikelihood_verbose(
        self, instances: List[RequestInstance], requests: List[GenerateUntilAndLoglikelihoodRequest]
    ):
        """
        seperate and copy arguments into both request types. run independently, then combine results here.
        """
        results_for_requests: List[dict] = []

        # Copy request context for perplexity
        requests_ll: List[GenerateUntilAndLoglikelihoodRequest] = copy.deepcopy(requests)
        for request in requests_ll:
            if hasattr(request, "perplexity_context"):
                request.context = request.perplexity_context

        loglikelihood_requests = [
            LoglikelihoodRequest(context=req.context, continuation=req.continuation)
            for req in requests_ll
        ]
        ll_output = self.loglikelihood_verbose(requests=loglikelihood_requests)
        results_ll = collate_results(instances, ll_output)

        generate_until_requests = [
            GenerateUntilRequest(
                context=req.context,
                stop_sequences=req.stop_sequences,
                generation_kwargs=req.generation_kwargs,
            )
            for req in requests
        ]
        gen_output = self.generate_until_verbose(requests=generate_until_requests)
        results_generate = collate_results(instances, gen_output)

        results_for_requests = []
        for ll_result, generate_result in zip(results_ll, results_generate):
            # These two lists will have different "request" and "model_resps" fields
            result = ll_result

            # Copy the unique request types
            result["model_resps_loglikelihood"] = ll_result["model_resps"]
            result["model_resps_generate"] = generate_result["model_resps"]
            result["request_loglikelihood"] = ll_result["request"]
            result["request_generate"] = generate_result["request"]

            # The final result is a mix of both results
            result["model_resps"] = {
                # Generated continuation
                "continuation": result["model_resps_generate"]["continuation"],
                # Logits under the provided correct continuation
                "sum_logits": result["model_resps_loglikelihood"]["sum_logits"],
                "num_tokens": result["model_resps_loglikelihood"]["num_tokens"],
                "num_tokens_all": result["model_resps_loglikelihood"]["num_tokens_all"],
            }

            results_for_requests += [result]

        return results_for_requests
