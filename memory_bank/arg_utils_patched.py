# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import json
import os
import argparse
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from functools import cached_property
from pathlib import Path
import logging
from typing import Any, Dict, List, Literal, Mapping, Optional, Type, Union
from nimlib import (
    NIM_MAX_IMAGES_PER_PROMPT_ENV_NAME,
    NIM_RELAX_MEM_CONSTRAINTS_ENV_NAME,
    VLLM_ALLOW_LONG_MAX_MODEL_LEN_ENV_NAME,
    NIM_USE_DYNAMO_ENV_NAME,
    NIM_USE_DYNAMO_NIM_WORKERS_ENV_NAME,
)

logger = logging.getLogger(__name__)


# Shared helpers for ModelOpt quantization metadata
def _read_hf_quant_config(model_path: str) -> Optional[dict]:
    """Load hf_quant_config.json if present for the given model path."""
    hf_quant_config_path = f"{model_path}/hf_quant_config.json"
    if os.path.exists(hf_quant_config_path):
        try:
            with open(hf_quant_config_path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _producer_is_modelopt(hf_quant_config: Optional[dict]) -> bool:
    return bool(
        hf_quant_config
        and hf_quant_config.get("producer", {}).get("name") == "modelopt"
    )


# Import optional dependencies silently at module level
# These are only used inside container/LLM environments
try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoConfig, PretrainedConfig
    from nimlib.nvml import GPUUnit
except ImportError:
    AutoConfig = None
    PretrainedConfig = None
    GPUUnit = None

try:
    from vllm.config import MultiModalConfig
except ImportError:
    MultiModalConfig = None


class CheckpointType(Enum):
    HF = "HF-SAFETENSOR"
    TRTLLM_ENGINE = "TRTLLM-ENGINE"
    TRTLLM_CKPT = "TRTLLM-CKPT"
    GGUF = "GGUF"

    def to_string(self):
        return self.value.lower()

    @classmethod
    def from_string(cls, value: str):
        for member in cls:
            if member.value == value.upper():
                return member
        raise ValueError(f"'{value}' is not a valid Value")

    @classmethod
    def get_all_types(cls):
        return [member.to_string() for member in cls]


class EngineType(Enum):
    TRTLLM = "trtllm"
    VLLM = "vllm"
    SGLANG = "sglang"
    TRTLLM_LLM_API = "trtllm_llm_api"

    @staticmethod
    def from_name(name: str):
        try:
            return {
                "tensorrt_llm": EngineType.TRTLLM,
                "vllm": EngineType.VLLM,
                "sglang": EngineType.SGLANG,
                "trtllm_llm_api": EngineType.TRTLLM_LLM_API,
            }[name]
        except KeyError:
            raise Exception(f"Invalid EngineType string {name}")


@dataclass
class AsyncEngineArgsBase:
    tllm_buildable: Optional[bool] = None
    tllm_engine_config_json_str: Optional[str] = None
    tllm_ckpt_config_json_str: Optional[str] = None
    tllm_runtime_config_json_str: Optional[str] = None
    checkpoint_type: Optional[CheckpointType] = None
    profile_id: Optional[str] = None
    selected_gpus: List[GPUUnit] = field(default_factory=list)
    comm_orchestrator: Optional[bool] = None

    @property
    def is_hf_model(self):
        if self.checkpoint_type is None:
            return False
        return self.checkpoint_type.to_string().lower() == "hf-safetensor"

    @property
    def is_trtllm_ckpt(self):
        if self.checkpoint_type is None:
            return False
        return self.checkpoint_type.to_string().lower() == "trtllm-ckpt"

    @property
    def is_trtllm_engine(self):
        if self.checkpoint_type is None:
            return False
        return self.checkpoint_type.to_string().lower() == "trtllm-engine"

    def to_dict(self):
        async_engine_args_dict = asdict(self)
        async_engine_args_dict["selected_gpus"] = (
            [gpu_unit.asdict() for gpu_unit in self.selected_gpus]
            if self.selected_gpus
            else []
        )
        if self.checkpoint_type is not None:
            async_engine_args_dict["checkpoint_type"] = self.checkpoint_type.to_string()
        return async_engine_args_dict

    def detect_engine_type(self, llm_engine_tag: str | None = None):
        from nimlib.llm.profile_utils import is_trt_llm_model
        from nimlib.env import get_backend_type_from_env

        # Check explicit engine tag first
        if llm_engine_tag is not None:
            try:
                return EngineType.from_name(llm_engine_tag)
            except Exception:
                pass

        # Check explicit BACKEND_TYPE env var
        backend_type = get_backend_type_from_env()
        if backend_type == "sglang":
            return EngineType.SGLANG
        if backend_type == "vllm":
            return EngineType.VLLM
        if backend_type in ("tensorrt_llm", "tensorrt_llm_nv_ext"):
            return EngineType.TRTLLM_LLM_API

        # Fallback to model-based detection
        if is_trt_llm_model(self):
            return EngineType.TRTLLM_LLM_API
        if Path(self.model).is_dir():
            return EngineType.VLLM

        raise Exception("Could not determine EngineType")


# custom data classes for dynamo launch
@dataclass
class CacheConfig:
    """Configuration for the KV cache."""

    block_size: Optional[int] = None  # type: ignore
    gpu_memory_utilization: float = 0.6
    swap_space: float = 4
    cache_dtype: str = "auto"
    is_attention_free: bool = False
    num_gpu_blocks_override: Optional[int] = None
    sliding_window: Optional[int] = None
    enable_prefix_caching: Optional[bool] = None
    prefix_caching_hash_algo: str = "builtin"
    cpu_offload_gb: float = 0
    calculate_kv_scales: bool = False
    cpu_kvcache_space_bytes: Optional[int] = None
    mamba_page_size_padded: Optional[int] = None
    mamba_cache_dtype: str = "auto"
    mamba_ssm_cache_dtype: str = "auto"
    # Will be set after profiling.
    num_gpu_blocks: Optional[int] = field(default=None, init=False)
    num_cpu_blocks: Optional[int] = field(default=None, init=False)
    kv_sharing_fast_prefill: bool = False


@dataclass
class ParallelConfig:
    """Configuration for the distributed execution."""

    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    data_parallel_size_local: int = 1
    data_parallel_rank: int = 0
    data_parallel_rank_local: Optional[int] = None
    data_parallel_master_ip: str = "127.0.0.1"
    data_parallel_rpc_port: int = 29550
    data_parallel_master_port: int = 29500
    data_parallel_backend: str = "mp"
    data_parallel_external_lb: bool = False
    data_parallel_hybrid_lb: bool = False
    enable_expert_parallel: bool = False
    enable_eplb: bool = False
    num_redundant_experts: int = 0
    eplb_window_size: int = 1000
    eplb_step_interval: int = 3000
    eplb_log_balancedness: bool = False
    max_parallel_loading_workers: Optional[int] = None
    disable_custom_all_reduce: bool = False
    ray_workers_use_nsight: bool = False
    ray_runtime_env: Optional[str] = None
    placement_group: Optional[str] = None
    distributed_executor_backend: Optional[str] = None
    worker_cls: str = "auto"
    sd_worker_cls: str = "auto"
    worker_extension_cls: str = ""
    world_size: int = field(init=False)
    rank: int = 0
    enable_multimodal_encoder_data_parallel: bool = False


@dataclass
class SchedulerConfig:
    """Scheduler configuration."""

    runner_type: str = "generate"
    max_num_batched_tokens: Optional[int] = None  # type: ignore
    max_num_seqs: Optional[int] = None  # type: ignore
    max_model_len: Optional[int] = None  # type: ignore
    max_num_partial_prefills: int = 1
    max_long_partial_prefills: int = 1
    long_prefill_token_threshold: int = 0
    num_lookahead_slots: int = 0
    cuda_graph_sizes: list[int] = field(default_factory=list)
    delay_factor: float = 0.0
    enable_chunked_prefill: Optional[bool] = None  # type: ignore
    is_multimodal_model: bool = False
    max_num_encoder_input_tokens: int = field(init=False)
    encoder_cache_size: int = field(init=False)
    preemption_mode: Optional[str] = None
    send_delta_data: bool = False
    policy: str = "fcfs"
    chunked_prefill_enabled: bool = field(init=False)
    disable_chunked_mm_input: bool = False
    scheduler_cls: Union[str, type[object]] = "vllm.core.scheduler.Scheduler"
    disable_hybrid_kv_cache_manager: bool = False
    async_scheduling: bool = False


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) support."""

    max_lora_rank: int = 16
    max_loras: int = 1
    lora_extra_vocab_size: int = 256
    lora_dtype: Optional[str] = "auto"
    max_cpu_loras: Optional[int] = None
    peft_source: Optional[str] = None


@dataclass
class ModelConfig:
    """Configuration for the model and tokenizer."""

    model: str = "facebook/opt-125m"
    hf_config_path: Optional[str] = None
    task: Literal[
        "auto",
        "generate",
        "embedding",
        "embed",
        "classify",
        "score",
        "reward",
        "transcription",
    ] = "auto"
    tokenizer: Optional[str] = None
    tokenizer_mode: Optional[str] = None
    trust_remote_code: bool = False
    allowed_local_media_path: Optional[str] = None
    dtype: str = "auto"
    seed: Optional[int] = None
    revision: Optional[str] = None
    code_revision: Optional[str] = None
    rope_scaling: Optional[Dict[str, Any]] = field(default_factory=dict)
    rope_theta: Optional[float] = None
    hf_token: Optional[str] = None
    hf_overrides: Optional[Dict[str, Any]] = field(default_factory=dict)
    tokenizer_revision: Optional[str] = None
    max_model_len: Optional[int] = None
    quantization: Optional[str] = None
    enforce_eager: bool = False
    max_seq_len_to_capture: int = 8192
    max_logprobs: int = 20
    disable_sliding_window: bool = False
    disable_cascade_attn: bool = False
    skip_tokenizer_init: bool = False
    served_model_name: Optional[Union[str, List[str]]] = None
    limit_mm_per_prompt: Optional[Mapping[str, int]] = None
    use_async_output_proc: bool = True
    config_format: str = "auto"
    mm_processor_kwargs: Optional[Dict[str, Any]] = None
    disable_mm_preprocessor_cache: bool = False
    override_neuron_config: Optional[Dict[str, Any]] = field(default_factory=dict)
    override_pooler_config: Optional[Any] = None
    logits_processor_pattern: Optional[str] = None
    generation_config: Optional[str] = "auto"
    override_generation_config: Optional[Dict[str, Any]] = field(default_factory=dict)
    enable_sleep_mode: bool = False
    model_impl: str = "auto"
    enable_prompt_embeds: bool = False
    hf_config: Optional[PretrainedConfig] = None

    def get_max_model_len(self) -> int:
        return _get_and_verify_max_len(
            self.hf_config,
            self.max_model_len,
            self.disable_sliding_window,
            self.get_hf_config_sliding_window(),
        )

    @cached_property
    def sliding_window_len(self) -> Optional[Union[int, List[Optional[int]]]]:
        """Get the sliding window size, or None if disabled."""
        return self.get_hf_config_sliding_window()

    @cached_property
    def hf_text_config(self):
        # This block should be unnecessary after https://github.com/huggingface/transformers/pull/37517
        if hasattr(self.hf_config, "thinker_config"):
            # TODO(suyang.fy): Refactor code.
            #  For Qwen2.5-Omni, change hf_text_config to
            #  thinker_config.text_config.
            return self.hf_config.thinker_config.text_config

        text_config = self.hf_config.get_text_config()

        if text_config is not self.hf_config:
            # The code operates under the assumption that text_config should have
            # `num_attention_heads` (among others). Assert here to fail early
            # if transformers config doesn't align with this assumption.
            assert hasattr(text_config, "num_attention_heads")

        return text_config

    def get_hf_config_sliding_window(self) -> Union[Optional[int], list[Optional[int]]]:
        """Get the sliding window size, or None if disabled."""

        # Some models, like Qwen2 and Qwen1.5, use `use_sliding_window` in
        # addition to sliding window size. We check if that field is present
        # and if it's False, return None.
        if (
            hasattr(self.hf_config, "use_sliding_window")
            and not self.hf_text_config.use_sliding_window
        ):
            return None
        return getattr(self.hf_text_config, "sliding_window", None)


@dataclass
class DecodingConfig:
    """Configuration for guided decoding and reasoning."""

    backend: str = "auto"
    disable_fallback: bool = False
    disable_any_whitespace: bool = False
    disable_additional_properties: bool = False
    reasoning_backend: str = ""


@dataclass
class _EngineConfig:
    model_config: ModelConfig = field(default_factory=ModelConfig)
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    device: str = "auto"
    lora_config: Optional[LoRAConfig] = field(default_factory=LoRAConfig)


# branch out until feature branch is created for dynamo
if os.environ.get(NIM_USE_DYNAMO_ENV_NAME) and not os.environ.get(
    NIM_USE_DYNAMO_NIM_WORKERS_ENV_NAME
):

    @dataclass
    class NimAsyncEngineArgs(AsyncEngineArgsBase):
        """Generic arguments for engine launched using dynamo.
        Adapted using OSS vLLM 0.9.0 as reference."""

        model: str = "facebook/opt-125m"
        served_model_name: Optional[Union[str, List[str]]] = None
        tokenizer: Optional[str] = None  # type: ignore
        hf_config_path: Optional[str] = None
        task: Literal[
            "auto",
            "generate",
            "embedding",
            "embed",
            "classify",
            "score",
            "reward",
            "transcription",
        ] = "auto"
        skip_tokenizer_init: bool = False
        enable_prompt_embeds: bool = False
        tokenizer_mode: Literal["auto", "slow", "mistral", "custom"] = "auto"
        trust_remote_code: bool = False
        allowed_local_media_path: str = ""
        download_dir: Optional[str] = None
        load_format: str = "auto"
        config_format: str = "auto"
        dtype: Literal["auto", "half", "float16", "bfloat16", "float", "float32"] = (
            "auto"
        )
        kv_cache_dtype: Literal["auto", "fp8", "fp8_e4m3", "fp8_e5m2"] = "auto"
        seed: Optional[int] = None
        max_model_len: Optional[int] = None  # type: ignore
        cuda_graph_sizes: list[int] = field(default_factory=lambda: [512])
        distributed_executor_backend: Optional[
            Literal["ray", "mp", "uni", "external_launcher"]
        ] = None
        pipeline_parallel_size: int = 1
        tensor_parallel_size: int = 1
        data_parallel_size: int = 1
        data_parallel_size_local: Optional[int] = None
        data_parallel_address: Optional[str] = None
        data_parallel_rpc_port: Optional[int] = None
        enable_expert_parallel: bool = False
        max_parallel_loading_workers: Optional[int] = None
        block_size: Optional[int] = None  # type: ignore
        enable_prefix_caching: Optional[bool] = None
        prefix_caching_hash_algo: Literal["builtin", "sha256"] = "builtin"
        disable_sliding_window: bool = False
        disable_cascade_attn: bool = False
        use_v2_block_manager: bool = True
        swap_space: float = 4
        cpu_offload_gb: float = 0
        gpu_memory_utilization: float = 0.6
        max_num_batched_tokens: Optional[int] = None  # type: ignore
        max_num_partial_prefills: int = 1
        max_long_partial_prefills: int = 1
        long_prefill_token_threshold: int = 0
        max_num_seqs: Optional[int] = None  # type: ignore
        max_logprobs: int = 20
        disable_log_stats: bool = False
        revision: Optional[str] = None
        code_revision: Optional[str] = None
        rope_scaling: dict[str, Any] = field(default_factory=dict)
        rope_theta: Optional[float] = None
        hf_token: Optional[Union[bool, str]] = None
        hf_overrides: Optional[dict[str, Any]] = field(default_factory=dict)
        tokenizer_revision: Optional[str] = None
        quantization: Optional[str] = None
        enforce_eager: bool = False
        max_seq_len_to_capture: int = 8192
        disable_custom_all_reduce: bool = False
        tokenizer_pool_size: int = 0
        tokenizer_pool_type: str = "ray"
        tokenizer_pool_extra_config: Optional[Dict[str, Any]] = None
        limit_mm_per_prompt: Optional[Mapping[str, int]] = None
        mm_processor_kwargs: Optional[Dict[str, Any]] = None
        disable_mm_preprocessor_cache: bool = False
        # LoRA fields
        enable_lora: bool = False
        enable_lora_bias: bool = False
        max_loras: int = 1
        max_lora_rank: int = 16
        fully_sharded_loras: bool = False
        max_cpu_loras: Optional[int] = None
        lora_dtype: Optional[Union[str, torch.dtype]] = "auto"
        lora_extra_vocab_size: int = 256
        long_lora_scaling_factors: Optional[tuple[float, ...]] = None
        # PromptAdapter fields
        enable_prompt_adapter: bool = False
        max_prompt_adapters: int = 1
        max_prompt_adapter_token: int = 0

        device: str = "auto"
        num_scheduler_steps: int = 1
        multi_step_stream_outputs: bool = True
        ray_workers_use_nsight: bool = False
        num_gpu_blocks_override: Optional[int] = None
        num_lookahead_slots: int = 0
        model_loader_extra_config: dict = field(default_factory=dict)
        ignore_patterns: Optional[Union[str, List[str]]] = None
        preemption_mode: Optional[str] = None

        scheduler_delay_factor: float = 0.0
        enable_chunked_prefill: Optional[bool] = None  # type: ignore
        disable_chunked_mm_input: bool = False

        guided_decoding_backend: str = "auto"
        guided_decoding_disable_fallback: bool = False
        guided_decoding_disable_any_whitespace: bool = False
        guided_decoding_disable_additional_properties: bool = False
        logits_processor_pattern: Optional[str] = ""

        speculative_config: Optional[Dict[str, Any]] = None

        qlora_adapter_name_or_path: Optional[str] = None
        show_hidden_metrics_for_version: Optional[str] = None
        otlp_traces_endpoint: Optional[str] = None
        collect_detailed_traces: Optional[list[Literal["model", "worker", "all"]]] = (
            None
        )
        disable_async_output_proc: bool = False
        scheduling_policy: Literal["fcfs", "priority"] = "fcfs"
        scheduler_cls: Union[str, Type[object]] = "vllm.core.scheduler.Scheduler"

        override_neuron_config: dict[str, Any] = field(default_factory=dict)
        override_pooler_config: Optional[Any] = None
        compilation_config: Optional[Any] = None
        worker_cls: str = "auto"
        worker_extension_cls: str = "str"

        kv_transfer_config: Optional[Any] = None
        kv_events_config: Optional[Any] = None

        generation_config: str = "auto"
        enable_sleep_mode: bool = False
        override_generation_config: dict[str, Any] = field(default_factory=dict)
        model_impl: str = "auto"

        calculate_kv_scales: bool = False

        additional_config: Optional[Dict[str, Any]] = None
        enable_reasoning: Optional[bool] = None  # DEPRECATED
        reasoning_parser: str = ""

        use_tqdm_on_load: bool = True
        pt_load_map_location: str = "cpu"

        disable_log_requests: bool = False

        # Nim base args
        tllm_buildable: Optional[bool] = None
        tllm_engine_config_json_str: Optional[str] = None
        tllm_ckpt_config_json_str: Optional[str] = None
        tllm_runtime_config_json_str: Optional[str] = None
        checkpoint_type: Optional[CheckpointType] = None
        profile_id: Optional[str] = None
        selected_gpus: List[GPUUnit] = field(default_factory=list)
        comm_orchestrator: Optional[bool] = None
        hf_config: Optional[PretrainedConfig] = None

        def __post_init__(self):
            # Initialize the base class using the values from the dataclass fields
            super().__init__(
                tllm_buildable=self.tllm_buildable,
                tllm_engine_config_json_str=self.tllm_engine_config_json_str,
                tllm_ckpt_config_json_str=self.tllm_ckpt_config_json_str,
                tllm_runtime_config_json_str=self.tllm_runtime_config_json_str,
                checkpoint_type=self.checkpoint_type,
                profile_id=self.profile_id,
                selected_gpus=self.selected_gpus,
                comm_orchestrator=self.comm_orchestrator,
            )
            self._model_config = None
            self._unified_hf_quant_config = None
            # Set modelopt quantization
            self.set_modelopt_quantization()

        @classmethod
        def from_cli_args(cls, args: argparse.Namespace):
            # Get the list of attributes of this dataclass.
            attrs = [attr.name for attr in fields(cls)]

            # Set the attributes from the parsed arguments.
            engine_args = cls(
                **{attr: getattr(args, attr) for attr in attrs if hasattr(args, attr)}
            )

            return engine_args

        def to_dict(self):
            async_engine_args_dict = asdict(self)
            async_engine_args_dict["selected_gpus"] = (
                [gpu_unit.asdict() for gpu_unit in self.selected_gpus]
                if self.selected_gpus
                else []
            )
            if self.hf_config is not None:
                async_engine_args_dict["hf_config"] = self.hf_config.to_dict()
            if self.checkpoint_type is not None:
                async_engine_args_dict["checkpoint_type"] = (
                    self.checkpoint_type.to_string()
                )

            # New from VLLM in 0.9.0RC1, PosixPath and Set are not JSON serializable.
            async_engine_args_dict["compilation_config"] = (
                self._serialize_compilation_config(
                    async_engine_args_dict["compilation_config"]
                )
            )

            return async_engine_args_dict

        @classmethod
        def from_dict(cls, async_engine_args_dict):
            engine_args = cls(**async_engine_args_dict)
            if async_engine_args_dict.get("selected_gpus"):
                engine_args.selected_gpus = [
                    GPUUnit.from_dict(gpu_unit_dict)
                    for gpu_unit_dict in async_engine_args_dict["selected_gpus"]
                ]
            if async_engine_args_dict.get("hf_config"):
                engine_args.hf_config = PretrainedConfig.from_dict(
                    async_engine_args_dict["hf_config"]
                )
            if async_engine_args_dict.get("checkpoint_type"):
                engine_args.checkpoint_type = CheckpointType.from_string(
                    async_engine_args_dict.get("checkpoint_type")
                )

            return engine_args

        def set_modelopt_quantization(self):
            cfg = _read_hf_quant_config(self.model)
            if _producer_is_modelopt(cfg):
                self.quantization = "modelopt"

        def _serialize_compilation_config(self, compilation_config):
            """Serialize compilation config by converting PosixPath objects to strings."""
            if compilation_config is None:
                return None

            # Convert dump_graph_dir PosixPath to string if present
            if (
                compilation_config.get("pass_config", {}).get("dump_graph_dir")
                is not None
            ):
                compilation_config["pass_config"]["dump_graph_dir"] = str(
                    compilation_config["pass_config"]["dump_graph_dir"]
                )

            self._remove_unserializable_fields(compilation_config)

            return json.dumps(compilation_config)

        def _remove_unserializable_fields(self, compilation_config):
            # Remove these fields from the compilation config because they cannot be loaded back in from args
            compilation_config.pop("max_capture_size", None)
            compilation_config.pop("local_cache_dir", None)
            compilation_config.pop("bs_to_padded_graph_size", None)
            compilation_config.pop("enabled_custom_ops", None)
            compilation_config.pop("disabled_custom_ops", None)
            compilation_config.pop("traced_files", None)
            compilation_config.pop("compilation_time", None)
            compilation_config.pop("static_forward_context", None)

        def set_model_config(self):
            # singleton
            if self._model_config is None:
                self._model_config = self.create_model_config()

        def set_hf_config(self):
            self.set_model_config()
            self.set_unified_hf_quant_config()
            # singleton
            if self.hf_config is None:
                self.hf_config = self._model_config.hf_config
                self.hf_config.unified_quant_config = self._unified_hf_quant_config

        def get_hf_config(self):
            if self.hf_config is None:
                self.set_hf_config()
            return self.hf_config

        def get_model_config(self):
            if self._model_config is None:
                self.set_model_config()
            return self._model_config

        def set_unified_hf_quant_config(self):
            cfg = _read_hf_quant_config(self.model)
            if _producer_is_modelopt(cfg):
                self._unified_hf_quant_config = cfg.get("quantization", {})

        def get_unified_hf_quant_config(self):
            if self._unified_hf_quant_config is None:
                self.set_unified_hf_quant_config()
            return self._unified_hf_quant_config

        def create_model_config(self) -> ModelConfig:
            if _check_gguf_file(self.model):
                self.quantization = self.load_format = "gguf"

            config_kwargs = {"trust_remote_code": self.trust_remote_code}
            if self.hf_token is not None:
                config_kwargs["token"] = self.hf_token
            if self.revision is not None:
                config_kwargs["revision"] = self.revision

            hf_config = AutoConfig.from_pretrained(self.model, **config_kwargs)

            model_config = ModelConfig(
                model=self.model,
                hf_config_path=self.hf_config_path,
                task=self.task,
                tokenizer=self.tokenizer,
                tokenizer_mode=self.tokenizer_mode,
                trust_remote_code=self.trust_remote_code,
                allowed_local_media_path=self.allowed_local_media_path,
                dtype=self.dtype,
                seed=self.seed,
                revision=self.revision,
                code_revision=self.code_revision,
                rope_scaling=self.rope_scaling,
                rope_theta=self.rope_theta,
                hf_token=self.hf_token,
                hf_overrides=self.hf_overrides,
                tokenizer_revision=self.tokenizer_revision,
                max_model_len=self.max_model_len,
                quantization=self.quantization,
                enforce_eager=self.enforce_eager,
                max_seq_len_to_capture=self.max_seq_len_to_capture,
                max_logprobs=self.max_logprobs,
                disable_sliding_window=self.disable_sliding_window,
                disable_cascade_attn=self.disable_cascade_attn,
                skip_tokenizer_init=self.skip_tokenizer_init,
                served_model_name=self.served_model_name,
                limit_mm_per_prompt=self.limit_mm_per_prompt,
                use_async_output_proc=not self.disable_async_output_proc,
                config_format=self.config_format,
                mm_processor_kwargs=self.mm_processor_kwargs,
                disable_mm_preprocessor_cache=self.disable_mm_preprocessor_cache,
                override_neuron_config=self.override_neuron_config,
                override_pooler_config=self.override_pooler_config,
                logits_processor_pattern=self.logits_processor_pattern,
                generation_config=self.generation_config,
                override_generation_config=self.override_generation_config,
                enable_sleep_mode=self.enable_sleep_mode,
                model_impl=self.model_impl,
                hf_config=hf_config,
            )

            if os.environ.get(NIM_MAX_IMAGES_PER_PROMPT_ENV_NAME) is not None:
                # we do this here instead of in the vllm args parser
                # because for models not supported in vllm, vllm can't
                # know they are VLMs, and rejects this arg
                if MultiModalConfig is not None:
                    multimodal_config = (
                        getattr(model_config, "multimodal_config") or MultiModalConfig()
                    )
                    multimodal_config.limit_per_prompt["image"] = os.environ.get(
                        NIM_MAX_IMAGES_PER_PROMPT_ENV_NAME
                    )
                    model_config.multimodal_config = multimodal_config
                else:
                    logger.warning(
                        "MultiModalConfig not available, skipping multimodal configuration"
                    )

            return model_config

        def create_engine_config(self) -> _EngineConfig:
            model_config = self.create_model_config()
            cache_config = CacheConfig(
                block_size=self.block_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                swap_space=self.swap_space,
                cache_dtype=self.kv_cache_dtype,
                num_gpu_blocks_override=self.num_gpu_blocks_override,
                sliding_window=None,  # TODO is this needed
                enable_prefix_caching=self.enable_prefix_caching,
            )

            data_parallel_size_local = (
                self.data_parallel_size
                if (self.data_parallel_size_local is None)
                else self.data_parallel_size_local
            )

            data_parallel_address = self.data_parallel_address
            parallel_config = ParallelConfig(
                pipeline_parallel_size=self.pipeline_parallel_size,
                tensor_parallel_size=self.tensor_parallel_size,
                data_parallel_size=self.data_parallel_size,
                data_parallel_size_local=data_parallel_size_local,
                data_parallel_master_ip=data_parallel_address,
                enable_expert_parallel=self.enable_expert_parallel,
                max_parallel_loading_workers=self.max_parallel_loading_workers,
                disable_custom_all_reduce=self.disable_custom_all_reduce,
                ray_workers_use_nsight=self.ray_workers_use_nsight,
                distributed_executor_backend=self.distributed_executor_backend,
                worker_cls=self.worker_cls,
                worker_extension_cls=self.worker_extension_cls,
            )
            scheduler_config = SchedulerConfig(
                max_num_batched_tokens=self.max_num_batched_tokens,
                max_num_seqs=self.max_num_seqs,
                max_model_len=model_config.max_model_len,
                num_lookahead_slots=self.num_lookahead_slots,
            )
            lora_config = (
                LoRAConfig(
                    max_lora_rank=self.max_lora_rank,
                    max_loras=self.max_loras,
                    lora_extra_vocab_size=self.lora_extra_vocab_size,
                    lora_dtype=self.lora_dtype,
                    max_cpu_loras=self.max_cpu_loras,
                )
                if self.enable_lora
                else None
            )
            return _EngineConfig(
                model_config=model_config,
                cache_config=cache_config,
                parallel_config=parallel_config,
                scheduler_config=scheduler_config,
                device=self.device,
                lora_config=lora_config,
            )

    def _check_gguf_file(model_path: str) -> bool:
        return model_path.lower().endswith(".gguf")

    def get_min_sliding_window(sliding_window: Union[int, list[Optional[int]]]) -> int:
        if isinstance(sliding_window, list):
            return min(s for s in sliding_window if s is not None)

        return sliding_window

    # Copied from https://github.com/vllm-project/vllm/blob/bffddd9a05a6d0d3ea04b7baad1966e4f57f94c7/vllm/config.py#L2400
    def _get_and_verify_max_len(
        hf_config: PretrainedConfig,
        max_model_len: Optional[int],
        disable_sliding_window: bool,
        sliding_window_len: Optional[Union[int, List[Optional[int]]]],
        spec_target_max_model_len: Optional[int] = None,
        encoder_config: Optional[Any] = None,
    ) -> int:
        """Get and verify the model's maximum length."""
        derived_max_model_len = float("inf")
        possible_keys = [
            # OPT
            "max_position_embeddings",
            # GPT-2
            "n_positions",
            # MPT
            "max_seq_len",
            # ChatGLM2
            "seq_length",
            # Command-R
            "model_max_length",
            # Whisper
            "max_target_positions",
            # Others
            "max_sequence_length",
            "max_seq_length",
            "seq_len",
        ]
        # Choose the smallest "max_length" from the possible keys.
        max_len_key = None
        for key in possible_keys:
            max_len = getattr(hf_config, key, None)
            if max_len is not None:
                max_len_key = key if max_len < derived_max_model_len else max_len_key
                derived_max_model_len = min(derived_max_model_len, max_len)

        # If sliding window is manually disabled, max_length should be less
        # than the sliding window length in the model config.
        if disable_sliding_window and sliding_window_len is not None:
            sliding_window_len_min = get_min_sliding_window(sliding_window_len)
            max_len_key = (
                "sliding_window"
                if sliding_window_len_min < derived_max_model_len
                else max_len_key
            )
            derived_max_model_len = min(derived_max_model_len, sliding_window_len_min)

        # If none of the keys were found in the config, use a default and
        # log a warning.
        if derived_max_model_len == float("inf"):
            if max_model_len is not None:
                # If max_model_len is specified, we use it.
                return max_model_len

            if spec_target_max_model_len is not None:
                # If this is a speculative draft model, we use the max model len
                # from the target model.
                return spec_target_max_model_len

            default_max_len = 2048
            logger.warning(
                "The model's config.json does not contain any of the following "
                "keys to determine the original maximum length of the model: "
                "%s. Assuming the model's maximum length is %d.",
                possible_keys,
                default_max_len,
            )
            derived_max_model_len = default_max_len

        rope_scaling = getattr(hf_config, "rope_scaling", None)
        if rope_scaling is not None:
            # No need to consider "type" key because of patch_rope_scaling when
            # loading HF config
            rope_type = rope_scaling["rope_type"]

            if rope_type not in ("su", "longrope", "llama3"):
                if disable_sliding_window:
                    # TODO(robertgshaw): Find a model that supports rope_scaling
                    # with sliding window to see if this case should be allowed.
                    raise NotImplementedError(
                        "Disabling sliding window is not supported for models "
                        "with rope_scaling. Please raise an issue so we can "
                        "investigate."
                    )

                # NOTE: rope_type == "default" does not define factor
                # https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/modeling_rope_utils.py
                scaling_factor = rope_scaling.get("factor", 1.0)

                if rope_type == "yarn":
                    derived_max_model_len = rope_scaling[
                        "original_max_position_embeddings"
                    ]
                derived_max_model_len *= scaling_factor

        if encoder_config and "max_seq_length" in encoder_config:
            derived_max_model_len = encoder_config["max_seq_length"]

        # If the user specified a max length, make sure it is smaller than the
        # derived length from the HF model config.
        if max_model_len is None:
            max_model_len = int(derived_max_model_len)
        elif max_model_len > derived_max_model_len:
            # Some models might have a separate key for specifying model_max_length
            # that will be bigger than derived_max_model_len. We compare user input
            # with model_max_length and allow this override when it's smaller.
            model_max_length = getattr(hf_config, "model_max_length", None)
            if model_max_length is not None and max_model_len <= model_max_length:
                if disable_sliding_window:
                    # TODO(robertgshaw): Find a model that has model_max_length
                    # with sliding window to see if this case should be allowed.
                    raise NotImplementedError(
                        "Disabling sliding window is not supported for models "
                        "model_max_length in the config. Please raise an issue "
                        "so we can investigate."
                    )
            else:
                msg = (
                    f"User-specified max_model_len ({max_model_len}) is greater "
                    f"than the derived max_model_len ({max_len_key}="
                    f"{derived_max_model_len} or model_max_length="
                    f"{model_max_length} in model's config.json). This may lead "
                    "to incorrect model outputs or CUDA errors."
                )
                if os.environ.get(VLLM_ALLOW_LONG_MAX_MODEL_LEN_ENV_NAME):
                    logger.warning(
                        "%s Make sure the value is correct and within the "
                        "model context size.",
                        msg,
                    )
                else:
                    raise ValueError(
                        f"{msg} To allow overriding this maximum, set "
                        "the env var VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
                    )
        return int(max_model_len)

else:
    # vLLM is optional and not available on Python 3.13 in our constraints.
    # Try to import; if unavailable, provide a graceful fallback so unrelated
    # code paths (e.g., Triton tests) can import this module without error.
    try:
        from vllm.config import CompilationConfig
        from vllm.engine.arg_utils import AsyncEngineArgs as vllmAsyncEngineArgs

        _VLLM_AVAILABLE = True
    except ImportError:
        CompilationConfig = None  # type: ignore
        vllmAsyncEngineArgs = None  # type: ignore
        _VLLM_AVAILABLE = False

    if _VLLM_AVAILABLE:

        @dataclass
        class NimAsyncEngineArgs(AsyncEngineArgsBase, vllmAsyncEngineArgs):
            # Nim base args
            tllm_buildable: Optional[bool] = None
            tllm_engine_config_json_str: Optional[str] = None
            tllm_ckpt_config_json_str: Optional[str] = None
            tllm_runtime_config_json_str: Optional[str] = None
            checkpoint_type: Optional[CheckpointType] = None
            profile_id: Optional[str] = None
            selected_gpus: List[GPUUnit] = field(default_factory=list)
            comm_orchestrator: Optional[bool] = None
            hf_config: Optional[PretrainedConfig] = None

            def __post_init__(self):
                # Initialize the base class using the values from the dataclass fields
                AsyncEngineArgsBase.__init__(
                    self,
                    tllm_buildable=self.tllm_buildable,
                    tllm_engine_config_json_str=self.tllm_engine_config_json_str,
                    tllm_ckpt_config_json_str=self.tllm_ckpt_config_json_str,
                    tllm_runtime_config_json_str=self.tllm_runtime_config_json_str,
                    checkpoint_type=self.checkpoint_type,
                    profile_id=self.profile_id,
                    selected_gpus=self.selected_gpus,
                    comm_orchestrator=self.comm_orchestrator,
                )
                self._model_config = None
                self._unified_hf_quant_config = None
                # Set modelopt quantization
                self.set_modelopt_quantization()

            @classmethod
            def from_cli_args(cls, args: argparse.Namespace):
                engine_args = vllmAsyncEngineArgs.from_cli_args(args)
                return cls(**engine_args.__dict__)

            def to_dict(self):
                async_engine_args_dict = asdict(self)
                async_engine_args_dict["selected_gpus"] = (
                    [gpu_unit.asdict() for gpu_unit in self.selected_gpus]
                    if self.selected_gpus
                    else []
                )
                if self.hf_config is not None:
                    async_engine_args_dict["hf_config"] = self.hf_config.to_dict()
                if self.checkpoint_type is not None:
                    async_engine_args_dict["checkpoint_type"] = (
                        self.checkpoint_type.to_string()
                    )

                # New from VLLM in 0.9.0RC1, PosixPath and Set are not JSON serializable.
                async_engine_args_dict["compilation_config"] = (
                    self._serialize_compilation_config(
                        async_engine_args_dict["compilation_config"]
                    )
                )

                return async_engine_args_dict

            @classmethod
            def from_dict(cls, async_engine_args_dict):
                engine_args = cls(**async_engine_args_dict)
                if async_engine_args_dict.get("selected_gpus"):
                    engine_args.selected_gpus = [
                        GPUUnit.from_dict(gpu_unit_dict)
                        for gpu_unit_dict in async_engine_args_dict["selected_gpus"]
                    ]
                if async_engine_args_dict.get("hf_config"):
                    engine_args.hf_config = PretrainedConfig.from_dict(
                        async_engine_args_dict["hf_config"]
                    )
                if async_engine_args_dict.get("checkpoint_type"):
                    engine_args.checkpoint_type = CheckpointType.from_string(
                        async_engine_args_dict.get("checkpoint_type")
                    )

                # Deserialize the compilation config
                if async_engine_args_dict.get("compilation_config"):
                    engine_args.compilation_config = (
                        engine_args._deserialize_compilation_config(
                            async_engine_args_dict["compilation_config"]
                        )
                    )
                return engine_args

            def set_modelopt_quantization(self):
                cfg = _read_hf_quant_config(self.model)
                if _producer_is_modelopt(cfg):
                    self.quantization = "modelopt"

            def _serialize_compilation_config(self, compilation_config):
                """Serialize compilation config by converting PosixPath objects to strings."""
                if compilation_config is None:
                    return None

                # Convert dump_graph_dir PosixPath to string if present
                if (
                    compilation_config.get("pass_config", {}).get("dump_graph_dir")
                    is not None
                ):
                    compilation_config["pass_config"]["dump_graph_dir"] = str(
                        compilation_config["pass_config"]["dump_graph_dir"]
                    )

                self._remove_unserializable_fields(compilation_config)

                return json.dumps(compilation_config)

            def _remove_unserializable_fields(self, compilation_config):
                # Remove these fields from the compilation config because they cannot be loaded back in from args
                compilation_config.pop("max_capture_size", None)
                compilation_config.pop("local_cache_dir", None)
                compilation_config.pop("bs_to_padded_graph_size", None)
                compilation_config.pop("enabled_custom_ops", None)
                compilation_config.pop("disabled_custom_ops", None)
                compilation_config.pop("traced_files", None)
                compilation_config.pop("compilation_time", None)
                compilation_config.pop("static_forward_context", None)

            def _deserialize_compilation_config(self, compilation_config):
                if compilation_config is None:
                    return None
                return CompilationConfig(**(json.loads(compilation_config)))

            def set_model_config(self):
                # singleton
                if self._model_config is None:
                    self._model_config = self.create_model_config()

            def set_hf_config(self):
                self.set_model_config()
                self.set_unified_hf_quant_config()
                # singleton
                if self.hf_config is None:
                    self.hf_config = self._model_config.hf_config
                    self.hf_config.unified_quant_config = self._unified_hf_quant_config

            def get_hf_config(self):
                if self.hf_config is None:
                    self.set_hf_config()
                return self.hf_config

            def get_model_config(self):
                if self._model_config is None:
                    self.set_model_config()
                return self._model_config

            def set_unified_hf_quant_config(self):
                hf_quant_config_path = f"{self.model}/hf_quant_config.json"
                if os.path.exists(hf_quant_config_path):
                    with open(hf_quant_config_path, "r") as f:
                        hf_quant_config = json.load(f)
                    if hf_quant_config.get("producer", {}).get("name") == "modelopt":
                        self._unified_hf_quant_config = hf_quant_config.get(
                            "quantization", {}
                        )

            def get_unified_hf_quant_config(self):
                if self._unified_hf_quant_config is None:
                    self.set_unified_hf_quant_config()
                return self._unified_hf_quant_config

            def create_model_config(self):
                model_config = super().create_model_config()

                if os.environ.get(NIM_MAX_IMAGES_PER_PROMPT_ENV_NAME) is not None:
                    # we do this here instead of in the vllm args parser
                    # because for models not supported in vllm, vllm can't
                    # know they are VLMs, and rejects this arg
                    if MultiModalConfig is not None:
                        multimodal_config = (
                            getattr(model_config, "multimodal_config")
                            or MultiModalConfig()
                        )
                        multimodal_config.limit_per_prompt["image"] = os.environ.get(
                            NIM_MAX_IMAGES_PER_PROMPT_ENV_NAME
                        )
                        model_config.multimodal_config = multimodal_config
                    else:
                        logger.warning(
                            "MultiModalConfig not available, skipping multimodal configuration"
                        )

                return model_config
    else:

        @dataclass
        class NimAsyncEngineArgs(AsyncEngineArgsBase):
            """Fallback NimAsyncEngineArgs when vLLM is unavailable.

            This allows importing this module (and running unrelated tests) without
            having vLLM installed. Any attempt to use vLLM-specific entry points
            should raise a clear error.
            """

            # Nim base args
            tllm_buildable: Optional[bool] = None
            tllm_engine_config_json_str: Optional[str] = None
            tllm_ckpt_config_json_str: Optional[str] = None
            tllm_runtime_config_json_str: Optional[str] = None
            checkpoint_type: Optional[CheckpointType] = None
            profile_id: Optional[str] = None
            selected_gpus: List[GPUUnit] = field(default_factory=list)
            comm_orchestrator: Optional[bool] = None
            hf_config: Optional[PretrainedConfig] = None

            @classmethod
            def from_cli_args(cls, args: argparse.Namespace):
                raise ImportError(
                    "vLLM is required for CLI parsing and engine args when using the vLLM backend. "
                    "Please install the 'vllm' extra or set environment to use a non-vLLM backend."
                )

            def to_dict(self):
                async_engine_args_dict = asdict(self)
                async_engine_args_dict["selected_gpus"] = (
                    [gpu_unit.asdict() for gpu_unit in self.selected_gpus]
                    if self.selected_gpus
                    else []
                )
                if self.hf_config is not None:
                    async_engine_args_dict["hf_config"] = getattr(
                        self.hf_config, "to_dict", lambda: {}
                    )()
                if self.checkpoint_type is not None:
                    async_engine_args_dict["checkpoint_type"] = (
                        self.checkpoint_type.to_string()
                    )
                return async_engine_args_dict

        # Nim base args
        tllm_buildable: Optional[bool] = None
        tllm_engine_config_json_str: Optional[str] = None
        tllm_ckpt_config_json_str: Optional[str] = None
        tllm_runtime_config_json_str: Optional[str] = None
        checkpoint_type: Optional[CheckpointType] = None
        profile_id: Optional[str] = None
        selected_gpus: List[GPUUnit] = field(default_factory=list)
        comm_orchestrator: Optional[bool] = None
        hf_config: Optional[PretrainedConfig] = None

        def __post_init__(self):
            # Initialize the base class using the values from the dataclass fields
            AsyncEngineArgsBase.__init__(
                self,
                tllm_buildable=self.tllm_buildable,
                tllm_engine_config_json_str=self.tllm_engine_config_json_str,
                tllm_ckpt_config_json_str=self.tllm_ckpt_config_json_str,
                tllm_runtime_config_json_str=self.tllm_runtime_config_json_str,
                checkpoint_type=self.checkpoint_type,
                profile_id=self.profile_id,
                selected_gpus=self.selected_gpus,
                comm_orchestrator=self.comm_orchestrator,
            )
            self._model_config = None
            self._unified_hf_quant_config = None
            # Set modelopt quantization
            self.set_modelopt_quantization()

        @classmethod
        def from_cli_args(cls, args: argparse.Namespace):
            engine_args = vllmAsyncEngineArgs.from_cli_args(args)
            return cls(**engine_args.__dict__)

        def to_dict(self):
            async_engine_args_dict = asdict(self)
            async_engine_args_dict["selected_gpus"] = (
                [gpu_unit.asdict() for gpu_unit in self.selected_gpus]
                if self.selected_gpus
                else []
            )
            if self.hf_config is not None:
                async_engine_args_dict["hf_config"] = self.hf_config.to_dict()
            if self.checkpoint_type is not None:
                async_engine_args_dict["checkpoint_type"] = (
                    self.checkpoint_type.to_string()
                )

            # New from VLLM in 0.9.0RC1, PosixPath and Set are not JSON serializable.
            async_engine_args_dict["compilation_config"] = (
                self._serialize_compilation_config(
                    async_engine_args_dict["compilation_config"]
                )
            )

            return async_engine_args_dict

        @classmethod
        def from_dict(cls, async_engine_args_dict):
            engine_args = cls(**async_engine_args_dict)
            if async_engine_args_dict.get("selected_gpus"):
                engine_args.selected_gpus = [
                    GPUUnit.from_dict(gpu_unit_dict)
                    for gpu_unit_dict in async_engine_args_dict["selected_gpus"]
                ]
            if async_engine_args_dict.get("hf_config"):
                engine_args.hf_config = PretrainedConfig.from_dict(
                    async_engine_args_dict["hf_config"]
                )
            if async_engine_args_dict.get("checkpoint_type"):
                engine_args.checkpoint_type = CheckpointType.from_string(
                    async_engine_args_dict.get("checkpoint_type")
                )

            # Deserialize the compilation config
            if async_engine_args_dict.get("compilation_config"):
                engine_args.compilation_config = (
                    engine_args._deserialize_compilation_config(
                        async_engine_args_dict["compilation_config"]
                    )
                )
            return engine_args

        def set_modelopt_quantization(self):
            cfg = _read_hf_quant_config(self.model)
            if _producer_is_modelopt(cfg):
                self.quantization = "modelopt"

        def _serialize_compilation_config(self, compilation_config):
            """Serialize compilation config by converting PosixPath objects to strings."""
            if compilation_config is None:
                return None

            # Convert dump_graph_dir PosixPath to string if present
            if (
                compilation_config.get("pass_config", {}).get("dump_graph_dir")
                is not None
            ):
                compilation_config["pass_config"]["dump_graph_dir"] = str(
                    compilation_config["pass_config"]["dump_graph_dir"]
                )

            self._remove_unserializable_fields(compilation_config)

            return json.dumps(compilation_config)

        def _remove_unserializable_fields(self, compilation_config):
            # Remove these fields from the compilation config because they cannot be loaded back in from args
            compilation_config.pop("max_capture_size", None)
            compilation_config.pop("local_cache_dir", None)
            compilation_config.pop("bs_to_padded_graph_size", None)
            compilation_config.pop("enabled_custom_ops", None)
            compilation_config.pop("disabled_custom_ops", None)
            compilation_config.pop("traced_files", None)
            compilation_config.pop("compilation_time", None)
            compilation_config.pop("static_forward_context", None)

        def _deserialize_compilation_config(self, compilation_config):
            if compilation_config is None:
                return None
            return CompilationConfig(**(json.loads(compilation_config)))

        def set_model_config(self):
            # singleton
            if self._model_config is None:
                self._model_config = self.create_model_config()

        def set_hf_config(self):
            self.set_model_config()
            self.set_unified_hf_quant_config()
            # singleton
            if self.hf_config is None:
                self.hf_config = self._model_config.hf_config
                self.hf_config.unified_quant_config = self._unified_hf_quant_config

        def get_hf_config(self):
            if self.hf_config is None:
                self.set_hf_config()
            return self.hf_config

        def get_model_config(self):
            if self._model_config is None:
                self.set_model_config()
            return self._model_config

        def set_unified_hf_quant_config(self):
            hf_quant_config_path = f"{self.model}/hf_quant_config.json"
            if os.path.exists(hf_quant_config_path):
                with open(hf_quant_config_path, "r") as f:
                    hf_quant_config = json.load(f)
                if hf_quant_config.get("producer", {}).get("name") == "modelopt":
                    self._unified_hf_quant_config = hf_quant_config.get(
                        "quantization", {}
                    )

        def get_unified_hf_quant_config(self):
            if self._unified_hf_quant_config is None:
                self.set_unified_hf_quant_config()
            return self._unified_hf_quant_config

        def create_model_config(self):
            model_config = super().create_model_config()

            if os.environ.get(NIM_MAX_IMAGES_PER_PROMPT_ENV_NAME) is not None:
                # we do this here instead of in the vllm args parser
                # because for models not supported in vllm, vllm can't
                # know they are VLMs, and rejects this arg
                if MultiModalConfig is not None:
                    multimodal_config = (
                        getattr(model_config, "multimodal_config") or MultiModalConfig()
                    )
                    multimodal_config.limit_per_prompt["image"] = os.environ.get(
                        NIM_MAX_IMAGES_PER_PROMPT_ENV_NAME
                    )
                    model_config.multimodal_config = multimodal_config
                else:
                    logger.warning(
                        "MultiModalConfig not available, skipping multimodal configuration"
                    )

            return model_config


def _get_vllm_parser_defaults() -> dict:
    """Get default values for vLLM parser from environment variables."""
    import nimlib

    defaults = {}

    # Environment variable mappings
    env_mappings = {
        "guided_decoding_backend": nimlib.NIM_GUIDED_DECODING_BACKEND_ENV_NAME,
        "model": nimlib.NIM_MODEL_NAME_ENV_NAME,
        "served_model_name": nimlib.NIM_SERVED_MODEL_NAME_ENV_NAME,
        "uvicorn_log_level": "UVICORN_LOG_LEVEL",
        "peft_source": nimlib.NIM_PEFT_SOURCE_ENV_NAME,
        "port": nimlib.NIM_SERVER_PORT_ENV_NAME,
        "max_lora_rank": nimlib.NIM_MAX_LORA_RANK_ENV_NAME,
        "peft_refresh_interval": nimlib.NIM_PEFT_REFRESH_INTERVAL_ENV_NAME,
        "disable_log_requests": nimlib.NIM_DISABLE_LOG_REQUESTS_ENV_NAME,
        "max_loras": nimlib.NIM_MAX_GPU_LORAS_ENV_NAME,
        "max_cpu_loras": nimlib.NIM_MAX_CPU_LORAS_ENV_NAME,
        "tokenizer_mode": nimlib.NIM_TOKENIZER_MODE_ENV_NAME,
        "enable_prefix_caching": nimlib.NIM_ENABLE_KV_CACHE_REUSE_ENV_NAME,
        "max_model_len": nimlib.NIM_MAX_MODEL_LEN_ENV_NAME,
        "trust_remote_code": nimlib.NIM_TRUST_REMOTE_CODE_ENV_NAME,
        "enforce_eager": nimlib.NIM_DISABLE_CUDA_GRAPH_ENV_NAME,
        "enable_auto_tool_choice": nimlib.NIM_ENABLE_AUTO_TOOL_CHOICE_ENV_NAME,
        "tool_call_parser": nimlib.NIM_TOOL_CALL_PARSER_ENV_NAME,
        "tool_parser_plugin": "NIM_TOOL_PARSER_PLUGIN",
        "chat_template": nimlib.NIM_CHAT_TEMPLATE_ENV_NAME,
        "gpu_memory_utilization": nimlib.NIM_KV_CACHE_PERCENT_ENV_NAME,
        "enable_lora": nimlib.NIM_LORA_ENV_NAME,
    }

    for arg_name, env_name in env_mappings.items():
        env_value = os.environ.get(env_name)
        if env_value is not None:
            # Convert string env values to appropriate types
            if arg_name in [
                "port",
                "max_lora_rank",
                "peft_refresh_interval",
                "max_loras",
                "max_cpu_loras",
                "max_model_len",
            ]:
                try:
                    defaults[arg_name] = int(env_value)
                except ValueError:
                    logger.warning(f"Invalid integer value for {env_name}: {env_value}")
            elif arg_name in ["gpu_memory_utilization"]:
                try:
                    defaults[arg_name] = float(env_value)
                except ValueError:
                    logger.warning(f"Invalid float value for {env_name}: {env_value}")
            elif arg_name in [
                "trust_remote_code",
                "enforce_eager",
                "enable_prefix_caching",
                "disable_log_requests",
                "enable_auto_tool_choice",
                "enable_lora",
            ]:
                defaults[arg_name] = env_value.lower() in ("true", "1", "yes", "on")
            else:
                defaults[arg_name] = env_value

    # Special handling for enable_lora - only set if peft_source is provided
    if defaults.get("peft_source") is not None:
        defaults["enable_lora"] = True

    return defaults


# Set num_gpu_blocks_override based on environment variables
def _maybe_set_gpu_blocks_override(engine_args):
    """Set GPU blocks override based on environment variables."""
    should_relax_mem_constraints = os.environ.get(NIM_RELAX_MEM_CONSTRAINTS_ENV_NAME)
    max_seq_len_factor = os.environ.get("NIM_NUM_KV_CACHE_SEQ_LENS")
    if not should_relax_mem_constraints or not max_seq_len_factor:
        return engine_args

    logger.warning(
        "Setting NIM_NUM_KV_CACHE_SEQ_LENS overrides the default KV cache allocation logic and may result in degraded performance or memory errors."
    )

    try:
        engine_config = engine_args.create_engine_config()
        max_seq_len = engine_config.model_config.max_model_len
        max_tokens_kv_cache = int(float(max_seq_len_factor) * max_seq_len)

        if max_tokens_kv_cache < max_seq_len:
            logger.warning(
                f"The maximum tokens in the KV cache ({max_tokens_kv_cache}) must be >= to the maximum sequence length of the model ({max_seq_len})"
            )
            logger.warning(f"Setting maximum tokens in the KV cache to {max_seq_len}")
            max_tokens_kv_cache = max_seq_len
            logger.info(f"Max tokens in KV cache set to {max_tokens_kv_cache}")

        block_size = engine_config.cache_config.block_size
        if block_size is not None:
            num_gpu_blocks = max_tokens_kv_cache // block_size
            engine_args.num_gpu_blocks_override = num_gpu_blocks
    except Exception as e:
        logger.warning(f"Failed to set GPU blocks override: {e}")

    return engine_args


def create_nim_async_engine_args_from_env() -> NimAsyncEngineArgs:
    """
    Create NimAsyncEngineArgs instance populated entirely from environment variables.

    This function reads ALL supported environment variables and creates a fully
    configured NimAsyncEngineArgs instance. This is the most convenient way to
    configure the engine when all settings come from the environment.

    Returns:
        NimAsyncEngineArgs: Fully configured instance from environment variables

    Example:
        # Set environment variables
        os.environ['MODEL'] = 'facebook/opt-125m'
        os.environ['TENSOR_PARALLEL_SIZE'] = '2'
        os.environ['NIM_ENABLE_LORA'] = 'true'
        os.environ['NIM_MAX_MODEL_LEN'] = '8192'

        # Create engine args from environment
        engine_args = create_nim_async_engine_args_from_env()
    """
    # Get all defaults from environment variables
    env_defaults = _get_vllm_parser_defaults()

    # Create NimAsyncEngineArgs with only basic required fields
    # Most fields have defaults, so we can create an empty instance
    engine_args = NimAsyncEngineArgs()

    # Set all environment variable values on the instance
    for field_name, env_value in env_defaults.items():
        if hasattr(engine_args, field_name):
            setattr(engine_args, field_name, env_value)
        else:
            logger.warning(
                f"Field '{field_name}' not found in NimAsyncEngineArgs, skipping"
            )

    # Apply any additional environment-based processing
    engine_args = _maybe_set_gpu_blocks_override(engine_args)

    return engine_args


def create_nim_async_engine_args_from_json(json_str: str) -> NimAsyncEngineArgs:
    """
    Create NimAsyncEngineArgs instance from JSON string.

    This function parses a JSON string and creates a NimAsyncEngineArgs instance
    with the specified configuration. This is useful for passing complex
    configurations as a single JSON parameter.

    Args:
        json_str: JSON string containing NimAsyncEngineArgs configuration

    Returns:
        NimAsyncEngineArgs: Configured instance from JSON

    Example:
        json_config = '''
        {
            "model": "facebook/opt-125m",
            "tensor_parallel_size": 2,
            "enable_lora": true,
            "max_model_len": 8192,
            "gpu_memory_utilization": 0.85
        }
        '''
        engine_args = create_nim_async_engine_args_from_json(json_config)
    """
    try:
        config_dict = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {e}")

    # Create NimAsyncEngineArgs instance
    engine_args = NimAsyncEngineArgs()

    # Set all values from the JSON config
    for field_name, value in config_dict.items():
        if hasattr(engine_args, field_name):
            # Handle special cases for complex types
            if field_name == "checkpoint_type" and isinstance(value, str):
                try:
                    setattr(engine_args, field_name, CheckpointType(value))
                except ValueError:
                    logger.warning(f"Invalid checkpoint_type value: {value}")
            elif field_name == "selected_gpus" and isinstance(value, list):
                # Convert list of dicts to GPUUnit objects
                gpu_units = []
                for gpu_dict in value:
                    if isinstance(gpu_dict, dict):
                        gpu_units.append(GPUUnit.from_dict(gpu_dict))
                setattr(engine_args, field_name, gpu_units)
            else:
                setattr(engine_args, field_name, value)
        else:
            logger.warning(
                f"Field '{field_name}' not found in NimAsyncEngineArgs, skipping"
            )

    # Apply any additional environment-based processing
    engine_args = _maybe_set_gpu_blocks_override(engine_args)

    return engine_args


def create_nim_async_engine_args() -> NimAsyncEngineArgs:
    """
    Create NimAsyncEngineArgs with correct precedence: JSON -> Environment Variables.

    This function first checks for a JSON configuration in the NIM_ENGINE_ARGS_JSON
    environment variable. If found, it loads from that. Then, it overrides any
    of those values with any individual environment variables that are set.

    This ensures environment variables have higher precedence.

    Returns:
        NimAsyncEngineArgs: The fully configured instance.
    """
    # Start with a base config from JSON if it exists
    json_config = os.environ.get("NIM_ENGINE_ARGS_JSON")
    if json_config:
        logger.info("Loading base configuration from NIM_ENGINE_ARGS_JSON...")
        try:
            engine_args = create_nim_async_engine_args_from_json(json_config)
        except ValueError as e:
            logger.error(
                f"Failed to parse NIM_ENGINE_ARGS_JSON: {e}. Starting with default config."
            )
            engine_args = NimAsyncEngineArgs()
    else:
        # Otherwise, start with a default instance
        engine_args = NimAsyncEngineArgs()

    # Override with individual environment variables
    logger.info("Overriding configuration with individual environment variables...")
    engine_args = update_nim_async_engine_args_from_env(engine_args)

    return engine_args


def update_nim_async_engine_args_from_env(
    engine_args: NimAsyncEngineArgs,
) -> NimAsyncEngineArgs:
    """
    Update an existing NimAsyncEngineArgs instance with values from environment variables.

    This function takes an existing NimAsyncEngineArgs instance and updates any fields
    that have corresponding environment variables set. Only non-None environment values
    will override existing settings.

    Args:
        engine_args: Existing NimAsyncEngineArgs instance to update

    Returns:
        NimAsyncEngineArgs: Updated instance (modifies in place and returns)

    Example:
        # Start with programmatic configuration
        engine_args = NimAsyncEngineArgs(model='facebook/opt-125m', tensor_parallel_size=1)

        # Override with environment variables if set
        os.environ['TENSOR_PARALLEL_SIZE'] = '4'  # This will override the value
        engine_args = update_nim_async_engine_args_from_env(engine_args)
        # Now engine_args.tensor_parallel_size == 4
    """
    env_defaults = _get_vllm_parser_defaults()

    # Update only fields that have environment variable values
    for field_name, env_value in env_defaults.items():
        if hasattr(engine_args, field_name):
            setattr(engine_args, field_name, env_value)
        else:
            logger.warning(f"Field '{field_name}' not found in NimAsyncEngineArgs")

    # Apply any additional environment-based processing
    engine_args = _maybe_set_gpu_blocks_override(engine_args)

    return engine_args
