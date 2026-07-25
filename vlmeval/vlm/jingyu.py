# Copyright Larry. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import logging
import os
import warnings
from typing import Any, Optional

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from vllm import LLM, SamplingParams

from .base import BaseModel

VLLM_MAX_IMAGE_INPUT_NUM = 24


class Jingyu(BaseModel):
    """Jingyu multimodal model with HuggingFace and vLLM inference backends."""

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(
        self,
        model_path: str,
        use_vllm: bool = False,
        max_new_tokens: int = 4096,
        temperature: float = 0.1,
        top_p: float = 0.95,
        top_k: int = 20,
        repetition_penalty: float = 1.1,
        system_prompt: Optional[str] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initializes the Jingyu evaluator wrapper.

        Args:
            model_path (str): Path or HF id of an exported Jingyu checkpoint.
            use_vllm (bool): If True, use vLLM; otherwise HuggingFace. Defaults to False.
            max_new_tokens (int): Max tokens to generate. Defaults to 4096.
            temperature (float): Sampling temperature; <=0 disables sampling. Defaults to 0.1.
            top_p (float): Nucleus sampling threshold. Defaults to 0.95.
            top_k (int): Top-k sampling threshold. Defaults to 20.
            repetition_penalty (float): Repetition penalty. Defaults to 1.1.
            system_prompt (Optional[str]): Optional system prompt. Defaults to None.
            verbose (bool): Whether to print prompts/responses. Defaults to False.
            **kwargs (Any): Extra options such as `limit_mm_per_prompt`, `device_map`,
                `dtype`/`torch_dtype`, `gpu_utils`, and `max_num_seqs`.
        """
        assert model_path is not None
        self.model_path = model_path
        self.use_vllm = use_vllm
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.limit_mm_per_prompt = kwargs.pop("limit_mm_per_prompt", VLLM_MAX_IMAGE_INPUT_NUM)

        do_sample = kwargs.pop("do_sample", temperature is not None and temperature > 0)
        self.generate_kwargs = dict(max_new_tokens=self.max_new_tokens, do_sample=do_sample, use_cache=True)
        if do_sample:
            self.generate_kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)
        if repetition_penalty != 1.0:
            self.generate_kwargs["repetition_penalty"] = repetition_penalty

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        if self.use_vllm:
            self._init_vllm(**kwargs)
        else:
            self._init_transformers(**kwargs)
        torch.cuda.empty_cache()

    def _init_transformers(self, **kwargs: Any) -> None:
        """Loads the model with HuggingFace Transformers.

        Args:
            **kwargs (Any): May include `device_map`, `dtype`, or `torch_dtype`.
        """
        load_kwargs = dict(trust_remote_code=True, device_map=kwargs.pop("device_map", "auto"))
        # Prefer ``dtype`` (newer transformers); fall back to ``torch_dtype``.
        dtype = kwargs.pop("dtype", kwargs.pop("torch_dtype", "auto"))
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, dtype=dtype, **load_kwargs)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=dtype, **load_kwargs)
        self.model.eval()

    def _init_vllm(self, **kwargs: Any) -> None:
        """Loads the model with vLLM (requires `jingyu_vllm` plugin).

        Args:
            **kwargs (Any): May include `gpu_utils` and `max_num_seqs`.
        """
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        gpu_count = torch.cuda.device_count()
        tp_size = gpu_count if gpu_count > 0 else 1
        logging.info(f"Using vLLM for {self.model_path} inference with {tp_size} GPUs (available: {gpu_count})")
        if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn":
            logging.warning("VLLM_WORKER_MULTIPROC_METHOD is not set to spawn. Use 'export VLLM_WORKER_MULTIPROC_METHOD=spawn'")
        self.llm = LLM(
            model=self.model_path,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": self.limit_mm_per_prompt},
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=kwargs.get("gpu_utils", 0.9),
            max_num_seqs=kwargs.get("max_num_seqs", 8),
        )

    def _prepare_content(self, message: list[dict]) -> tuple[list[dict], list[Image.Image]]:
        """Converts a VLMEvalKit message list into chat content and PIL images.

        Args:
            message (list[dict]): Items with keys `type` (`text`/`image`) and `value`.

        Returns:
            tuple[list[dict], list[Image.Image]]: Chat content parts and opened RGB images.

        Raises:
            ValueError: If an item has an unsupported `type`.
        """
        content: list[dict] = []
        images: list[Image.Image] = []
        for item in message:
            if item["type"] == "text":
                content.append({"type": "text", "text": item["value"]})
            elif item["type"] == "image":
                if len(images) >= self.limit_mm_per_prompt:
                    warnings.warn(f"Number of images exceeds limit {self.limit_mm_per_prompt}; extra images are dropped.")
                    continue
                images.append(Image.open(item["value"]).convert("RGB"))
                content.append({"type": "image"})
            else:
                raise ValueError(f"Unsupported message type: {item['type']}")
        return content, images

    def _build_prompt_and_images(self, message: list[dict]) -> tuple[str, list[Image.Image]]:
        """Builds a chat-templated prompt string and the corresponding images.

        Args:
            message (list[dict]): VLMEvalKit multimodal message list.

        Returns:
            tuple[str, list[Image.Image]]: Prompt text and PIL images.
        """
        content, images = self._prepare_content(message)
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": content})
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if self.verbose:
            print(f"\033[31m{messages}\033[0m")
            print(f"\033[33m{prompt}\033[0m")
        return prompt, images

    def _decode_hf_output(self, generated_ids: torch.Tensor, prompt_len: int, has_image: bool) -> str:
        """Decodes HF generate outputs into a response string.

        Multimodal generation may return only completion ids; otherwise the prompt
        prefix is sliced away when the full sequence is returned.

        Args:
            generated_ids (torch.Tensor): Token ids from `model.generate`.
            prompt_len (int): Length of the prompt `input_ids`.
            has_image (bool): Whether the request included images.

        Returns:
            str: Decoded assistant response.
        """
        if generated_ids.shape[1] > prompt_len:
            new_tokens = generated_ids[:, prompt_len:]
        else:
            new_tokens = generated_ids
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    def generate_inner_transformers(self, message: list[dict], dataset: Optional[str] = None) -> str:
        """Runs one-step inference with HuggingFace Transformers.

        Args:
            message (list[dict]): VLMEvalKit multimodal message list.
            dataset (Optional[str]): Dataset name (unused). Defaults to None.

        Returns:
            str: Model prediction text.
        """
        del dataset
        prompt, images = self._build_prompt_and_images(message)
        processor_kwargs: dict[str, Any] = dict(text=prompt, return_tensors="pt")
        if images:
            processor_kwargs["images"] = images if len(images) > 1 else images[0]
        inputs = self.processor(**processor_kwargs)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **self.generate_kwargs)
        response = self._decode_hf_output(generated_ids, prompt_len=inputs["input_ids"].shape[1], has_image=bool(images))
        if self.verbose:
            print(f"\033[32m{response}\033[0m")
        return response

    def generate_inner_vllm(self, message: list[dict], dataset: Optional[str] = None) -> str:
        """Runs one-step inference with vLLM.

        Args:
            message (list[dict]): VLMEvalKit multimodal message list.
            dataset (Optional[str]): Dataset name (unused). Defaults to None.

        Returns:
            str: Model prediction text.
        """
        del dataset
        prompt, images = self._build_prompt_and_images(message)
        sampling_params = SamplingParams(
            temperature=self.temperature if self.temperature is not None else 0.0,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
        )
        req: dict[str, Any] = {"prompt": prompt}
        if images:
            req["multi_modal_data"] = {"image": images}
        response = self.llm.generate([req], sampling_params=sampling_params)[0].outputs[0].text
        if self.verbose:
            print(f"\033[32m{response}\033[0m")
        return response

    def generate_inner(self, message: list[dict], dataset: Optional[str] = None) -> str:
        """Dispatches generation to the configured backend.

        Args:
            message (list[dict]): VLMEvalKit multimodal message list.
            dataset (Optional[str]): Dataset name forwarded to backends. Defaults to None.

        Returns:
            str: Model prediction text.
        """
        if self.use_vllm:
            return self.generate_inner_vllm(message, dataset=dataset)
        return self.generate_inner_transformers(message, dataset=dataset)
