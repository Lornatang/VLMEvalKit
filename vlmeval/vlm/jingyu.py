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
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from vllm import LLM, SamplingParams

from .base import BaseModel

VLLM_MAX_IMAGE_INPUT_NUM = 24


class Jingyu(BaseModel):
    """Runs Jingyu multimodal inference for VLMEvalKit evaluations."""

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(
        self,
        model_path: str,
        use_vllm: bool = False,
        max_new_tokens: int = 4096,
        temperature: float | None = 0.1,
        top_p: float = 0.95,
        top_k: int = 20,
        repetition_penalty: float = 1.1,
        system_prompt: str | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initializes the Jingyu evaluator wrapper.

        Args:
            model_path (str): Local path or Hugging Face ID of an exported Jingyu checkpoint.
            use_vllm (bool): Whether to use vLLM instead of Hugging Face. Defaults to False.
            max_new_tokens (int): Maximum number of completion tokens. Defaults to 4096.
            temperature (float | None): Sampling temperature. None or nonpositive values disable sampling.
                Defaults to 0.1.
            top_p (float): Nucleus sampling threshold. Defaults to 0.95.
            top_k (int): Top-k sampling threshold. Zero disables top-k filtering. Defaults to 20.
            repetition_penalty (float): Repetition penalty. Defaults to 1.1.
            system_prompt (str | None): System message prepended to the user prompt, or None.
            verbose (bool): Whether to print prompts and responses. Defaults to False.
            **kwargs (Any): Model-loading or vLLM engine options, such as `device_map`, `dtype`,
                `torch_dtype`, `revision`, `tensor_parallel_size`, `gpu_utils`, and `max_num_seqs`.
                `processor_kwargs` configures the processor separately. `limit_mm_per_prompt`
                is a nonnegative integer image limit. `do_sample=False` disables sampling.

        Raises:
            ValueError: The model path is empty or the image limit is not a nonnegative integer.
        """
        super().__init__()
        if not model_path:
            raise ValueError("model_path must be a nonempty path or Hugging Face model ID.")
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
        if not isinstance(self.limit_mm_per_prompt, int) or self.limit_mm_per_prompt < 0:
            raise ValueError("limit_mm_per_prompt must be a nonnegative integer.")

        self.do_sample = bool(kwargs.pop("do_sample", True)) and temperature is not None and temperature > 0
        self.generate_kwargs = dict(max_new_tokens=self.max_new_tokens, do_sample=self.do_sample, use_cache=True)
        if self.do_sample:
            self.generate_kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)
        if repetition_penalty != 1.0:
            self.generate_kwargs["repetition_penalty"] = repetition_penalty

        processor_kwargs = dict(kwargs.pop("processor_kwargs", {}))
        for key in ("revision", "cache_dir", "token", "local_files_only", "trust_remote_code"):
            if key in kwargs:
                processor_kwargs.setdefault(key, kwargs[key])
        processor_kwargs.setdefault("trust_remote_code", True)
        self.processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
        if self.use_vllm:
            self._init_vllm(**kwargs)
        else:
            self._init_transformers(**kwargs)
        torch.cuda.empty_cache()

    def _init_transformers(self, **kwargs: Any) -> None:
        """Loads the model with Hugging Face Transformers.

        Args:
            **kwargs (Any): Options forwarded to `AutoModelForCausalLM.from_pretrained`.

        Raises:
            TypeError: Model construction fails for a reason other than an unsupported `dtype` keyword.
        """
        load_kwargs = dict(kwargs)
        load_kwargs.setdefault("trust_remote_code", True)
        load_kwargs.setdefault("device_map", "auto")
        # Retry legacy Transformers only when the loader rejects the dtype keyword.
        dtype = load_kwargs.pop("dtype", load_kwargs.pop("torch_dtype", "auto"))
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, dtype=dtype, **load_kwargs)
        except TypeError as error:
            if "unexpected keyword argument 'dtype'" not in str(error):
                raise
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=dtype, **load_kwargs)
        self.model.eval()

    def _init_vllm(self, **kwargs: Any) -> None:
        """Loads the model with vLLM (requires `jingyu_vllm` plugin).

        Args:
            **kwargs (Any): Options forwarded to `LLM`. `gpu_utils` aliases `gpu_memory_utilization`,
                and `torch_dtype` aliases `dtype`. Canonical names take precedence over aliases.
                Tensor parallelism defaults to the number of visible GPUs unless explicitly set.
        """
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        gpu_count = torch.cuda.device_count()
        engine_kwargs = dict(kwargs)
        gpu_utils = engine_kwargs.pop("gpu_utils", 0.9)
        torch_dtype = engine_kwargs.pop("torch_dtype", "auto")
        engine_kwargs.setdefault("gpu_memory_utilization", gpu_utils)
        engine_kwargs.setdefault("dtype", torch_dtype)
        engine_kwargs.setdefault("tensor_parallel_size", max(gpu_count, 1))
        engine_kwargs.setdefault("max_num_seqs", 8)
        engine_kwargs.setdefault("trust_remote_code", True)
        engine_kwargs["limit_mm_per_prompt"] = {"image": self.limit_mm_per_prompt}
        tp_size = engine_kwargs["tensor_parallel_size"]
        logging.info(f"Using vLLM for {self.model_path} inference with {tp_size} GPUs (available: {gpu_count})")
        if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn":
            logging.warning("VLLM_WORKER_MULTIPROC_METHOD is not set to spawn. Use 'export VLLM_WORKER_MULTIPROC_METHOD=spawn'")
        self.sampling_params = SamplingParams(
            temperature=self.temperature if self.do_sample else 0.0,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p,
            top_k=-1 if self.top_k == 0 else self.top_k,
            repetition_penalty=self.repetition_penalty,
        )
        self.llm = LLM(model=self.model_path, **engine_kwargs)

    def _prepare_content(self, message: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
        """Converts a VLMEvalKit message list into chat content and PIL images.

        Args:
            message (List[Dict[str, Any]]): Ordered items with `type` (`text` or `image`) and `value` keys.

        Returns:
            A tuple containing the ordered chat content parts and the corresponding RGB images.

        Raises:
            ValueError: A message type is unsupported or the number of images exceeds the limit.
            OSError: An image cannot be opened or decoded.
        """
        image_count = sum(item["type"] == "image" for item in message)
        if image_count > self.limit_mm_per_prompt:
            raise ValueError(
                f"Received {image_count} images, exceeding limit_mm_per_prompt="
                f"{self.limit_mm_per_prompt}. Refusing to evaluate an incomplete image set."
            )
        content: List[Dict[str, Any]] = []
        images: List[Image.Image] = []
        for item in message:
            if item["type"] == "text":
                content.append({"type": "text", "text": item["value"]})
            elif item["type"] == "image":
                with Image.open(item["value"]) as image:
                    images.append(image.convert("RGB"))
                content.append({"type": "image"})
            else:
                raise ValueError(f"Unsupported message type: {item['type']}")
        return content, images

    def _build_prompt_and_images(self, message: List[Dict[str, Any]]) -> Tuple[str, List[Image.Image]]:
        """Builds a chat-templated prompt string and the corresponding images.

        Args:
            message (List[Dict[str, Any]]): Ordered VLMEvalKit text and image items.

        Returns:
            A tuple containing the rendered prompt and its ordered RGB images.
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

        Jingyu image generation uses `inputs_embeds` and returns completion ids only.
        Text-only generation returns the input prefix followed by completion ids.

        Args:
            generated_ids (torch.Tensor): Token IDs or a generation result with a `sequences` attribute.
            prompt_len (int): Number of tokens in the original `input_ids`.
            has_image (bool): Whether the request included images and therefore returns completion IDs only.

        Returns:
            The assistant response with surrounding whitespace removed.
        """
        generated_ids = getattr(generated_ids, "sequences", generated_ids)
        new_tokens = generated_ids if has_image else generated_ids[:, prompt_len:]
        decoded = self.processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return decoded[0].strip()

    def generate_inner_transformers(self, message: List[Dict[str, Any]], dataset: str | None = None) -> str:
        """Runs single-turn inference with Hugging Face Transformers.

        Args:
            message (List[Dict[str, Any]]): Ordered VLMEvalKit text and image items.
            dataset (str | None): Dataset name, accepted for interface compatibility and otherwise unused.

        Returns:
            The generated assistant response.
        """
        del dataset  # Kept for compatibility with the VLMEvalKit model interface.
        prompt, images = self._build_prompt_and_images(message)
        processor_kwargs: Dict[str, Any] = dict(text=prompt, return_tensors="pt")
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

    def generate_inner_vllm(self, message: List[Dict[str, Any]], dataset: str | None = None) -> str:
        """Runs single-turn inference with vLLM.

        Args:
            message (List[Dict[str, Any]]): Ordered VLMEvalKit text and image items.
            dataset (str | None): Dataset name, accepted for interface compatibility and otherwise unused.

        Returns:
            The generated assistant response.
        """
        del dataset  # Kept for compatibility with the VLMEvalKit model interface.
        prompt, images = self._build_prompt_and_images(message)
        req: Dict[str, Any] = {"prompt": prompt}
        if images:
            req["multi_modal_data"] = {"image": images}
        response = self.llm.generate([req], sampling_params=self.sampling_params, use_tqdm=False)[0].outputs[0].text
        if self.verbose:
            print(f"\033[32m{response}\033[0m")
        return response

    def generate_inner(self, message: List[Dict[str, Any]], dataset: str | None = None) -> str:
        """Dispatches generation to the configured backend.

        Args:
            message (List[Dict[str, Any]]): Ordered VLMEvalKit text and image items.
            dataset (str | None): Dataset name forwarded to the selected backend, or None.

        Returns:
            The generated assistant response.

        Raises:
            ValueError: A message type is unsupported or the number of images exceeds the limit.
            OSError: An image cannot be opened or decoded.
        """
        if self.use_vllm:
            return self.generate_inner_vllm(message, dataset=dataset)
        return self.generate_inner_transformers(message, dataset=dataset)
