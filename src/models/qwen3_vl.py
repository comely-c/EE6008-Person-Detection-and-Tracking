"""
Qwen3-VL wrapper, loaded via Hugging Face Transformers.

Requires:
    pip install "transformers>=4.57.0" accelerate safetensors

Qwen3-VL is used here as an independent, non-blocking module: it does
not gate the detection/tracking pipeline. Its exact role (e.g. scene
description, re-identification hints, prompt-based filtering of which
"person" instances matter) should be decided once the baseline
detection+tracking loop is verified to work end to end.
"""

import torch


class Qwen3VLModel:

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_new_tokens: int = 128,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens

        self.model = None
        self.processor = None

    def load(self):
        print("[Qwen3-VL] Loading model...")

        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.processor = AutoProcessor.from_pretrained(self.model_name)

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )

        print("[Qwen3-VL] Model loaded.")

    def analyze(self, image_path: str, prompt: str) -> str:
        """
        Args:
            image_path: path to an image file on disk
            prompt: text instruction, e.g.
                "Identify the people in this image."

        Returns:
            Generated text response.
        """
        if self.model is None:
            raise RuntimeError("Qwen3-VL model is not loaded. Call load() first.")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0]
