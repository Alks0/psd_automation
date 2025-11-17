#!/usr/bin/env python3
"""Gemini client helper for PSD layer renaming (SDK + REST fallback)."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import requests

try:
    from google import genai
    from google.genai import types
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "google-genai is required. Install it via pip install google-genai."
    ) from exc


def load_prompt(template_path: Path) -> str:
    return template_path.read_text(encoding="utf-8")


class GeminiRenamer:
    """Wrapper that can use google-genai SDK or REST endpoint."""

    def __init__(
        self,
        model: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_output_tokens: int,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.model = model
        self.base_url = base_url or os.environ.get("GEMINI_ENDPOINT")
        self.base_url = self.base_url.rstrip("/") if self.base_url else None
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("GEMINI_API_KEY is not configured.")
        self.api_key = api_key

        self.use_rest = bool(self.base_url)
        if self.use_rest:
            self.client = None  # type: ignore[assignment]
            self.generation_config = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_output_tokens": max_output_tokens,
            }
        else:
            self.client = genai.Client(api_key=self.api_key)
            # 配置所有安全类别为 BLOCK_NONE
            safety_settings = [
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_CIVIC_INTEGRITY",
                    threshold="BLOCK_NONE",
                ),
            ]
            self.generation_config = types.GenerateContentConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_output_tokens,
                safety_settings=safety_settings,
            )

    def build_contents(
        self,
        prompt_text: str,
        composite_png: Path,
        layer_png: Path,
        context_text: str,
    ):
        parts = [
            types.Part.from_text(text=prompt_text),
            types.Part.from_bytes(data=composite_png.read_bytes(), mime_type="image/png"),
            types.Part.from_bytes(data=layer_png.read_bytes(), mime_type="image/png"),
            types.Part.from_text(text=context_text),
        ]
        return parts

    def rename_layer(
        self,
        prompt_text: str,
        composite_png: Path,
        layer_png: Path,
        psd_name: str,
        original_name: str,
    ) -> str:
        context = f"{psd_name}：{original_name}"
        if self.use_rest:
            return self._rename_via_rest(prompt_text, composite_png, layer_png, context)

        parts = self.build_contents(prompt_text, composite_png, layer_png, context)
        response = self.client.models.generate_content(  # type: ignore[union-attr]
            model=self.model,
            contents=parts,
            config=self.generation_config,
        )
        text = self._extract_text_from_sdk_response(response)
        if not text:
            raise RuntimeError("Gemini response did not contain text output.")
        return text

    def _rename_via_rest(
        self,
        prompt_text: str,
        composite_path: Path,
        layer_path: Path,
        context_text: str,
    ) -> str:
        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        comp_b64 = base64.b64encode(composite_path.read_bytes()).decode("ascii")
        layer_b64 = base64.b64encode(layer_path.read_bytes()).decode("ascii")
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt_text},
                        {"inlineData": {"mimeType": "image/png", "data": comp_b64}},
                        {"inlineData": {"mimeType": "image/png", "data": layer_b64}},
                        {"text": context_text},
                    ],
                }
            ],
            "generationConfig": self.generation_config,
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
            ],
        }
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        if response.status_code >= 400:
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            raise RuntimeError(f"Gemini REST error: {detail}")
        data = response.json()
        text = self._extract_text_from_rest_response(data)
        if not text:
            raise RuntimeError("Gemini response did not contain text output.")
        return text

    @staticmethod
    def _extract_text_from_rest_response(data: Dict[str, Any]) -> str:
        import logging
        logger = logging.getLogger("gemini_client")

        candidates = data.get("candidates") or []
        if not candidates:
            logger.error("No candidates in response: %s", json.dumps(data, ensure_ascii=False)[:500])
            return ""

        for idx, candidate in enumerate(candidates):
            # 检查 finishReason
            finish_reason = candidate.get("finishReason", "UNKNOWN")
            if finish_reason not in ("STOP", "MAX_TOKENS"):
                # 如果是安全过滤，记录详细的安全评分
                if finish_reason == "SAFETY":
                    safety_ratings = candidate.get("safetyRatings", [])
                    logger.warning(
                        "Candidate %d blocked by SAFETY filter. Safety ratings: %s",
                        idx,
                        json.dumps(safety_ratings, ensure_ascii=False)
                    )
                else:
                    logger.warning(
                        "Candidate %d has unusual finishReason: %s. Full candidate: %s",
                        idx,
                        finish_reason,
                        json.dumps(candidate, ensure_ascii=False)[:500]
                    )

            content = candidate.get("content") or {}
            parts = content.get("parts") or []

            # 收集所有文本parts
            texts = []
            for part in parts:
                text = part.get("text")
                if text:
                    texts.append(text)

            if texts:
                full_text = "".join(texts).strip()
                if full_text:
                    # 检查是否完整
                    if finish_reason == "MAX_TOKENS":
                        logger.warning("Response may be truncated (MAX_TOKENS): %s", full_text[:100])
                    return full_text

        logger.error("No valid text found in any candidate. Response: %s", json.dumps(data, ensure_ascii=False)[:500])
        return ""

    @staticmethod
    def _extract_text_from_sdk_response(response) -> str:
        import logging
        logger = logging.getLogger("gemini_client")

        # 尝试快速获取text属性
        if getattr(response, "text", None):
            text = response.text.strip()
            # 检查 finishReason
            if hasattr(response, "candidates") and response.candidates:
                finish_reason = getattr(response.candidates[0], "finish_reason", None)
                if finish_reason and str(finish_reason) not in ("STOP", "1"):  # 1 is STOP in enum
                    logger.warning("Response has unusual finish_reason: %s. Text: %s", finish_reason, text[:100])
            return text

        candidates = getattr(response, "candidates", []) or []
        if not candidates:
            logger.error("No candidates in SDK response")
            return ""

        for idx, candidate in enumerate(candidates):
            # 检查 finishReason
            finish_reason = getattr(candidate, "finish_reason", None)
            if finish_reason:
                finish_reason_str = str(finish_reason)
                if finish_reason_str not in ("STOP", "1", "FinishReason.STOP"):
                    # 如果是安全过滤，记录详细的安全评分
                    if "SAFETY" in finish_reason_str.upper():
                        safety_ratings = getattr(candidate, "safety_ratings", [])
                        logger.warning(
                            "Candidate %d blocked by SAFETY filter. finish_reason: %s, safety_ratings: %s",
                            idx,
                            finish_reason,
                            safety_ratings
                        )
                    else:
                        logger.warning(
                            "Candidate %d has unusual finish_reason: %s",
                            idx,
                            finish_reason
                        )

            content = getattr(candidate, "content", None)
            if not content:
                continue

            parts = getattr(content, "parts", []) or []
            # 收集所有文本parts
            texts = []
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    texts.append(text)

            if texts:
                full_text = "".join(texts).strip()
                if full_text:
                    return full_text

        logger.error("No valid text found in SDK response candidates")
        return ""


def render_prompt(template: str, context: Dict[str, str]) -> str:
    return template.format(**context)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True, help="提示词模板")
    parser.add_argument("--composite", type=Path, required=True)
    parser.add_argument("--layer", type=Path, required=True)
    parser.add_argument("--psd", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", type=Path, help="生成配置 JSON")
    args = parser.parse_args()

    template_text = load_prompt(args.template)
    config = {
        "model": "gemini-2.5-pro",
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 1024,
    }
    if args.config and args.config.exists():
        user_cfg = json.loads(args.config.read_text(encoding="utf-8"))
        config.update(user_cfg)

    renamer = GeminiRenamer(**config)
    prompt = render_prompt(
        template_text,
        {
            "psd_name": args.psd,
            "original_name": args.name,
            "group_path": "",
            "layer_metadata": "",
            "merged_png": str(args.composite),
            "layer_png": str(args.layer),
            "new_name": "NEW",
        },
    )
    result = renamer.rename_layer(
        prompt,
        args.composite,
        args.layer,
        args.psd,
        args.name,
    )
    print(result)


if __name__ == "__main__":
    main()
