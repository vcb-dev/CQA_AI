import json
import os
import re
from typing import List

from langchain_openai import ChatOpenAI

from src.models.translate_models import (
    DetectLangRequest,
    DetectLangResponse,
    TranslateBatchRequest,
    TranslateBatchResponse,
    TranslateBatchResultItem,
    TranslateRequest,
    TranslateResponse,
)

LANG_LABELS = {
    "vi": "Tiếng Việt",
    "th": "Tiếng Thái",
    "en": "English",
    "zh": "中文",
    "ja": "日本語",
    "ko": "한국어",
    "id": "Bahasa Indonesia",
    "ms": "Bahasa Melayu",
    "lo": "ພາສາລາວ",
    "km": "ភាសាខ្មែរ",
}


class TranslateService:
    def __init__(self):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.llm = ChatOpenAI(
            model_name="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0.2,
        )

    def _parse_json(self, raw: str) -> dict:
        text = (raw or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                return json.loads(m.group(0))
            raise

    async def translate(self, req: TranslateRequest) -> TranslateResponse:
        text = (req.text or "").strip()
        if not text:
            return TranslateResponse(
                original_text="",
                translated_text="",
                detected_lang="und",
                source_lang=req.source_lang or "auto",
                target_lang=req.target_lang,
                same_language=True,
            )

        source = (req.source_lang or "auto").strip().lower() or "auto"
        target = (req.target_lang or "vi").strip().lower() or "vi"
        direction = (req.direction or "").strip().lower()
        context = [c.strip() for c in (req.context_messages or []) if c and c.strip()][-12:]

        if target == "vi":
            voice = (
                "Dịch sang tiếng Việt tự nhiên, dễ hiểu cho nhân viên CSKH đọc nhanh.\n"
                "Dùng văn nói lịch sự (ạ/nhé khi phù hợp), tránh dịch word-by-word cứng.\n"
                "Ưu tiên nghĩa rõ: hỏi giá / tư vấn trang sức / vàng bạc / size nhẫn / giao hàng.\n"
            )
            if direction == "outbound":
                voice += "Đây là tin shop đã gửi khách — dịch lại cho NV xem mình đã nói gì.\n"
            else:
                voice += "Đây là tin khách gửi — dịch đúng ý khách hỏi.\n"
        else:
            voice = (
                "Dịch sang ngôn ngữ khách, giọng CSKH thân thiện, rõ ý, không máy móc.\n"
                "Giữ phép lịch sự phù hợp văn hóa đích (vd Thái: ครับ/ค่ะ khi hợp).\n"
            )

        context_block = ""
        if context:
            context_block = (
                "Ngữ cảnh hội thoại (cũ → mới), chỉ để hiểu đúng nghĩa, KHÔNG dịch lại các dòng này:\n"
                + "\n".join(f"- {c[:240]}" for c in context)
                + "\n"
            )

        prompt = (
            "Bạn là biên dịch viên CSKH Facebook Messenger cho shop trang sức (vàng/bạc/nhẫn).\n"
            f"{voice}"
            "Giữ nguyên emoji, link, SĐT, mã SKU, số tiền, tên riêng.\n"
            "CHỈ dịch đúng câu trong text_to_translate. KHÔNG thay bằng câu CSKH khác, KHÔNG copy dòng ngữ cảnh.\n"
            "Nếu text_to_translate đã là ngôn ngữ đích: same_language=true, translated_text=nguyên văn.\n"
            "KHÔNG thêm giải thích ngoài bản dịch. KHÔNG bọc markdown.\n"
            "CHỈ trả JSON thuần:\n"
            '{"detected_lang":"xx","translated_text":"...","same_language":false}\n'
            "same_language=true CHỈ khi ngôn ngữ nguồn đã trùng đích (không cần dịch).\n"
            f"source_lang={source}\n"
            f"target_lang={target}\n"
            f"{context_block}"
            f"text_to_translate={json.dumps(text, ensure_ascii=False)}"
        )

        try:
            result = await self.llm.ainvoke(prompt)
            raw = result.content if hasattr(result, "content") else str(result)
            parsed = self._parse_json(str(raw))
            detected = str(
                parsed.get("detected_lang")
                or (source if source != "auto" else "und")
            ).strip().lower()[:16]
            translated = str(parsed.get("translated_text") or "").strip() or text
            # Cùng ngôn ngữ đích → không cần dòng dịch phụ
            same = detected == target or (
                source != "auto" and source == target and translated == text
            )
            if same:
                translated = text
            return TranslateResponse(
                original_text=text,
                translated_text=translated,
                detected_lang=detected,
                source_lang=source,
                target_lang=target,
                same_language=same,
            )
        except Exception:
            return TranslateResponse(
                original_text=text,
                translated_text=text,
                detected_lang=source if source != "auto" else "und",
                source_lang=source,
                target_lang=target,
                same_language=True,
            )

    async def detect_lang(self, req: DetectLangRequest) -> DetectLangResponse:
        # Lấy nhiều tin trong hội thoại (ưu tiên tin khách / tin dài)
        samples: List[str] = [t.strip() for t in (req.texts or []) if t and t.strip()]
        noise = {"[Ảnh]", "[Video]", "[Sticker]", "[attachment]", "[Gửi ảnh/media]"}
        samples = [s for s in samples if s not in noise and len(s) > 1][:60]
        if not samples:
            return DetectLangResponse(lang="vi", lang_label=LANG_LABELS["vi"], confidence="low")

        # Ưu tiên tin dài hơn để nhận diện chắc
        ranked = sorted(samples, key=lambda s: len(s), reverse=True)[:24]
        joined = "\n".join(f"- {s[:220]}" for s in ranked)
        prompt = (
            "Bạn phát hiện ngôn ngữ chính mà KHÁCH đang dùng trong hội thoại Messenger.\n"
            "Bỏ qua tin quá ngắn (vd 'ok', 'a2t') nếu có tin dài hơn.\n"
            "CHỈ trả JSON: {\"lang\":\"xx\",\"confidence\":\"high|medium|low\"}\n"
            "xx = ISO 639-1 (vi, th, en, zh, ja, ko, id, ms, lo, km, ...).\n"
            f"Các tin trong hội thoại:\n{joined}"
        )
        try:
            result = await self.llm.ainvoke(prompt)
            raw = result.content if hasattr(result, "content") else str(result)
            parsed = self._parse_json(str(raw))
            lang = str(parsed.get("lang") or "vi").strip().lower()[:16]
            conf = str(parsed.get("confidence") or "medium").strip().lower()
            if conf not in ("high", "medium", "low"):
                conf = "medium"
            return DetectLangResponse(
                lang=lang,
                lang_label=LANG_LABELS.get(lang, lang),
                confidence=conf,
            )
        except Exception:
            return DetectLangResponse(lang="vi", lang_label=LANG_LABELS["vi"], confidence="low")

    async def translate_batch(self, req: TranslateBatchRequest) -> TranslateBatchResponse:
        """Một lần gọi LLM dịch nhiều tin — nhanh hơn N lần /translate."""
        target = (req.target_lang or "vi").strip().lower() or "vi"
        items = list(req.items or [])[:24]
        if not items:
            return TranslateBatchResponse(items=[], target_lang=target)

        context = [c.strip() for c in (req.context_messages or []) if c and c.strip()][-10:]
        context_block = ""
        if context:
            context_block = (
                "Ngữ cảnh hội thoại (tham khảo, không dịch lại):\n"
                + "\n".join(f"- {c[:200]}" for c in context)
                + "\n"
            )

        payload = [
            {
                "id": it.id,
                "direction": (it.direction or "inbound").strip().lower(),
                "text": (it.text or "").strip()[:500],
            }
            for it in items
            if (it.text or "").strip()
        ]

        prompt = (
            "Bạn là biên dịch viên CSKH Messenger shop trang sức (vàng/bạc/nhẫn).\n"
            f"Dịch TỪNG tin sang `{target}` — tiếng Việt tự nhiên, dễ đọc, đúng nghĩa CSKH.\n"
            "Tránh dịch word-by-word. Giữ emoji, link, SĐT, SKU, số tiền.\n"
            "Nếu tin đã là ngôn ngữ đích: translated_text = nguyên văn, same_language=true, detected_lang=target.\n"
            "CHỈ trả JSON thuần (không markdown):\n"
            '{"items":[{"id":"...","detected_lang":"th","translated_text":"...","same_language":false}]}\n'
            "Phải đủ đúng số id như input, giữ nguyên id.\n"
            f"{context_block}"
            f"input={json.dumps(payload, ensure_ascii=False)}"
        )

        try:
            result = await self.llm.ainvoke(prompt)
            raw = result.content if hasattr(result, "content") else str(result)
            parsed = self._parse_json(str(raw))
            raw_items = parsed.get("items") if isinstance(parsed, dict) else None
            by_id = {}
            if isinstance(raw_items, list):
                for row in raw_items:
                    if not isinstance(row, dict):
                        continue
                    rid = str(row.get("id") or "").strip()
                    if not rid:
                        continue
                    by_id[rid] = row

            out: List[TranslateBatchResultItem] = []
            for it in items:
                src = (it.text or "").strip()
                row = by_id.get(it.id)
                if not row:
                    out.append(
                        TranslateBatchResultItem(
                            id=it.id,
                            original_text=src,
                            translated_text=src,
                            detected_lang="und",
                            same_language=True,
                        )
                    )
                    continue
                detected = str(row.get("detected_lang") or "und").strip().lower()[:16]
                translated = str(row.get("translated_text") or "").strip() or src
                same = detected == target or translated == src
                if same:
                    translated = src
                    detected = target if detected == "und" else detected
                out.append(
                    TranslateBatchResultItem(
                        id=it.id,
                        original_text=src,
                        translated_text=translated,
                        detected_lang=detected,
                        same_language=same,
                    )
                )
            return TranslateBatchResponse(items=out, target_lang=target)
        except Exception:
            # Fallback: trả nguyên văn — BE có thể retry từng tin
            return TranslateBatchResponse(
                items=[
                    TranslateBatchResultItem(
                        id=it.id,
                        original_text=(it.text or "").strip(),
                        translated_text=(it.text or "").strip(),
                        detected_lang="und",
                        same_language=True,
                    )
                    for it in items
                ],
                target_lang=target,
            )
