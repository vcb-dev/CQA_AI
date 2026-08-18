"""Tải tài liệu nội bộ (md/txt) và truy xuất theo từ khóa — fallback khi chưa có FAISS."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger("cqa_ai.knowledge")

_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ỹđĐ]+", re.UNICODE)
_HEADING_RE = re.compile(r"(?m)^#{1,4}\s+.+$")


@dataclass
class KnowledgeChunk:
    title: str
    text: str
    source: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _iter_doc_files(folder: Path):
    if not folder.exists():
        return
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in {".md", ".txt"} and path.is_file():
            yield path


def _split_markdown(text: str, source: str) -> List[KnowledgeChunk]:
    parts = _HEADING_RE.split(text)
    headings = _HEADING_RE.findall(text)
    chunks: List[KnowledgeChunk] = []
    if headings:
        preamble = parts[0].strip() if parts else ""
        if preamble:
            chunks.append(KnowledgeChunk(title=Path(source).stem, text=preamble[:1800], source=source))
        bodies = parts[1:] if len(parts) > 1 else []
        for heading, body in zip(headings, bodies):
            title = heading.lstrip("#").strip()
            content = f"{title}\n{body.strip()}".strip()
            if len(content) < 40:
                continue
            chunks.append(KnowledgeChunk(title=title, text=content[:1800], source=source))
        return chunks or [KnowledgeChunk(title=Path(source).stem, text=text[:1800], source=source)]

    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    buf: List[str] = []
    for para in paras:
        buf.append(para)
        joined = "\n\n".join(buf)
        if len(joined) >= 500:
            chunks.append(KnowledgeChunk(title=Path(source).stem, text=joined[:1800], source=source))
            buf = []
    if buf:
        chunks.append(KnowledgeChunk(title=Path(source).stem, text="\n\n".join(buf)[:1800], source=source))
    return chunks


class KnowledgeBase:
    def __init__(self) -> None:
        self.chunks: List[KnowledgeChunk] = []
        self.sop_text = ""

    def load(self) -> int:
        docs_path = Path(os.getenv("DOCS_PATH") or (_repo_root() / "data" / "docs"))
        knowledge_path = Path(os.getenv("KNOWLEDGE_PATH") or (_repo_root() / "data" / "knowledge"))
        self.chunks = []
        sop_parts: List[str] = []
        for folder in (docs_path, knowledge_path):
            for path in _iter_doc_files(folder) or []:
                try:
                    raw = path.read_text(encoding="utf-8")
                except OSError as exc:
                    logger.warning("Không đọc được %s: %s", path, exc)
                    continue
                if "quy_tac" in path.name.lower() or "cskh" in path.name.lower():
                    sop_parts.append(raw.strip())
                self.chunks.extend(_split_markdown(raw, str(path.name)))
                logger.info("Knowledge: %s (%s chunks)", path.name, len(self.chunks))
        self.sop_text = "\n\n".join(sop_parts).strip()
        logger.info("Knowledge base: %s chunks, SOP=%s chars", len(self.chunks), len(self.sop_text))
        return len(self.chunks)

    def search(self, query: str, k: int = 4) -> List[dict]:
        tokens = {t.lower() for t in _TOKEN_RE.findall(query or "") if len(t) >= 2}
        if not tokens or not self.chunks:
            return [
                {"title": c.title, "snippet": c.text[:400]}
                for c in self.chunks[:k]
            ]
        scored: List[tuple[int, KnowledgeChunk]] = []
        for chunk in self.chunks:
            hay = chunk.text.lower()
            score = sum(1 for t in tokens if t in hay)
            if "quy tắc" in hay or "cskh" in hay:
                score += 1
            if score:
                scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        picked = [c for _, c in scored[:k]] or self.chunks[:k]
        return [{"title": c.title, "snippet": c.text[:400]} for c in picked]
