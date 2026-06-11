import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Optional

logger = logging.getLogger("talent_ai")


class DeepSeekBalanceService:
    def _resolve_model(self) -> str:
        return (
            os.getenv("AUDIT_LLM_MODEL")
            or os.getenv("ASSISTANT_LLM_MODEL")
            or "deepseek-chat"
        ).strip()

    def _fetch_balance_sync(self) -> dict[str, Any]:
        api_key = (os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
        base_url = (os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")

        if not api_key:
            return {"error": True, "message": "Chưa cấu hình DEEPSEEK_API_KEY trên AI service"}

        req = urllib.request.Request(
            f"{base_url}/user/balance",
            headers={"Authorization": f"Bearer {api_key}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            logger.warning("DeepSeek balance HTTP %s: %s", exc.code, body[:200])
            return {"error": True, "message": "Không lấy được số dư DeepSeek API"}
        except Exception as exc:
            logger.warning("DeepSeek balance fetch failed: %s", exc)
            return {"error": True, "message": "Không lấy được số dư DeepSeek API"}

        infos = payload.get("balance_infos") or []
        info: Optional[dict[str, Any]] = None
        for row in infos:
            if str(row.get("currency", "")).upper() == "USD":
                info = row
                break
        if info is None and infos:
            info = infos[0]

        def parse_amount(value: Any) -> float:
            try:
                return float(str(value or "0").replace(",", ""))
            except (TypeError, ValueError):
                return 0.0

        currency = str((info or {}).get("currency") or "USD").upper()
        return {
            "is_available": bool(payload.get("is_available")),
            "currency": currency,
            "total_balance": parse_amount((info or {}).get("total_balance")),
            "granted_balance": parse_amount((info or {}).get("granted_balance")),
            "topped_up_balance": parse_amount((info or {}).get("topped_up_balance")),
            "model": self._resolve_model(),
        }

    async def get_balance(self) -> dict[str, Any]:
        import asyncio

        return await asyncio.to_thread(self._fetch_balance_sync)
