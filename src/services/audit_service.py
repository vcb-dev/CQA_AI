
import os
import json
import traceback
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

class AuditService:
    def __init__(self):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        
        self.llm = ChatOpenAI(
            model_name="deepseek-chat", 
            openai_api_key=api_key, 
            openai_api_base=base_url,
            temperature=0.2,
        )
        
        self.rules_text = self._load_rules()

    def _load_rules(self):
        rules_path = "data/docs/rules_extracted.txt"
        if os.path.exists(rules_path):
            with open(rules_path, "r", encoding="utf-8") as f:
                return f.read()
        return "Sử dụng quy tắc CSKH chuẩn: Lễ phép, Dạ/Vâng, Cá nhân hóa tên khách, Tư vấn đúng nhu cầu."

    def _extract_token_usage(self, response) -> dict:
        """Lấy token usage từ LangChain AIMessage (DeepSeek/OpenAI-compatible)."""
        usage = getattr(response, "usage_metadata", None) or {}
        if not usage and hasattr(response, "response_metadata"):
            tu = (response.response_metadata or {}).get("token_usage") or {}
            usage = {
                "input_tokens": tu.get("prompt_tokens", 0),
                "output_tokens": tu.get("completion_tokens", 0),
                "total_tokens": tu.get("total_tokens", 0),
            }
        prompt = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        completion = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        total = int(usage.get("total_tokens") or (prompt + completion))
        model = getattr(self.llm, "model_name", None) or "deepseek-chat"
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "model": model,
        }

    def _format_transcript_line(self, m: dict) -> Optional[str]:
        text = (m.get("text") or m.get("content") or "").strip()
        msg_type = str(m.get("type") or m.get("messageType") or "").lower()
        has_media = bool(m.get("imageUrl") or m.get("attachmentUrl") or m.get("videoUrl"))

        if text in ("[Ảnh]", "[attachment]", "[Sticker]"):
            text = "[Gửi ảnh sản phẩm]"
        elif text == "[Video]":
            text = "[Gửi video]"
        elif not text and (has_media or msg_type in ("image", "video", "sticker")):
            text = "[Gửi ảnh sản phẩm]" if msg_type != "video" else "[Gửi video]"

        if not text:
            return None
        return f"{m.get('sender', '?')}: {text}"

    def _build_transcript_str(self, transcript_data: list) -> str:
        lines = [
            line
            for m in transcript_data
            if (line := self._format_transcript_line(m)) is not None
        ]
        return "\n".join(lines)

    def _empty_token_usage(self) -> dict:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "model": getattr(self.llm, "model_name", None) or "deepseek-chat",
        }

    def _clamp_score_total(self, value, default=0) -> int:
        try:
            n = int(float(value))
        except (TypeError, ValueError):
            n = default
        return max(0, min(100, n))

    def _get_response_schemas(self):
        return [
            ResponseSchema(name="score", description="Điểm số tổng từ 0 đến 100"),
            ResponseSchema(
                name="feedback",
                description="Nhận xét tổng quan ngắn gọn (2-4 câu). KHÔNG liệt kê bullet ưu/nhược ở đây — dùng strengths/weaknesses.",
            ),
            ResponseSchema(
                name="action_items",
                description=(
                    "Danh sách LỖI/VI PHẠM cụ thể — mỗi mục kèm 1 gợi ý trả lời tương ứng. "
                    "Format BẮT BUỘC mỗi dòng: + [mô tả lỗi/vi phạm cụ thể] || [mẫu tin NV copy-paste gửi khách]. "
                    "Dùng dấu || để phân cách. Gợi ý trả lời phải Dạ/Vâng, lịch sự."
                ),
            ),
            ResponseSchema(name="violations", description="Để trống — đã gộp vào action_items."),
            ResponseSchema(name="suggested_replies", description="Để trống — đã gộp vào action_items."),
            ResponseSchema(name="customer_name", description="Tên thật khách hàng từ transcript."),
            ResponseSchema(name="agent_name", description="Tên thật nhân viên rep tin (không phải tên Page)."),
            ResponseSchema(name="score_greeting", description="Điểm tiêu chí Chào hỏi, thiện cảm (0-20)"),
            ResponseSchema(name="score_needs", description="Điểm tiêu chí Khai thác nhu cầu (0-20)"),
            ResponseSchema(name="score_consult", description="Điểm tiêu chí Tư vấn, giải đáp (0-20)"),
            ResponseSchema(name="score_objection", description="Điểm tiêu chí Xử lý từ chối/thắc mắc (0-20)"),
            ResponseSchema(name="score_closing", description="Điểm tiêu chí Kết thúc, CS sau bán (0-20)"),
            ResponseSchema(name="strengths", description="Ưu điểm cụ thể, mỗi dòng bắt đầu +. Tối đa 5 dòng."),
            ResponseSchema(name="weaknesses", description="Điểm cần cải thiện, mỗi dòng bắt đầu +. Tối đa 5 dòng."),
            ResponseSchema(
                name="keywords",
                description=(
                    "5-8 cụm có nghĩa: tên SP/model khách nhắc, chủ đề (Giá, Size, Vận chuyển, Đổi trả...). "
                    "Cách nhau dấu phẩy. KHÔNG dùng từ đơn lẻ vô nghĩa (ảnh, nhận, gửi, luôn, còn...)."
                ),
            ),
            ResponseSchema(name="sentiment_label", description="Một trong: Tích cực | Trung tính | Cần chú ý"),
            ResponseSchema(name="sentiment_customer", description="Một câu mô tả cảm xúc/phản ứng khách hàng"),
            ResponseSchema(name="sentiment_staff", description="Một câu mô tả thái độ nhân viên"),
            ResponseSchema(name="sentiment_tone", description="positive | neutral | negative"),
            ResponseSchema(
                name="tags",
                description="Tag có căn cứ: VIP, Quan tâm giá, Cần follow-up, Chưa rep... Cách nhau dấu phẩy. Để trống nếu không có.",
            ),
        ]

    def _clamp_criterion(self, value, default=0) -> int:
        try:
            n = int(float(value))
        except (TypeError, ValueError):
            n = default
        return max(0, min(20, n))

    def _enrich_parsed_output(self, parsed_output: dict, no_reply: bool) -> dict:
        if no_reply:
            parsed_output["score"] = 0
            criteria = {k: 0 for k in ("greeting", "needs", "consult", "objection", "closing")}
        else:
            criteria = {
                "greeting": self._clamp_criterion(parsed_output.get("score_greeting")),
                "needs": self._clamp_criterion(parsed_output.get("score_needs")),
                "consult": self._clamp_criterion(parsed_output.get("score_consult")),
                "objection": self._clamp_criterion(parsed_output.get("score_objection")),
                "closing": self._clamp_criterion(parsed_output.get("score_closing")),
            }
            total = sum(criteria.values())
            score = self._clamp_score_total(parsed_output.get("score"), 0)
            if total == 0 and score > 0:
                base = score // 5
                rem = score - base * 5
                keys = list(criteria.keys())
                for i, k in enumerate(keys):
                    criteria[k] = base + (1 if i < rem else 0)
            elif total > 0 and abs(total - score) > 2:
                parsed_output["score"] = min(100, total)
            else:
                parsed_output["score"] = score if score else min(100, total)

        parsed_output["criteria_scores"] = criteria
        return parsed_output

    _ANALYSIS_FIELDS_INSTRUCTION = """
PHÂN TÍCH CHI TIẾT (BẮT BUỘC):
- Chấm 5 tiêu chí (mỗi tiêu chí 0-20, tổng ≈ score):
  score_greeting = Chào hỏi, thiện cảm
  score_needs = Khai thác nhu cầu (mục đích, ngân sách, size/sở thích)
  score_consult = Tư vấn, giải đáp (giá, chất liệu, chính sách)
  score_objection = Xử lý từ chối / thắc mắc
  score_closing = Kết thúc, CS sau bán (CTA, cảm ơn, không kết cụt)
- strengths: chỉ ưu điểm thực tế (mỗi dòng +)
- weaknesses: chỉ điểm cần cải thiện (mỗi dòng +), không trùng strengths
- keywords: cụm từ có nghĩa — ưu tiên TÊN SẢN PHẨM/MODEL khách nhắc, rồi chủ đề (Giá, Size, Màu, Vận chuyển, Đổi trả, Bảo hành...). Mỗi mục 1-4 từ. KHÔNG liệt kê từ đơn lẻ tiếng Việt (ảnh, nhận, hàng, gửi, luôn, khi, còn...)
- sentiment_label / sentiment_customer / sentiment_staff / sentiment_tone: đánh giá cảm xúc thực tế
- tags: VIP (khách quan trọng/lặp lại), Quan tâm giá, Cần follow-up, Chưa rep — chỉ gắn khi có căn cứ
"""

    async def audit_transcript(
        self,
        transcript_data: list,
        no_reply: bool = False,
        agent_name: str = None,
        customer_name: str = None,
    ):
        try:
            transcript_str = self._build_transcript_str(transcript_data)

            has_staff = any(m.get("sender") == "Staff" for m in transcript_data)
            # An toàn: chỉ 0 điểm khi transcript không có tin Staff
            if not has_staff:
                no_reply = True

            response_schemas = self._get_response_schemas()
            output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
            format_instructions = output_parser.get_format_instructions()

            if no_reply:
                # Chỉ khi NV hoàn toàn không rep trong ngày — ép 0 điểm
                prompt = ChatPromptTemplate.from_template(
                    """Bạn là chuyên gia kiểm soát chất lượng CSKH (QC).

BỘ QUY TẮC CÔNG TY:
{rules}

TÌNH HUỐNG: Khách hàng đã gửi tin nhắn NHƯNG NHÂN VIÊN CHƯA PHẢN HỒI.
Đây là vi phạm nghiêm trọng quy tắc I (Không bỏ sót khách) và quy tắc II (Tốc độ phản hồi).

NHIỆM VỤ CỦA BẠN:
1. Đọc toàn bộ nội dung khách hàng gửi (role 'Customer').
2. Tóm tắt khách đang hỏi/yêu cầu gì cụ thể.
3. Đánh giá mức độ khẩn cấp: cần xử lý ngay hay có thể chờ.
4. Chỉ ra các quy tắc bị vi phạm do chưa phản hồi (trong action_items).
5. action_items: mỗi lỗi 1 dòng format "+ [vi phạm] || [mẫu tin NV gửi khách]". Bắt buộc ít nhất 1 mục.
6. feedback: chỉ nhận xét tổng quan, không ghi gợi ý trả lời.
7. Điểm số: 0 (nhân viên hoàn toàn chưa phản hồi = vi phạm tuyệt đối).
{analysis_fields}

LƯU Ý QUAN TRỌNG:
- Nếu transcript trống hoặc chỉ có tin nhắn hệ thống: ghi rõ "Không có nội dung hội thoại từ khách".
- Tên nhân viên (agent_name): nếu biết là {agent_name}, dùng tên đó. Nếu không, để 'Chưa được phân công'.
- Tên khách (customer_name): nếu biết là {customer_name}, dùng tên đó. Nếu không, xác định từ nội dung.

{format_instructions}

TRANSCRIPT (chỉ có tin nhắn từ khách, nhân viên chưa trả lời):
{transcript}
"""
                )

                messages = prompt.format_messages(
                    transcript=transcript_str if transcript_str.strip() else "(Không có nội dung)",
                    rules=self.rules_text,
                    format_instructions=format_instructions,
                    agent_name=agent_name or "Chưa xác định",
                    customer_name=customer_name or "Khách hàng",
                    analysis_fields=self._ANALYSIS_FIELDS_INSTRUCTION,
                )
            else:
                # NV đã rep — chấm dựa trên TOÀN BỘ cuộc hội thoại (khách + NV)
                prompt = ChatPromptTemplate.from_template(
                    """Bạn là một chuyên gia kiểm soát chất lượng dịch vụ khách hàng (QC). 
Nhiệm vụ của bạn là phân tích cuộc hội thoại dưới đây giữa 'Staff' (Nhân viên) và 'Customer' (Khách hàng).

NGỮ CẢNH TRANSCRIPT:
- Transcript gồm TOÀN BỘ tin nhắn từ lúc bắt đầu hội thoại đến HẾT ngày được chấm audit (không chỉ tin trong 1 ngày).
- Hội thoại được đưa vào audit vì CÓ hoạt động trong ngày chấm — nhưng phải đánh giá dựa trên cả lịch sử trước đó.
- KHÔNG cho 0 điểm / không kết luận "NV chưa rep" nếu transcript có tin Staff từ các ngày trước.

BỘ QUY TẮC CÔNG TY:
{rules}

BẮT BUỘC:
1. Đọc và đánh giá TẤT CẢ tin nhắn qua lại giữa Khách hàng và Nhân viên trong transcript.
2. Xác định tên thật của Khách hàng từ lời chào của Nhân viên (vd: "Chào anh Nam", "Dạ chị Lan ơi") hoặc tên Facebook. BẮT BUỘC không trả về 'Khách hàng' nếu transcript có thể suy ra tên.
3. Xác định tên thật của Nhân viên từ chữ ký hoặc lời giới thiệu (vd: "Em là Mai", "ký tên: Minh"). KHÔNG dùng tên Page/Shop làm tên nhân viên.
4. Chấm điểm (0-100) dựa trên thái độ, quy trình CSKH, tốc độ và độ chính xác — xét cả cuộc hội thoại.
5. KHÔNG cho 0 điểm chỉ vì khách nhắn tin cuối cùng trong ngày audit. Transcript này có Staff — phải chấm theo chất lượng tư vấn thực tế.
6. feedback: nhận xét tổng quan (ưu/nhược), mỗi dòng bắt đầu +. KHÔNG ghi gợi ý trả lời ở đây.
7. action_items: mỗi lỗi/vi phạm cụ thể = 1 dòng "+ [lỗi] || [mẫu tin NV gửi khách để sửa lỗi đó]". Gợi ý phải khớp đúng lỗi trên cùng dòng.
{analysis_fields}

NGỮ CẢNH BÁO GIÁ (BẮT BUỘC — tránh chấm sai):
- Quy tắc "không báo giá khi chưa hiểu nhu cầu" áp dụng khi NV CHỦ ĐỘNG đưa giá mà khách chưa hỏi.
- Nếu KHÁCH chủ động hỏi giá ("bao nhiêu", "tính giá", "giá sao", "quote"): NV ĐƯỢC trả lời phân khúc/khoảng giá — KHÔNG ghi vi phạm "báo giá quá sớm" chỉ vì có nhắc số tiền.
- Khách gửi ảnh mẫu + hỏi giá mẫu đó: NV trả lời theo phân khúc là HỢP LÝ.
- Nếu NV vừa nêu khoảng giá VÀ ngay sau đó hỏi ngân sách/mục đích/size/sở thích → coi là ĐÚNG bước khai thác, không trừ điểm mục báo giá sớm (có thể ghi nhận tích cực trong feedback).
- Chỉ ghi "báo giá quá sớm" khi NV đưa giá cụ thể/chốt giá trong khi khách chưa hỏi giá và NV cũng chưa hỏi thêm nhu cầu.

{format_instructions}

TRANSCRIPT (toàn bộ hội thoại đến hết ngày audit — Khách hàng + Nhân viên):
{transcript}
"""
                )

                messages = prompt.format_messages(
                    transcript=transcript_str,
                    rules=self.rules_text,
                    format_instructions=format_instructions,
                    analysis_fields=self._ANALYSIS_FIELDS_INSTRUCTION,
                )

            response = await self.llm.ainvoke(messages)
            token_usage = self._extract_token_usage(response)
            
            try:
                parsed_output = output_parser.parse(response.content)
                parsed_output = self._enrich_parsed_output(parsed_output, no_reply)
                parsed_output["token_usage"] = token_usage
                return parsed_output
            except Exception as e:
                return {
                    "score": 0 if no_reply else 50,
                    "feedback": f"Lỗi phân tích kết quả từ AI: {str(e)}. Nội dung thô: {response.content}",
                    "action_items": "",
                    "suggested_replies": "",
                    "violations": [],
                    "customer_name": customer_name or "Khách hàng",
                    "agent_name": agent_name or "Nhân viên",
                    "token_usage": token_usage,
                }
        except Exception as e:
            print(f"[AuditService] CRITICAL ERROR: {traceback.format_exc()}")
            from src.utils.user_facing_error import to_user_facing_error

            safe = to_user_facing_error(str(e))
            return {
                "score": 0,
                "feedback": f"Không thể phân tích hội thoại: {safe}",
                "action_items": "",
                "suggested_replies": "",
                "violations": [],
                "customer_name": customer_name or "Khách hàng",
                "agent_name": agent_name or "Nhân viên",
                "token_usage": self._empty_token_usage(),
            }
