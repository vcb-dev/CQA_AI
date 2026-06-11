# CQA CRM Assistant - AI Service

## Mô tả
Trợ lý AI cho hệ thống CQA CRM, hỗ trợ người dùng trong việc quản lý khách hàng và nghiệp vụ.

## Endpoints chính
- `POST /assistant/chat` - Chat với trợ lý AI

## Tích hợp
- Yêu cầu header `X-Assistant-Secret` để xác thực từ BE
- Trả về response theo chuẩn `AssistantChatResponse`
