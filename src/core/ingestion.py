import fitz  # PyMuPDF
import os


def extract_pdf_content(pdf_path: str) -> str:
    """Trích xuất toàn bộ nội dung text từ file PDF."""
    doc = fitz.open(pdf_path)
    full_text = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        full_text += f"\n--- TRANG {page_num + 1} ---\n"
        full_text += page.get_text()

    return full_text


if __name__ == "__main__":
    docs_path = os.getenv("DOCS_PATH", "data/docs")
    for filename in os.listdir(docs_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(docs_path, filename)
            output_path = os.path.join(docs_path, filename.replace(".pdf", "_extracted.txt"))
            text = extract_pdf_content(pdf_path)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"✅ Đã trích xuất: {filename}")
