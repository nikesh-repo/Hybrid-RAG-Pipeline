import hashlib
import json
import os
from pathlib import Path
import ollama
from pypdf import PdfReader
from supabase import Client, create_client
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = "./docs"
CHUNKS_JSON = "./chunks.json"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma4:e4b"

WORD_CHUNK_SIZE = 90
WORD_OVERLAP = 30


def get_supabase() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError(
            "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in environment variables."
        )
    return create_client(url, key)


def chunk_words(text: str, chunk_size: int = WORD_CHUNK_SIZE, overlap: int = WORD_OVERLAP) -> list[str]:
    words = text.split()
    if not words:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return chunks


def load_documents(folder_path: str = DOCS_DIR) -> list[dict]:
    docs = []
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)

    for file_path in sorted(folder.iterdir()):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        if suffix in {".txt", ".md"}:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            if content.strip():
                docs.append({"source": file_path.name, "text": content})
        elif suffix == ".pdf":
            reader = PdfReader(str(file_path))
            content = "\n".join(page.extract_text() or "" for page in reader.pages)
            if content.strip():
                docs.append({"source": file_path.name, "text": content})

    return docs


def save_word_chunks_to_json(docs: list[dict], json_path: str = CHUNKS_JSON) -> list[dict]:
    records = []
    for doc in docs:
        source = doc.get("source", "unknown")
        for idx, chunk_text in enumerate(chunk_words(doc.get("text", ""))):
            chunk_id = hashlib.md5(f"{source}:{idx}:{chunk_text}".encode("utf-8")).hexdigest()
            records.append(
                {
                    "id": chunk_id,
                    "source": source,
                    "chunk_index": idx,
                    "word_count": len(chunk_text.split()),
                    "text": chunk_text,
                }
            )

    out_path = Path(json_path)
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(records)} word chunks to {out_path}")
    return records


def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = ollama.embed(model=EMBED_MODEL, input=texts)
    return response["embeddings"]


def index_chunks_to_supabase(supabase: Client, json_path: str = CHUNKS_JSON, batch_size: int = 50) -> None:
    records = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if not records:
        print("No chunks found in chunks.json.")
        return

    all_ids = [r["id"] for r in records]
    existing = supabase.table("rag_chunks").select("id").in_("id", all_ids).execute()
    existing_ids = {row["id"] for row in (existing.data or [])}

    new_records = [r for r in records if r["id"] not in existing_ids]
    if not new_records:
        print("All chunks are already indexed in Supabase.")
        return

    for i in range(0, len(new_records), batch_size):
        batch = new_records[i : i + batch_size]
        vectors = get_embeddings([r["text"] for r in batch])
        rows = [
            {
                "id": r["id"],
                "source": r["source"],
                "chunk_index": r["chunk_index"],
                "word_count": r["word_count"],
                "content": r["text"],
                "embedding": vec,
            }
            for r, vec in zip(batch, vectors)
        ]
        supabase.table("rag_chunks").upsert(rows).execute()

    print(f"Indexed {len(new_records)} new chunks into Supabase.")


def answer_question(supabase: Client, question: str, match_count: int = 3) -> str:
    query_vec = get_embeddings([question])[0]
    rpc_resp = supabase.rpc(
        "match_rag_chunks",
        {"query_embedding": query_vec, "match_count": match_count},
    ).execute()

    matches = rpc_resp.data or []
    if not matches:
        return "No relevant context found in Supabase."

    context_block = "\n\n---\n\n".join(
        f"[{m['source']} | chunk {m['chunk_index']} | sim={m['similarity']:.2f}]\n{m['content']}"
        for m in matches
    )

    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
        "If the context does not contain the answer, say you don't know based on the provided context."
    )
    user_prompt = f"Context:\n{context_block}\n\nQuestion:\n{question}"

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.message.content


def main():
    docs = load_documents(DOCS_DIR)
    if not docs:
        print("Put at least one .txt, .md, or .pdf file in ./docs and run again.")
        return

    save_word_chunks_to_json(docs, CHUNKS_JSON)
    supabase = get_supabase()
    index_chunks_to_supabase(supabase, CHUNKS_JSON)

    print("\nHybrid Supabase RAG is ready! Ask a question (or type 'exit' to quit).\n")
    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        print(f"\nAssistant:\n{answer_question(supabase, question)}\n")


if __name__ == "__main__":
    main()