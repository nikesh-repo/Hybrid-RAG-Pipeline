import hashlib
import json
from pathlib import Path
import chromadb
import ollama
from pypdf import PdfReader

DOCS_DIR = "./docs"
CHUNKS_JSON = "./chunks.json"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "local_rag_docs"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma4:e4b"

WORD_CHUNK_SIZE = 80
WORD_OVERLAP = 15


def chunk_words(text: str, chunk_size: int = 80, overlap: int = 15) -> list[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    if not words:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        slice_words = words[start : start + chunk_size]
        chunks.append(" ".join(slice_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def load_documents(folder_path: str = DOCS_DIR) -> list[dict]:
    """Load .txt, .md, and .pdf files from the documents folder."""
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
            pages = [page.extract_text() or "" for page in reader.pages]
            content = "\n".join(pages)
            if content.strip():
                docs.append({"source": file_path.name, "text": content})

    return docs


def save_word_chunks_to_json(
    docs: list[dict],
    json_path: str = CHUNKS_JSON,
    chunk_size: int = WORD_CHUNK_SIZE,
    overlap: int = WORD_OVERLAP,
) -> list[dict]:
    """Generate word-level chunks from documents and write them to a JSON file."""
    records = []

    for doc in docs:
        source = doc.get("source", "unknown")
        chunks = chunk_words(doc.get("text", ""), chunk_size=chunk_size, overlap=overlap)
        for idx, chunk_text in enumerate(chunks):
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(records)} word chunks to {out_path}")
    return records


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Compute embeddings using Ollama."""
    response = ollama.embed(model=EMBED_MODEL, input=texts)
    return response["embeddings"]


def index_chunks_from_json(collection, json_path: str = CHUNKS_JSON) -> None:
    """Index chunks from JSON into ChromaDB without duplicating existing IDs."""
    path = Path(json_path)
    if not path.exists():
        print(f"No JSON file found at {path}")
        return

    records = json.loads(path.read_text(encoding="utf-8"))
    if not records:
        print("JSON file contains no chunks to index.")
        return

    ids = [r["id"] for r in records]
    existing = collection.get(ids=ids)
    existing_ids = set(existing.get("ids", []))

    new_records = [r for r in records if r["id"] not in existing_ids]
    if not new_records:
        print("All chunks in JSON are already indexed in ChromaDB.")
        return

    new_ids = [r["id"] for r in new_records]
    new_texts = [r["text"] for r in new_records]
    new_meta = [
        {
            "source": r["source"],
            "chunk_index": r["chunk_index"],
            "word_count": r["word_count"],
        }
        for r in new_records
    ]

    embeddings = get_embeddings(new_texts)
    collection.add(
        ids=new_ids,
        documents=new_texts,
        metadatas=new_meta,
        embeddings=embeddings,
    )
    print(f"Indexed {len(new_records)} new chunks into ChromaDB.")


def answer_question(collection, question: str, n_results: int = 3) -> str:
    """Retrieve relevant word chunks and generate an answer with gemma4:e4b."""
    query_embedding = get_embeddings([question])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    docs_list = results.get("documents") or [[]]
    metas_list = results.get("metadatas") or [[]]
    retrieved_chunks = docs_list[0]
    retrieved_metas = metas_list[0]

    if not retrieved_chunks:
        return "No relevant context found in the indexed documents."

    formatted_parts = []
    for chunk_text, meta in zip(retrieved_chunks, retrieved_metas):
        src = meta.get("source", "unknown") if meta else "unknown"
        idx = meta.get("chunk_index", "?") if meta else "?"
        formatted_parts.append(f"\n{chunk_text}")

    context_block = "\n\n---\n\n".join(formatted_parts)

    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
        "If the context does not contain enough information, state clearly that you don't know based on the provided context. "
        "Be accurate and concise."
    )

    user_prompt = f"""Context:
{context_block}

Question:
{question}"""

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

    # Fallback sample document if ./docs is still empty on first run
    if not docs:
        docs = [
            {
                "source": "getting_started.txt",
                "text": (
                    "This local RAG pipeline splits documents into word-level chunks, saves them to chunks.json, "
                    "embeds them with nomic-embed-text in Ollama, stores vectors in ChromaDB, and answers questions "
                    "using gemma4:e4b on a Windows 11 laptop with 16 GB of RAM."
                ),
            }
        ]

    save_word_chunks_to_json(
        docs,
        json_path=CHUNKS_JSON,
        chunk_size=WORD_CHUNK_SIZE,
        overlap=WORD_OVERLAP,
    )

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    index_chunks_from_json(collection, json_path=CHUNKS_JSON)

    print("\nRAG is ready! Type a question and press Enter (or type 'exit' to quit).\n")
    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        answer = answer_question(collection, question, n_results=3)
        print(f"\nAssistant:\n{answer}\n")


if __name__ == "__main__":
    main()