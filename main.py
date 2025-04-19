# #!/usr/bin/env python3
# import os
# import argparse
# from dotenv import load_dotenv
# import fitz                           # PyMuPDF
# import numpy as np
# from pinecone import Pinecone, ServerlessSpec
# from pinecone.openapi_support.exceptions import PineconeApiException
# import google.generativeai as genai

# # === Step 0: Load .env into os.environ ===
# load_dotenv()

# # === Step 1: Parse command-line arguments ===
# parser = argparse.ArgumentParser(
#     description="📚 Gemini‑Book‑Bot: Ask questions about your PDF book."
# )
# parser.add_argument(
#     "pdf_path",
#     help="Path to the PDF file you want to index and query"
# )
# parser.add_argument(
#     "user_id",
#     help="Unique identifier for the user (e.g., username, email, or UUID)"
# )
# parser.add_argument(
#     "action",
#     nargs='?',
#     default="query",
#     choices=["query", "delete"],
#     help="Action to perform: query (default) or delete"
# )
# parser.add_argument(
#     "--list-books",
#     action="store_true",
#     help="List all books already indexed for the given user"
# )
# args = parser.parse_args()
# PDF_PATH = args.pdf_path
# USER_ID = args.user_id
# BOOK_NAME = os.path.basename(PDF_PATH)
# ACTION = args.action

# # Verify PDF file exists
# if not os.path.isfile(PDF_PATH):
#     raise FileNotFoundError(f"No such file: '{PDF_PATH}'")

# # === Step 2: Configure Gemini API Key ===
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     raise RuntimeError("GEMINI_API_KEY not set in .env")
# genai.configure(api_key=GEMINI_API_KEY)

# # === Step 3: Choose Gemini model ===
# model = genai.GenerativeModel("models/gemini-1.5-pro-001")

# # === Step 4: Initialize Pinecone ===
# PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")
# PINECONE_ENV       = os.getenv("PINECONE_ENVIRONMENT")
# PINECONE_CLOUD     = os.getenv("PINECONE_CLOUD")
# PINECONE_REGION    = os.getenv("PINECONE_REGION")

# if not (PINECONE_API_KEY and PINECONE_ENV and PINECONE_CLOUD and PINECONE_REGION):
#     raise RuntimeError(
#         "PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_CLOUD, and PINECONE_REGION must be set in .env"
#     )

# pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
# index_name = "book-index"
# dimension  = 768

# try:
#     if index_name not in pc.list_indexes():
#         pc.create_index(
#             name=index_name,
#             vector_type="dense",
#             dimension=dimension,
#             metric="cosine",
#             spec=ServerlessSpec(
#                 cloud=PINECONE_CLOUD,
#                 region=PINECONE_REGION
#             ),
#             deletion_protection="disabled"
#         )
# except PineconeApiException as e:
#     if getattr(e, 'status', None) == 409 or (hasattr(e, 'body') and b'ALREADY_EXISTS' in getattr(e, 'body', b'')):
#         print(f"Index '{index_name}' already exists, skipping creation.")
#     else:
#         raise

# index = pc.Index(index_name)

# # === Step 5: Check if this book is already indexed for this user ===
# def book_already_indexed(user_id: str, book_name: str) -> bool:
#     results = index.query(
#         vector=[0.0] * dimension,  # Dummy vector
#         top_k=1,
#         include_metadata=True,
#         filter={"user_id": user_id, "book_name": book_name}
#     )
#     return len(results.get("matches", [])) > 0

# # === Step 6: Extract & chunk text from PDF ===
# def extract_text_from_pdf(pdf_path: str) -> str:
#     doc = fitz.open(pdf_path)
#     return "\n".join(page.get_text() for page in doc)

# def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
#     return [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]

# # === Step 7: Generate embeddings if not already indexed ===
# def get_embedding(text: str) -> np.ndarray:
#     resp = genai.embed_content(
#         model="models/embedding-001",
#         content=text,
#         task_type="retrieval_document",
#         title="book content"
#     )
#     return np.array(resp["embedding"])

# # === Step 8: List books for user ===
# def list_books_for_user(user_id: str) -> list[str]:
#     response = index.query(
#         vector=[0.0] * dimension,
#         top_k=100,  # high enough to get representative samples
#         include_metadata=True,
#         filter={"user_id": user_id}
#     )
#     books = {match["metadata"]["book_name"] for match in response.get("matches", [])}
#     return sorted(books)

# # === Step 9: Handle listing books ===
# if args.list_books:
#     books = list_books_for_user(USER_ID)
#     print(f"\n📚 Books indexed for user '{USER_ID}':")
#     if books:
#         for book in books:
#             print(f" - {book}")
#     else:
#         print("No books indexed yet.")
#     exit(0)

# # === Step 10: Index the book if not already indexed ===
# if not book_already_indexed(USER_ID, BOOK_NAME):
#     print(f"🔍 Indexing new book: '{BOOK_NAME}' for user '{USER_ID}'")
#     pdf_text    = extract_text_from_pdf(PDF_PATH)
#     book_chunks = chunk_text(pdf_text)

#     vectors = []
#     for i, chunk in enumerate(book_chunks):
#         emb = get_embedding(chunk)
#         vectors.append({
#             "id":       f"{USER_ID}-{BOOK_NAME}-chunk-{i}",
#             "values":   emb.tolist(),
#             "metadata": {
#                 "user_id": USER_ID,
#                 "book_name": BOOK_NAME,
#                 "text": chunk
#             }
#         })

#     index.upsert(vectors)
# else:
#     print(f"✅ Book '{BOOK_NAME}' already indexed for user '{USER_ID}'. Skipping re-indexing.")

# # === Step 11: Q&A helpers ===
# def query_pinecone(query: str, top_k: int = 1) -> list[dict]:
#     q_emb   = get_embedding(query)
#     results = index.query(
#         vector=q_emb.tolist(),
#         top_k=top_k,
#         include_metadata=True,
#         filter={"user_id": USER_ID, "book_name": BOOK_NAME}
#     )
#     return results.get("matches", [])

# def delete_chunks(user_id: str, book_name: str) -> bool:
#     # Query all chunk ids for the user and book
#     results = index.query(
#         vector=[0.0] * dimension,
#         top_k=1000,
#         include_metadata=False,
#         filter={"user_id": user_id, "book_name": book_name}
#     )
#     ids_to_delete = [match["id"] for match in results.get("matches", [])]
#     if not ids_to_delete:
#         return False
#     # Delete all chunks by ids
#     index.delete(ids=ids_to_delete)
#     return True

# def ask_gemini(_query: str, context: str) -> str:
#     prompt = (
#     "You are a helpful assistant tasked with rewriting book excerpts in a clearer and more formal tone.\n\n"
#     f"**User's Question:**\n{_query}\n\n"
#     f"**Relevant Excerpt from the Book:**\n{context}\n\n"
#     "➡️ Please rephrase the excerpt to directly address the user's question, ensuring clarity, coherence, and a formal writing style."
# )

#     response = model.generate_content(prompt)
#     return response.text

# # === Step 12: Interactive loop ===
# if __name__ == "__main__":
#     if ACTION == "delete":
#         success = delete_chunks(USER_ID, BOOK_NAME)
#         if success:
#             print(f"✅ Deleted all chunks for book '{BOOK_NAME}' and user '{USER_ID}'.")
#         else:
#             print(f"⚠️ No chunks found for book '{BOOK_NAME}' and user '{USER_ID}'.")
#         exit(0)

#     print(f"\n📚 Ready! Book: '{BOOK_NAME}' for user: '{USER_ID}'\nEnter a keyword or phrase to query (type 'exit' to quit).")
#     while True:
#         q = input("\n> ").strip()
#         if q.lower() == "exit":
#             break

#         matches = query_pinecone(q)
#         if not matches:
#             print("No relevant content found.")
#             continue

#         best = matches[0]["metadata"]["text"]
#         paraphrase = ask_gemini(q, best)
#         print("\n🖋️ Gemini’s Paraphrase:\n", paraphrase)



#!/usr/bin/env python3
import os
import time
import argparse
from dotenv import load_dotenv
import fitz
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from pinecone.openapi_support.exceptions import PineconeApiException
import google.generativeai as genai

load_dotenv()

parser = argparse.ArgumentParser(description="📚 Gemini‑Book‑Bot: Ask questions about your PDF book.")
parser.add_argument("pdf_path", help="Path to the PDF file you want to index and query")
parser.add_argument("user_id", help="Unique identifier for the user")
parser.add_argument("action", nargs='?', default="query", choices=["query", "delete"], help="Action to perform")
parser.add_argument("--list-books", action="store_true", help="List all books already indexed for the given user")
args = parser.parse_args()

PDF_PATH = args.pdf_path
USER_ID = args.user_id
BOOK_NAME = os.path.basename(PDF_PATH)
ACTION = args.action

if not os.path.isfile(PDF_PATH):
    raise FileNotFoundError(f"No such file: '{PDF_PATH}'")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-pro-001")

PINECONE_API_KEY   = os.getenv("PINECONE_API_KEY")
PINECONE_ENV       = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_CLOUD     = os.getenv("PINECONE_CLOUD")
PINECONE_REGION    = os.getenv("PINECONE_REGION")

if not (PINECONE_API_KEY and PINECONE_ENV and PINECONE_CLOUD and PINECONE_REGION):
    raise RuntimeError("PINECONE settings missing in .env")

pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index_name = "book-index"
dimension  = 768

try:
    if index_name not in pc.list_indexes():
        pc.create_index(
            name=index_name,
            vector_type="dense",
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
            deletion_protection="disabled"
        )
except PineconeApiException as e:
    if getattr(e, 'status', None) == 409:
        print(f"Index '{index_name}' already exists.")
    else:
        raise

index = pc.Index(index_name)

def book_already_indexed(user_id: str, book_name: str) -> bool:
    results = index.query(
        vector=[0.0] * dimension,
        top_k=1,
        include_metadata=True,
        filter={"user_id": user_id, "book_name": book_name}
    )
    return len(results.get("matches", [])) > 0

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text: str, chunk_size: int = 300) -> list[str]:
    return [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]

def get_embedding(text: str) -> np.ndarray:
    resp = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document",
        title="book content"
    )
    return np.array(resp["embedding"])

def list_books_for_user(user_id: str) -> list[str]:
    response = index.query(
        vector=[0.0] * dimension,
        top_k=100,
        include_metadata=True,
        filter={"user_id": user_id}
    )
    books = {match["metadata"]["book_name"] for match in response.get("matches", [])}
    return sorted(books)

if args.list_books:
    books = list_books_for_user(USER_ID)
    print(f"\n📚 Books indexed for user '{USER_ID}':")
    if books:
        for book in books:
            print(f" - {book}")
    else:
        print("No books indexed yet.")
    exit(0)

if not book_already_indexed(USER_ID, BOOK_NAME):
    print(f"🔍 Indexing new book: '{BOOK_NAME}' for user '{USER_ID}'")
    pdf_text    = extract_text_from_pdf(PDF_PATH)
    book_chunks = chunk_text(pdf_text)

    vectors = []
    for i, chunk in enumerate(book_chunks):
        print(f"Embedding chunk {i+1}/{len(book_chunks)}")
        emb = get_embedding(chunk)
        vectors.append({
            "id":       f"{USER_ID}-{BOOK_NAME}-chunk-{i}",
            "values":   emb.tolist(),
            "metadata": {
                "user_id": USER_ID,
                "book_name": BOOK_NAME,
                "text": chunk
            }
        })
        time.sleep(0.3)

    BATCH_SIZE = 50
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i+BATCH_SIZE]
        index.upsert(vectors=batch)
else:
    print(f"✅ Book '{BOOK_NAME}' already indexed for user '{USER_ID}'.")

def query_pinecone(query: str, top_k: int = 1) -> list[dict]:
    q_emb = get_embedding(query)
    results = index.query(
        vector=q_emb.tolist(),
        top_k=top_k,
        include_metadata=True,
        filter={"user_id": USER_ID, "book_name": BOOK_NAME}
    )
    return results.get("matches", [])

def delete_chunks(user_id: str, book_name: str) -> bool:
    results = index.query(
        vector=[0.0] * dimension,
        top_k=1000,
        include_metadata=False,
        filter={"user_id": user_id, "book_name": book_name}
    )
    ids_to_delete = [match["id"] for match in results.get("matches", [])]
    if not ids_to_delete:
        return False
    index.delete(ids=ids_to_delete)
    return True

def ask_gemini(_query: str, context: str) -> str:
    prompt = (
        "You are a helpful assistant tasked with rewriting book excerpts in a clearer and more formal tone.\n\n"
        f"*User's Question:*\n{_query}\n\n"
        f"*Relevant Excerpt from the Book:*\n{context}\n\n"
        "➡ Please rephrase the excerpt to directly address the user's question."
    )
    response = model.generate_content(prompt)
    return response.text

if __name__ == "__main__":
    if ACTION == "delete":
        success = delete_chunks(USER_ID, BOOK_NAME)
        if success:
            print(f"✅ Deleted all chunks for book '{BOOK_NAME}' and user '{USER_ID}'.")
        else:
            print(f"⚠ No chunks found for book '{BOOK_NAME}' and user '{USER_ID}'.")
        exit(0)

    print(f"\n📚 Ready! Book: '{BOOK_NAME}' for user: '{USER_ID}'\nEnter a keyword or phrase to query (type 'exit' to quit).")
    while True:
        q = input("\n> ").strip()
        if q.lower() == "exit":
            break

        matches = query_pinecone(q)
        if not matches:
            print("No relevant content found.")
            continue

        best = matches[0]["metadata"]["text"]
        paraphrase = ask_gemini(q, best)
        print("\n🖋 Gemini’s Paraphrase:\n", paraphrase)
