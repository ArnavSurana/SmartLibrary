import os as os
import fitz                           # PyMuPDF
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai
import time
from pinecone import Pinecone, ServerlessSpec

# === Load .env ===
load_dotenv()

# === FastAPI app ===
app = FastAPI(title="📚 Gemini Book Bot API")

# === Delete chunks function ===
def delete_chunks(user_id: str, book_name: str) -> bool:
    # Query all chunk ids for the user and book
    results = index.query(
        vector=[0.0] * DIMENSION,
        top_k=1000,
        include_metadata=False,
        filter={"user_id": user_id, "book_name": book_name}
    )
    ids_to_delete = [match.id for match in results.matches]
    if not ids_to_delete:
        return False
    # Delete all chunks by ids
    index.delete(ids=ids_to_delete)
    return True

# Add CORS middleware
origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Gemini config ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing in .env")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("models/gemini-1.5-pro-001")

# === Pinecone config ===
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV     = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_CLOUD   = os.getenv("PINECONE_CLOUD")
PINECONE_REGION  = os.getenv("PINECONE_REGION")
INDEX_NAME       = "book-index"
DIMENSION        = 768

if not all([PINECONE_API_KEY, PINECONE_ENV, PINECONE_CLOUD, PINECONE_REGION]):
    raise RuntimeError("One or more Pinecone vars missing in .env")

# instantiate Pinecone client
pc = Pinecone(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

# create index if needed
existing_indexes = pc.list_indexes().names()
if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        vector_type="dense",
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION
        ),
        deletion_protection="disabled"
    )

# get handle to index
index = pc.Index(INDEX_NAME)

# === Helpers ===

def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def get_embedding(text: str) -> np.ndarray:
    resp = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document",
        title="book content"
    )
    return np.array(resp["embedding"])


def book_already_indexed(user_id: str, book_name: str) -> bool:
    res = index.query(
        vector=[0.0] * DIMENSION,
        top_k=1,
        include_metadata=True,
        filter={"user_id": user_id, "book_name": book_name}
    )
    return len(res.matches) > 0


def ask_gemini(question: str, context: str) -> str:
    prompt = (
        "You are a helpful assistant rewriting book excerpts in a clear, formal tone.\n\n"
        f"User's Question:\n{question}\n\n"
        f"Relevant Excerpt:\n{context}\n\n"
        "➡️ Rephrase to directly answer the user's question."
    )
    return gemini_model.generate_content(prompt).text.strip()

# === Pydantic schemas ===

class AskRequest(BaseModel):
    user_id: str
    book_name: str
    query: str

class AskResponse(BaseModel):
    answer: str

class BooksResponse(BaseModel):
    books: List[str]

# === API endpoints ===

import logging
from fastapi import status

logger = logging.getLogger("uvicorn.error")

# @app.post("/books/", response_model=dict)
# async def upload_book(
#     file: UploadFile = File(...),
#     user_id: str     = Form(...)
# ):
#     try:
#         # save to disk
#         os.makedirs("uploads", exist_ok=True)
#         path = f"uploads/{file.filename}"
#         with open(path, "wb") as f:
#             f.write(await file.read())

#         # skip if already indexed
#         if book_already_indexed(user_id, file.filename):
#             return {"status": f"✅ '{file.filename}' already indexed for {user_id}."}

#         # extract, chunk, embed, upsert
#         text   = extract_text_from_pdf(path)
#         chunks = chunk_text(text)
#         vectors = []
#         for i, chunk in enumerate(chunks):
#             emb = get_embedding(chunk)
#             vectors.append({
#                 "id":       f"{user_id}-{file.filename}-chunk-{i}",
#                 "values":   emb.tolist(),
#                 "metadata": {
#                     "user_id": user_id,
#                     "book_name": file.filename,  # This is correct, file.filename is the book name
#                     "text": chunk
#                 }
#             })
#         index.upsert(vectors=vectors)

#         return {"status": f"📚 Indexed '{file.filename}' for {user_id}."}
#     except Exception as e:
#         logger.error(f"Error in upload_book: {e}", exc_info=True)

@app.post("/books/", response_model=dict)
async def upload_book(file: UploadFile = File(...), user_id: str = Form(...)):
    try:
        os.makedirs("uploads", exist_ok=True)
        path = f"uploads/{file.filename}"
        with open(path, "wb") as f:
            f.write(await file.read())

        if book_already_indexed(user_id, file.filename):
            return {"status": f"✅ '{file.filename}' already indexed for {user_id}."}

        text   = extract_text_from_pdf(path)
        chunks = chunk_text(text, chunk_size=300)
        vectors = []

        for i, chunk in enumerate(chunks):
            print(f"Embedding chunk {i+1}/{len(chunks)}")
            emb = get_embedding(chunk)
            vectors.append({
                "id":       f"{user_id}-{file.filename}-chunk-{i}",
                "values":   emb.tolist(),
                "metadata": {
                    "user_id": user_id,
                    "book_name": file.filename,
                    "text": chunk
                }
            })
            time.sleep(0.3)

        BATCH_SIZE = 50
        for i in range(0, len(vectors), BATCH_SIZE):
            batch = vectors[i:i+BATCH_SIZE]
            index.upsert(vectors=batch)

        return {"status": f"📚 Indexed '{file.filename}' for {user_id}."}
    except Exception as e:
        logger.error(f"Error in upload_book: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error while indexing the book.")
    



@app.get("/books/{user_id}", response_model=BooksResponse)
def list_books(user_id: str):
    res = index.query(
        vector=[0.0] * DIMENSION,
        top_k=100,
        include_metadata=True,
        filter={"user_id": user_id}
    )
    books = {m.metadata["book_name"] for m in res.matches}
    return {"books": sorted(books)}

# Debug endpoint to list indexed chunks for a user and book
from fastapi.responses import JSONResponse

@app.get("/debug/indexed_chunks/{user_id}/{book_name}")
def list_indexed_chunks(user_id: str, book_name: str):
    res = index.query(
        vector=[0.0] * DIMENSION,
        top_k=100,
        include_metadata=True,
        filter={"user_id": user_id, "book_name": book_name}
    )
    chunks = [{"id": m.id, "text": m.metadata.get("text", "")} for m in res.matches]
    return JSONResponse(content={"chunks": chunks})

@app.post("/questions/", response_model=AskResponse)
def ask_question(req: AskRequest):
    # semantic search
    emb = get_embedding(req.query).tolist()
    res = index.query(
        vector=emb,
        top_k=1,
        include_metadata=True,
        filter={"user_id": req.user_id, "book_name": req.book_name}
    )
    # Log query details and number of matches found
    logger.info(f"ask_question called with user_id={req.user_id}, book_name={req.book_name}, query={req.query}")
    logger.info(f"Number of matches found: {len(res.matches)}")
    if not res.matches:
        raise HTTPException(status_code=404, detail="No relevant content found.")
    context = res.matches[0].metadata["text"]
    answer  = ask_gemini(req.query, context)
    return {"answer": answer}

from fastapi import status, HTTPException, Form

@app.post("/books/delete", status_code=status.HTTP_200_OK)
def delete_book(user_id: str = Form(...), book_name: str = Form(...)):
    success = delete_chunks(user_id, book_name)
    if success:
        return {"status": f"✅ Deleted all chunks for book '{book_name}' and user '{user_id}'."}
    else:
        raise HTTPException(status_code=404, detail=f"No chunks found for book '{book_name}' and user '{user_id}'.")
