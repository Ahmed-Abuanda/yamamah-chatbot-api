import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Union

# ---- Config ----
model_name = "BAAI/bge-m3"  # IMPORTANT: must match the model used to build the FAISS index
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# If RAM is tight or you're on CPU, consider a smaller model:
# model_name = "BAAI/bge-small-en-v1.5"  # 384-dim
# model_name = "BAAI/bge-base-en-v1.5"   # 768-dim

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval().to(device)

def embed_text(texts: Union[str, List[str]], batch_size: int = 8, max_length: int = 512) -> np.ndarray:
    """Embed a list of texts into L2-normalized float32 vectors."""
    # Handle single string input
    if isinstance(texts, str):
        texts = [texts]
    
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            # CLS pooling (what you had). If you prefer mean pooling, use: outputs.last_hidden_state.mean(dim=1)
            outputs = model(**inputs).last_hidden_state
            cls = outputs[:, 0]
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            vecs.append(cls.cpu())
    emb = torch.cat(vecs, dim=0).numpy().astype("float32", copy=False)
    return emb

# FastAPI app
app = FastAPI(
    title="Text Embedding API",
    version="1.0.0",
    description="API for generating text embeddings using BGE-M3 model"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic models
class EmbedTextRequest(BaseModel):
    text: Union[str, List[str]]
    batch_size: int = 8
    max_length: int = 512

class EmbedTextResponse(BaseModel):
    embeddings: List[List[float]]
    model_name: str
    text_count: int
    embedding_dimension: int

@app.post("/embed_text", response_model=EmbedTextResponse)
async def embed_text_endpoint(request: EmbedTextRequest):
    """
    Generate embeddings for input text(s)
    
    Args:
        request: Contains text (string or list of strings), batch_size, and max_length
        
    Returns:
        EmbedTextResponse with embeddings and metadata
    """
    try:
        # Generate embeddings
        embeddings = embed_text(
            texts=request.text,
            batch_size=request.batch_size,
            max_length=request.max_length
        )
        
        # Convert numpy array to list of lists
        embeddings_list = embeddings.tolist()
        
        return EmbedTextResponse(
            embeddings=embeddings_list,
            model_name=model_name,
            text_count=len(embeddings_list),
            embedding_dimension=len(embeddings_list[0]) if embeddings_list else 0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Text Embedding API",
        "model": model_name,
        "device": device,
        "documentation": "/docs",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": device
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)