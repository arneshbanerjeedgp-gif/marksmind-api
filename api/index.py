from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client
from openai import OpenAI
from fastembed import TextEmbedding
import os
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Using the ultra-fast Llama 3 model to avoid traffic jams
MODEL_ID = "google/gemma-4-31b-it:free"

# Load embedding model once at startup
embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")

class AnswerRequest(BaseModel):
    question: str
    student_answer: str
    class_num: int
    subject: str

def get_embedding(text):
    embeddings = list(embedding_model.embed([text]))
    return embeddings[0].tolist()

def clean_context_retrieval(retrieved_chunks):
    """
    Analyzes and filters out database chunks that are heavily weighted 
    with quiz questions, options, or blank fills rather than narrative/lesson content.
    """
    valid_chunks = []
    
    for chunk in retrieved_chunks:
        content = chunk.get("content", "")
        if not content:
            continue
            
        blank_lines = content.count("_____")
        option_markers = sum([content.count(f"({char})") for char in ['i', 'ii', 'iii', 'iv', 'a', 'b', 'c', 'd']])
        mcq_choices = sum([content.count(f"{char}. ") for char in ['A', 'B', 'C', 'D']])
        
        if blank_lines > 2 or (option_markers + mcq_choices) > 3:
            continue
            
        valid_chunks.append(chunk)
        
    return valid_chunks

def search_ncert(question, class_num, subject, top_k=5):
    question_embedding = get_embedding(question)
    result = supabase.rpc("match_chunks", {
        "query_embedding": question_embedding,
        "match_count": top_k,
        "filter_class": class_num,
        "filter_subject": subject
    }).execute()
    
    return clean_context_retrieval(result.data)

def search_marking_scheme(question, class_num, top_k=4):
    question_embedding = get_embedding(question)
    if class_num in [11, 12]:
        table_function = "match_marking_senior"
    else:
        table_function = "match_marking_secondary"
    result = supabase.rpc(table_function, {
        "query_embedding": question_embedding,
        "match_count": top_k,
        "filter_class": class_num
    }).execute()
    
    return clean_context_retrieval(result.data)

def score_answer(question, student_answer, class_num, subject):
    ncert_chunks = search_ncert(question, class_num, subject)
    # Reduced to 1 chunk for maximum speed
    ncert_context = "\n\n".join([r['content'] for r in ncert_chunks[:1]])
    
    ms_chunks = search_marking_scheme(question, class_num)
    ms_context = "\n\n".join([r['content'] for r in ms_chunks[:1]])

    prompt = f"""You are a strict CBSE examiner for Class {class_num} {subject}.

NCERT Textbook Content:
{ncert_context}

Official CBSE Marking Scheme:
{ms_context}

Question: {question}

Student Answer: {student_answer}

Score this answer exactly like a CBSE examiner. Respond in this exact format:
MARKS: X out of 5
WHAT YOU GOT RIGHT: (one line)
MISSING KEYWORDS: (exact words CBSE expects)
MODEL ANSWER: (perfect answer)
ADVICE: (one line on what to fix)"""

    # 3-Attempt Retry Loop to fight OpenRouter Rate Limits
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                timeout=15 
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️ OpenRouter Error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                print("Retrying in 2 seconds...")
                time.sleep(2) 
            else:
                return "MARKS: 0 out of 5\nWHAT YOU GOT RIGHT: N/A\nMISSING KEYWORDS: N/A\nMODEL ANSWER: The AI grading server is currently overloaded with free-tier traffic. Please wait a moment and click Verify again.\nADVICE: Server busy. Please try again later."

@app.get("/")
def root():
    return {"status": "MarksMind API is live"}

@app.post("/score")
def score(request: AnswerRequest):
    result = score_answer(
        request.question,
        request.student_answer,
        request.class_num,
        request.subject
    )
    return {"result": result}
