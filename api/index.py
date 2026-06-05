from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client
from openai import OpenAI
import requests
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://marksmind.in", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

MODEL_ID = "openai/gpt-4o-mini:free"

class AnswerRequest(BaseModel):
    question: str
    student_answer: str
    class_num: int
    subject: str

def get_embedding(text):
    api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(
        api_url,
        headers=headers,
        json={"inputs": text, "options": {"wait_for_model": True}}
    )
    embedding = response.json()
    # HF returns nested list for single input, flatten it
    if isinstance(embedding[0], list):
        embedding = embedding[0]
    return embedding

def search_ncert(question, class_num, subject, top_k=3):
    question_embedding = get_embedding(question)
    result = supabase.rpc("match_chunks", {
        "query_embedding": question_embedding,
        "match_count": top_k,
        "filter_class": class_num,
        "filter_subject": subject
    }).execute()
    return result.data

def search_marking_scheme(question, class_num, top_k=2):
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
    return result.data

def score_answer(question, student_answer, class_num, subject):
    ncert_chunks = search_ncert(question, class_num, subject)
    ncert_context = "\n\n".join([r['content'] for r in ncert_chunks])
    ms_chunks = search_marking_scheme(question, class_num)
    ms_context = "\n\n".join([r['content'] for r in ms_chunks])

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

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

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
