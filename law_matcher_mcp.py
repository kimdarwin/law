import os
import re
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="EasyLead Law Matcher API", version="1.0.0")

# OpenAI(GPTs)에서 호출할 수 있도록 CORS 허용 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_CSV_PATH = "law2easy.csv"

# [기존 TF-IDF 로직 동일 유지]
def get_similar_pairs(query: str, max_results: int = 10):
    if not os.path.exists(DEFAULT_CSV_PATH):
        return []

    df = pd.read_csv(DEFAULT_CSV_PATH).dropna(how="all")
    headers = list(df.columns)
    
    input_column = headers[1] if len(headers) > 1 else headers[0]
    output_column = headers[2] if len(headers) > 2 else headers[0]

    corpus = df[input_column].astype(str).str.strip().tolist()
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", analyzer="char_wb", ngram_range=(2, 4))
    tfidf_matrix = vectorizer.fit_transform(corpus)

    query_vector = vectorizer.transform([query.strip()])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    related_indices = similarities.argsort()[::-1]

    results = []
    match_count = 0
    for idx in related_indices:
        score = similarities[idx]
        if score <= 0.0 and match_count > 0:
            break
        row = df.iloc[idx]
        results.append({
            "score": float(score),
            "input": re.sub(r"\s+", " ", str(row[input_column])),
            "output": re.sub(r"\s+", " ", str(row[output_column]))
        })
        match_count += 1
        if match_count >= max_results:
            break
    return results

# GPTs가 호출할 엔드포인트 정의
@app.get("/search")
def search_law(query: str = Query(..., description="검색할 판례 원문 문장")):
    results = get_similar_pairs(query)
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)