import os
import pandas as pd
import re

from mcp.server.fastmcp import FastMCP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# MCP SERVER
# =========================
mcp = FastMCP("EasyLead Law Matcher")

DEFAULT_CSV_PATH = "law2easy.csv"

_cache = {
    "df": None,
    "vec": None,
    "mat": None,
    "col": None,
}


# =========================
# CSV 생성
# =========================
def create_default_csv_if_missing():
    if not os.path.exists(DEFAULT_CSV_PATH):
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "원문": [
                "이 사건 소를 각하한다.",
                "원고의 청구를 기각한다.",
                "피고인을 징역 1년에 처한다."
            ],
            "이지리드": [
                "소송을 받아주지 않는다.",
                "요구를 받아들이지 않는다.",
                "1년 감옥형"
            ]
        })
        df.to_csv(DEFAULT_CSV_PATH, index=False, encoding="utf-8-sig")


# =========================
# INDEX
# =========================
def get_index(csv_path, col):
    if (
        _cache["df"] is not None
        and _cache["col"] == col
    ):
        return _cache["df"], _cache["vec"], _cache["mat"]

    df = pd.read_csv(csv_path, encoding="utf-8-sig").dropna()

    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4)
    )

    mat = vec.fit_transform(df[col].astype(str).tolist())

    _cache.update({
        "df": df,
        "vec": vec,
        "mat": mat,
        "col": col,
    })

    return df, vec, mat


# =========================
# TOOL
# =========================
@mcp.tool()
def get_similar_pairs(
    query: str,
    csv_path: str = DEFAULT_CSV_PATH,
    input_column: str = "원문",
    output_column: str = "이지리드",
    max_results: int = 10,
) -> str:

    if csv_path == DEFAULT_CSV_PATH:
        create_default_csv_if_missing()

    df, vec, mat = get_index(csv_path, input_column)

    qv = vec.transform([query])
    scores = cosine_similarity(qv, mat).flatten()

    idxs = scores.argsort()[::-1][:max_results]

    results = []

    for i in idxs:
        row = df.iloc[i]

        src = re.sub(r"\s+", " ", str(row[input_column]))
        tgt = re.sub(r"\s+", " ", str(row[output_column]))

        results.append(f"입력: {src}\n출력: {tgt}")

    return "\n\n".join(results)


# =========================
# RUN (핵심: /sse 유지)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    # 🔥 FastMCP가 SSE endpoint (/sse)을 자동 생성
    """
    mcp.run(
        transport="sse",
        host="0.0.0.0",
        port=port
    )
    """
    mcp.run(transport="sse")