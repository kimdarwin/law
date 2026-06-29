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
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# CSV 생성
# =========================
def create_default_csv_if_missing():
    logger.info(f"DEFAULT_CSV_PATH = {DEFAULT_CSV_PATH}")
    logger.info(f"cwd = {os.getcwd()}")
    logger.info(f"exists? = {os.path.exists(DEFAULT_CSV_PATH)}")
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
# INDEX (요청하신 TF-IDF 로직 적용)
# =========================
def get_index(csv_path, col):
    if (
        _cache["df"] is not None
        and _cache["col"] == col
    ):
        return _cache["df"], _cache["vec"], _cache["mat"]

    df = pd.read_csv(csv_path, encoding="utf-8-sig").dropna()

    # TF-IDF 분절 및 피팅 진행 (어절 및 캐릭터 N-Gram 분석)
    corpus = df[col].astype(str).str.strip().tolist()
    vec = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        analyzer="char_wb",
        ngram_range=(2, 4)
    )
    mat = vec.fit_transform(corpus)

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
    # Render 등 외부 환경에서 제공하는 PORT 변수를 활용할 수 있도록 설정
    mcp.run(transport="sse")