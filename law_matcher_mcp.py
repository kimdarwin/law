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
    "path": None,   # 캐시 키에 csv_path도 포함 (버그 수정)
}

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================
# CSV 생성 (없을 때만 더미 생성)
# =========================
def create_default_csv_if_missing():
    logger.info(f"DEFAULT_CSV_PATH = {DEFAULT_CSV_PATH}")
    logger.info(f"cwd = {os.getcwd()}")
    logger.info(f"exists? = {os.path.exists(DEFAULT_CSV_PATH)}")
    if not os.path.exists(DEFAULT_CSV_PATH):
        logger.warning("CSV not found — creating dummy CSV.")
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "원문": [
                "이 사건 소를 각하한다.",
                "원고의 청구를 기각한다.",
                "피고인을 징역 1년에 처한다."
            ],
            "이지리드 문장": [
                "소송을 받아주지 않는다.",
                "요구를 받아들이지 않는다.",
                "1년 감옥형"
            ]
        })
        df.to_csv(DEFAULT_CSV_PATH, index=False, encoding="utf-8-sig")
    else:
        logger.info("Real CSV found, skipping dummy creation.")


# =========================
# INDEX (TF-IDF 로직)
# =========================
def get_index(csv_path: str, col: str):
    # 캐시 적중 조건에 csv_path까지 포함 (버그 수정)
    if (
        _cache["df"] is not None
        and _cache["col"] == col
        and _cache["path"] == csv_path
    ):
        return _cache["df"], _cache["vec"], _cache["mat"]

    logger.info(f"Loading csv from: {os.path.abspath(csv_path)}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if col not in df.columns:
        raise ValueError(
            f"'{col}' 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {list(df.columns)}"
        )

    # 전체 dropna() 대신 input_column 기준으로만 결측 제거 (버그 수정)
    df = df.dropna(subset=[col]).reset_index(drop=True)

    logger.info(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

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
        "path": csv_path,
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
    output_column: str = "이지리드 문장",  # 기본값을 실제 컬럼명과 일치시킴 (버그 수정)
    max_results: int = 10,
) -> str:
    if csv_path == DEFAULT_CSV_PATH:
        create_default_csv_if_missing()

    df, vec, mat = get_index(csv_path, input_column)

    if output_column not in df.columns:
        return f"'{output_column}' 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {list(df.columns)}"

    if len(df) == 0:
        return "검색할 데이터가 없습니다 (CSV가 비어있거나 모두 결측치로 제거됨)."

    qv = vec.transform([query])
    scores = cosine_similarity(qv, mat).flatten()

    n = min(max_results, len(df))
    idxs = scores.argsort()[::-1][:n]

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
    mcp.run(transport="sse")
