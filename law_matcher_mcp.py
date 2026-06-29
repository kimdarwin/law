import os
import re
import logging

import pandas as pd
from mcp.server.fastmcp import FastMCP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# MCP SERVER
# =========================
PORT = int(os.environ.get("PORT", 8000))

mcp = FastMCP(
    "EasyLead Law Matcher",
    host="0.0.0.0",
    port=PORT,
)

DEFAULT_CSV_PATH = "law2easy.csv"

_cache = {
    "df": None,
    "vec": None,
    "mat": None,
    "col": None,
    "output_col": None,
    "path": None,
}


# =========================
# CSV 생성 (없을 때만 더미 생성)
# =========================
def create_default_csv_if_missing():
    logger.info(f"DEFAULT_CSV_PATH = {DEFAULT_CSV_PATH}")
    logger.info(f"cwd = {os.getcwd()}")
    exists = os.path.exists(DEFAULT_CSV_PATH)
    logger.info(f"exists? = {exists}")

    if not exists:
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
# INDEX (TF-IDF 로직 및 필터링)
# =========================
def get_index(csv_path: str, input_col: str, output_col: str):
    # 캐시 키에 output_col 정보도 함께 검증하여 변경 시 대응하도록 합니다.
    if (
        _cache["df"] is not None
        and _cache["col"] == input_col
        and _cache.get("output_col") == output_col
        and _cache["path"] == csv_path
    ):
        return _cache["df"], _cache["vec"], _cache["mat"]

    logger.info(f"Loading csv from: {os.path.abspath(csv_path)}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 필수 컬럼 검증
    if input_col not in df.columns:
        raise ValueError(
            f"입력 컬럼 '{input_col}'을 찾을 수 없습니다. 사용 가능한 컬럼: {list(df.columns)}"
        )
    if output_col not in df.columns:
        raise ValueError(
            f"출력 컬럼 '{output_col}'을 찾을 수 없습니다. 사용 가능한 컬럼: {list(df.columns)}"
        )

    # 1. 두 컬럼에 대해 결측치(NaN)가 있는 행 제거
    df = df.dropna(subset=[input_col, output_col])

    # 2. 공백 제거 후 빈 문자열("") 상태인 행 제거 (필터링 핵심 부분)
    df = df[df[input_col].astype(str).str.strip() != ""]
    df = df[df[output_col].astype(str).str.strip() != ""].reset_index(drop=True)

    logger.info(f"Loaded {len(df)} valid rows after filtering empty values. Columns: {list(df.columns)}")

    corpus = df[input_col].astype(str).str.strip().tolist()
    vec = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        analyzer="char_wb",
        ngram_range=(2, 4),
    )
    mat = vec.fit_transform(corpus)

    _cache.update({
        "df": df,
        "vec": vec,
        "mat": mat,
        "col": input_col,
        "output_col": output_col,
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
    output_column: str = "이지리드 문장",
    max_results: int = 30,
) -> str:
    """TF-IDF 기반 법률 유사문장 검색 (비어있는 대상 문장은 검색 대상에서 자동 제외됩니다)"""
    if csv_path == DEFAULT_CSV_PATH:
        create_default_csv_if_missing()

    try:
        # 입력 컬럼과 출력 컬럼을 모두 인자로 전달하여 색인 시점에 필터링을 적용합니다.
        df, vec, mat = get_index(csv_path, input_column, output_column)
    except ValueError as e:
        return str(e)

    if len(df) == 0:
        return "검색할 데이터가 없습니다 (CSV가 비어있거나 필터링 후 남은 행이 없음)."

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
# RUN
# =========================
if __name__ == "__main__":
    mcp.run(transport="streamable-http")