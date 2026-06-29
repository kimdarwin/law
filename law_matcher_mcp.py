import os
import re
import pandas as pd
import uvicorn
from mcp.server.fastmcp import FastMCP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. FastMCP 인스턴스 초기화
mcp = FastMCP("EasyLead Law Matcher")

# 전역 캐싱 변수
_cached_csv_path = None
_cached_df = None
_vectorizer = None
_tfidf_matrix = None
_last_col_in = None

DEFAULT_CSV_PATH = "law2easy.csv"


def create_default_csv_if_missing():
    """기본 데이터 파일(law2easy.csv)이 존재하지 않는 경우 메모리 상에서 샘플 생성"""
    if not os.path.exists(DEFAULT_CSV_PATH):
        default_data = {
            "id": [1, 2, 3, 4, 5],
            "원문": [
                "이 사건 소를 각하한다.",
                "원고의 청구를 기각한다.",
                "피고인을 징역 1년에 처한다.",
                "피고인을 벌금 000만 원에 처한다.",
                "공무집행방해죄에 해당한다."
            ],
            "이지리드": [
                "이 소송은 받아주지 않는다.",
                "원고가 원하는 대로 해줄 수 없다.",
                "김이박은 1년 동안 감옥에 갇혀 있어야 한다.",
                "김이박은 벌금 000만 원을 내야 한다.",
                "경찰이 일을 제대로 할 수 없게 방해했다."
            ]
        }
        pd.DataFrame(default_data).to_csv(DEFAULT_CSV_PATH, index=False, encoding="utf-8-sig")


def get_or_build_index(csv_path: str, col_in: str):
    """지정된 CSV 파일과 입력 컬럼을 기준으로 TF-IDF 인덱스를 캐싱 및 빌드"""
    global _cached_csv_path, _cached_df, _vectorizer, _tfidf_matrix, _last_col_in

    if (
        _cached_csv_path == csv_path
        and _cached_df is not None
        and _vectorizer is not None
        and _last_col_in == col_in
    ):
        return _cached_df, _vectorizer, _tfidf_matrix

    if not os.path.exists(csv_path):
        if csv_path == DEFAULT_CSV_PATH:
            create_default_csv_if_missing()
        else:
            raise FileNotFoundError(f"지정한 CSV 파일을 찾을 수 없습니다: {csv_path}")

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig").dropna(how="all")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="cp949").dropna(how="all")

    if col_in not in df.columns:
        raise ValueError(f"CSV에 '{col_in}' 컬럼이 존재하지 않습니다. 존재 컬럼: {list(df.columns)}")

    corpus = df[col_in].astype(str).str.strip().tolist()
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        analyzer="char_wb",
        ngram_range=(2, 4)
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    _cached_csv_path = csv_path
    _cached_df = df
    _vectorizer = vectorizer
    _tfidf_matrix = tfidf_matrix
    _last_col_in = col_in

    return df, vectorizer, tfidf_matrix


# 2. MCP 도구 정의
@mcp.tool()
def get_similar_pairs(
    query: str,
    csv_path: str = DEFAULT_CSV_PATH,
    input_column: str = None,
    output_column: str = None,
    max_results: int = 10
) -> str:
    """
    어려운 판례나 법률 문장(query)을 입력하면, 데이터셋(CSV)에서 TF-IDF 기술을 사용해 
    가장 의미적으로 유사한 상위 10개의 번역 대조쌍(Few-shot 예시)을 추출하여 반환합니다.
    """
    try:
        if csv_path == DEFAULT_CSV_PATH:
            create_default_csv_if_missing()

        if not os.path.exists(csv_path):
            return f"오류: '{csv_path}' 경로에 파일이 존재하지 않습니다."

        try:
            temp_df = pd.read_csv(csv_path, encoding="utf-8-sig", nrows=1)
        except UnicodeDecodeError:
            temp_df = pd.read_csv(csv_path, encoding="cp949", nrows=1)

        headers = list(temp_df.columns)

        if not input_column:
            input_column = headers[1] if len(headers) > 1 else (headers[0] if headers else "")
        if not output_column:
            output_column = headers[2] if len(headers) > 2 else (headers[0] if headers else "")

        df, vectorizer, tfidf_matrix = get_or_build_index(csv_path, input_column)

        query_vector = vectorizer.transform([query.strip()])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        related_indices = similarities.argsort()[::-1]

        few_shot_results = []
        match_count = 0

        for idx in related_indices:
            score = similarities[idx]
            if score <= 0.0 and match_count > 0:
                break

            row = df.iloc[idx]
            in_val = str(row[input_column]).strip()
            out_val = str(row[output_column]).strip()

            clean_in = re.sub(r"\s+", " ", in_val)
            clean_out = re.sub(r"\s+", " ", out_val)

            few_shot_results.append(
                f"예시입력: {clean_in}\n출력: {clean_out}"
            )

            match_count += 1
            if match_count >= max_results:
                break

        if not few_shot_results:
            return "일치하거나 유사한 판례 문장쌍 예시를 데이터에서 찾지 못했습니다."

        header_info = f"[매칭 정보: {input_column} -> {output_column}]\n"
        return header_info + "\n\n".join(few_shot_results)

    except Exception as e:
        return f"작업 중 예외 발생: {str(e)}"


# 3. Render.com 호환 프록시 헤더 설정 및 SSE 서버 스타트업
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    
    # Render.com 호스트 검증 에러(421/Request validation failed)를 해소하는 중요 네트워크 옵션 주입
    uvicorn.run(
        mcp.app, 
        host="0.0.0.0", 
        port=port,
        proxy_headers=True,           # 프록시 도메인 헤더(onrender.com) 신뢰 설정
        forwarded_allow_ips="*"       # 모든 포워딩 사설 IP 대역 통과 허용
    )