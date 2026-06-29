import os
import re
import pandas as pd
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
    """테스트용 기본 데이터셋(law2easy.csv) 파일이 없을 경우 자동으로 생성합니다."""
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
    """지정된 CSV 파일과 입력 컬럼을 기준으로 TF-IDF 인덱스를 캐싱 및 빌드합니다."""
    global _cached_csv_path, _cached_df, _vectorizer, _tfidf_matrix, _last_col_in

    # 캐시 조건 충족 시 재사용
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

    # 인코딩 호환성을 고려하여 파일 읽기
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig").dropna(how="all")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="cp949").dropna(how="all")

    if col_in not in df.columns:
        raise ValueError(f"CSV에 '{col_in}' 컬럼이 존재하지 않습니다. 존재 컬럼: {list(df.columns)}")

    # TF-IDF 분절 및 피팅 진행 (어절 및 캐릭터 N-Gram 분석)
    corpus = df[col_in].astype(str).str.strip().tolist()
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        analyzer="char_wb",
        ngram_range=(2, 4)
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # 메모리 캐시 저장
    _cached_csv_path = csv_path
    _cached_df = df
    _vectorizer = vectorizer
    _tfidf_matrix = tfidf_matrix
    _last_col_in = col_in

    return df, vectorizer, tfidf_matrix


@mcp.tool()
def get_similar_pairs(
    query: str,
    csv_path: str = DEFAULT_CSV_PATH,
    input_column: str = "원문",
    output_column: str = "이지리드 문장",
    max_results: int = 10
) -> str:
    """
    어려운 판례나 법률 문장(query)을 입력하면, 데이터셋(CSV)에서 TF-IDF 기술을 사용해 
    가장 의미적으로 유사한 상위 10개의 번역 대조쌍(Few-shot 예시)을 추출하여 반환합니다.

    :param query: 검색의 타겟이 되는 원문 판례 또는 법률 문장
    :param csv_path: 로컬 머신 혹은 도커 컨테이너 내의 CSV 데이터베이스 파일 경로
    :param input_column: 비교의 기준이 되는 원문 컬럼 이름 (비어둘 시 자동 매핑)
    :param output_column: 대치하여 사용할 쉬운 말(이지리드) 컬럼 이름 (비어둘 시 자동 매핑)
    :param max_results: 도출할 최대 Few-shot 데이터 개수 (기본값: 10개)
    """
    try:
        if csv_path == DEFAULT_CSV_PATH:
            create_default_csv_if_missing()

        if not os.path.exists(csv_path):
            return f"오류: '{csv_path}' 경로에 파일이 존재하지 않습니다."

        # 컬럼 오토 디텍션 및 기본값 선택 처리
        try:
            temp_df = pd.read_csv(csv_path, encoding="utf-8-sig", nrows=1)
        except UnicodeDecodeError:
            temp_df = pd.read_csv(csv_path, encoding="cp949", nrows=1)

        headers = list(temp_df.columns)

        if not input_column:
            input_column = headers[1] if len(headers) > 1 else (headers[0] if headers else "")
        if not output_column:
            output_column = headers[2] if len(headers) > 2 else (headers[0] if headers else "")

        # 인덱스 계산 실행
        df, vectorizer, tfidf_matrix = get_or_build_index(csv_path, input_column)

        # 코사인 유사도 분석
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

        # 결합 가이드라인 빌드
        header_info = f"[매칭 정보: {input_column} -> {output_column}]\n"
        return header_info + "\n\n".join(few_shot_results)

    except Exception as e:
        return f"작업 중 예외 발생: {str(e)}"

import os
#import uvicorn

if __name__ == "__main__":
    # Render.com이 주입해주는 환경 변수 포트를 기본으로 사용합니다.
    port = int(os.environ.get("PORT", 8000))
    
    # sse 방식으로 실행하여 외부 포트(0.0.0.0)를 통해 대기하도록 설정합니다.
    mcp.run(transport="sse", host="0.0.0.0", port=port) #render.com에서는 무시됨
    #asgi_app = mcp.jsonrpc_app
    
    #uvicorn.run(asgi_app, host="0.0.0.0", port=port)
    #fastmcp run law_matcher_mcp.py --port $PORT