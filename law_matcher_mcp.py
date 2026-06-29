import os
import re
import pandas as pd
import uvicorn
from mcp.server import Server
from mcp.server.sse import SseServerTransport
import mcp.types as types
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import Response
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 로우레벨 MCP Server 정의 (버전 1.1.6 호환)
server = Server("EasyLead Law Matcher")

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


# 2. MCP 1.1.6 스타일 도구 선언 데코레이터 적용
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """도구 목록을 전송합니다."""
    return [
        types.Tool(
            name="get_similar_pairs",
            description="어려운 판례나 법률 문장(query)을 입력하면, 데이터셋(CSV)에서 유사한 이지리드 번역 대조쌍 상위 10개를 반환합니다.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "유사도를 판단할 기준 판결문 원문"},
                    "csv_path": {"type": "string", "description": "CSV 데이터베이스 파일 경로", "default": DEFAULT_CSV_PATH},
                    "input_column": {"type": "string", "description": "원문 컬럼 이름 (생략 가능)"},
                    "output_column": {"type": "string", "description": "쉬운 말 번역 컬럼 이름 (생략 가능)"},
                    "max_results": {"type": "integer", "description": "도출할 최대 결과 개수", "default": 10}
                },
                "required": ["query"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    """도구 호출에 응답합니다."""
    if name != "get_similar_pairs":
        raise ValueError(f"알 수 없는 도구 이름: {name}")

    if not arguments:
        arguments = {}

    query = arguments.get("query")
    csv_path = arguments.get("csv_path", DEFAULT_CSV_PATH)
    input_column = arguments.get("input_column")
    output_column = arguments.get("output_column")
    max_results = arguments.get("max_results", 10)

    try:
        if csv_path == DEFAULT_CSV_PATH:
            create_default_csv_if_missing()

        if not os.path.exists(csv_path):
            return [types.TextContent(type="text", text=f"오류: '{csv_path}' 경로에 파일이 존재하지 않습니다.")]

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
            return [types.TextContent(type="text", text="유사한 판례 문장쌍 예시를 찾지 못했습니다.")]

        header_info = f"[매칭 정보: {input_column} -> {output_column}]\n"
        return [types.TextContent(type="text", text=header_info + "\n\n".join(few_shot_results))]

    except Exception as e:
        return [types.TextContent(type="text", text=f"작업 중 예외 발생: {str(e)}")]


# 3. SSE 전송 인프라 및 스타렛 라우팅 설정
# 프록시 도메인 문제를 피하기 위해, 요청 스트림이 닿는 순간의 실시간 절대 도메인 주소로 SseTransport를 재생성합니다.
async def handle_sse_endpoint(request):
    """
    클라이언트의 Host 헤더를 실시간 판독하여 절대 경로 URL로 전송 포트를 매핑합니다.
    """
    # 1. 요청 도메인 정보 획득 (HTTPS 및 프록시 헤더 대응)
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.url.netloc)
    
    # 2. 실시간 절대 경로 주소 생성 (예: https://law-ss9n.onrender.com/messages)
    absolute_messages_url = f"{scheme}://{host}/messages"
    
    # 3. 실시간으로 경로를 래핑한 SseServerTransport 생성 및 통신 개시
    real_sse = SseServerTransport(absolute_messages_url)
    
    async with real_sse.connect_sse(request.scope, request.receive, request._send) as (read_stream, write_stream):
        await server.run(
            read_stream, 
            write_stream, 
            server.create_initialization_options()
        )
    return Response(status_code=200)

async def handle_message_endpoint(request):
    """메시지 전송을 임시 보정 처리"""
    # POST /messages 요청은 별도의 인스턴스 전송 방식으로 다이렉트 래핑 처리합니다.
    temp_sse = SseServerTransport("/messages")
    await temp_sse.handle_post_request(request.scope, request.receive, request._send)
    return Response(status_code=200)


# 스타렛(Starlette) 앱 및 라우트 선언
app = Starlette(
    debug=True,
    routes=[
        Route("/sse", endpoint=handle_sse_endpoint, methods=["GET"]),
        Route("/messages", endpoint=handle_message_endpoint, methods=["POST"]),
    ],
)


# 4. Render.com 호환 실행
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        proxy_headers=True,
        forwarded_allow_ips="*"
    )