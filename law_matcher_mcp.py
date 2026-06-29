import os
import re
import pandas as pd

from mcp.server import Server
from mcp.server.sse import SseServerTransport

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================
# MCP SERVER
# =====================
server = Server("EasyLead Law Matcher")

DEFAULT_CSV_PATH = "law2easy.csv"

_cache = {
    "df": None,
    "vectorizer": None,
    "matrix": None,
    "col": None,
}


# =====================
# INDEX
# =====================
def build_index(csv_path, col):
    df = pd.read_csv(csv_path, encoding="utf-8-sig").dropna()

    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
    )

    matrix = vec.fit_transform(df[col].astype(str).tolist())

    _cache.update({
        "df": df,
        "vectorizer": vec,
        "matrix": matrix,
        "col": col,
    })

    return df, vec, matrix


def get_index(csv_path, col):
    if _cache["matrix"] is not None and _cache["col"] == col:
        return _cache["df"], _cache["vectorizer"], _cache["matrix"]

    return build_index(csv_path, col)


# =====================
# TOOL (1.1.6 스타일)
# =====================
@server.list_tools()
async def list_tools():
    import mcp.types as types

    return [
        types.Tool(
            name="get_similar_pairs",
            description="TF-IDF 기반 법률 유사문장 검색",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "csv_path": {"type": "string", "default": DEFAULT_CSV_PATH},
                    "input_column": {"type": "string", "default": "원문"},
                    "output_column": {"type": "string", "default": "이지리드"},
                    "max_results": {"type": "integer", "default": 10},
                },
                "required": ["query"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    import mcp.types as types

    if name != "get_similar_pairs":
        return [types.TextContent(type="text", text="Unknown tool")]

    query = arguments["query"]
    csv_path = arguments.get("csv_path", DEFAULT_CSV_PATH)
    input_col = arguments.get("input_column", "원문")
    output_col = arguments.get("output_column", "이지리드")
    max_results = arguments.get("max_results", 10)

    df, vec, matrix = get_index(csv_path, input_col)

    qv = vec.transform([query])
    scores = cosine_similarity(qv, matrix).flatten()

    idxs = scores.argsort()[::-1][:max_results]

    out = []
    for i in idxs:
        row = df.iloc[i]
        out.append(
            f"입력: {row[input_col]}\n출력: {row[output_col]}"
        )

    return [types.TextContent(type="text", text="\n\n".join(out))]


# =====================
# SSE TRANSPORT (1.1.6 핵심)
# =====================
sse = SseServerTransport("/messages")


async def handle_sse(request):
    async with sse.connect_sse(
        request.scope,
        request.receive,
        request._send
    ) as (read, write):
        await server.run(
            read,
            write,
            server.create_initialization_options()
        )


async def handle_messages(request):
    # 🔥 1.1.6 핵심: 이게 정답
    await sse.handle_post_message(
        request.scope,
        request.receive,
        request._send
    )


# =====================
# STARLETTE APP
# =====================
from starlette.applications import Starlette
from starlette.routing import Route

app = Starlette(
    routes=[
        Route("/sse", handle_sse, methods=["GET"]),
        Route("/messages", handle_messages, methods=["POST"]),
    ]
)


# =====================
# RUN
# =====================
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        proxy_headers=True,
        forwarded_allow_ips="*"
    )