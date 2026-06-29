# ⚖️ EasyLead Law Matcher MCP Server

어려운 판례문이나 고난도의 법률 문장을 사용자가 입력하면, 로컬 CSV 데이터베이스에 존재하는 말뭉치에서 TF-IDF 의미론적 유사도 분석을 거쳐 가장 어조가 일치하는 10개의 **번역 대조 문장쌍(Few-shot)**을 도출하는 Model Context Protocol(MCP) 서버입니다.

이 도구를 사용하면 Claude나 Cursor 등의 AI 모델이 판례의 쉬운 말(이지리드) 번역 스타일을 실시간으로 참조하여 고도로 정밀한 쉬운 법률 해설서를 일관적으로 생산할 수 있습니다.

---

## 주요 기능 및 도구 (Tools)

### `get_similar_pairs`
사용자가 제시한 질의(Query)와 유사한 순서대로 매칭 결과 대조 데이터를 제공합니다.
- **인자**:
  - `query` (필수): 유사도를 비교할 판결문 원문 텍스트
  - `csv_path` (선택): 데이터 파일 경로 (기본값: `law2easy.csv`)
  - `input_column` (선택): 검색할 원문이 담긴 컬럼 (기본값: 자동 탐색 및 인덱스 1번 매핑)
  - `output_column` (선택): 인출하여 활용할 이지리드 결과 컬럼 (기본값: 자동 탐색 및 인덱스 2번 매핑)
  - `max_results` (선택): 예시 개수 제한 (기본값: 10개)

---

## 설치 및 연동 설정 방법

### 1. Claude Desktop 설정 연동
Claude Desktop의 환경 설정 파일(`claude_desktop_config.json`)을 열고 아래와 같이 환경에 맞게 추가합니다.

#### Python 직접 로컬 구동 시:
```json
{
  "mcpServers": {
    "law-matcher": {
      "command": "python",
      "args": [
        "/your/local/path/law_matcher_mcp.py"
      ]
    }
  }
}