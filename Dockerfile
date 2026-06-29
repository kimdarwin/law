# 1. 안정적인 파이썬 환경 사용
FROM python:3.11-slim

# 2. 컨테이너 내부 작업 디렉토리 지정
WORKDIR /app

# 3. 가상환경 구축 최적화를 위한 종속성 선 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 소스 코드 복사 (데이터셋 CSV 파일도 프로젝트 내 포함되도록 처리 가능)
COPY law_matcher_mcp.py .
COPY law2easy.csv .

# 5. 실행 환경 변수 및 입출력 버퍼 설정
ENV PYTHONUNBUFFERED=1

# 6. 표준 입출력(STDIO)을 통한 MCP 통신 실행
#ENTRYPOINT ["python", "law_matcher_mcp.py"]
ENTRYPOINT ["fastmcp", "run", "law_matcher_mcp.py"]