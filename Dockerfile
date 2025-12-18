# 파이썬 의존성 설치
FROM python:3.10.19-slim

# 파일 이동
WORKDIR /app

# 의존성 설치
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# 코드 복사
COPY . .

# fastAPI 실행하기
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
