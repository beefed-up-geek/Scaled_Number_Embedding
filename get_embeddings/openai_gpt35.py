import openai
import numpy as np

# OpenAI API 키 설정 (직접 입력하거나 환경 변수에서 불러오기)
# openai.api_key = ""

# 예제 문장
text = "GPT-3.5 embedding example"

# GPT-3.5의 임베딩 모델 사용
response = openai.Embedding.create(
    input=text,
    model="text-embedding-ada-002"  # GPT-3.5에서 사용 가능한 임베딩 모델
)

# 임베딩 벡터 추출
embedding_vector = response["data"][0]["embedding"]

# NumPy 배열로 변환
embedding_array = np.array(embedding_vector)

# 결과 출력
print("임베딩 차원:", embedding_array.shape)
print("임베딩 벡터 샘플:", embedding_array[:10])  # 처음 10개 요소 출력
