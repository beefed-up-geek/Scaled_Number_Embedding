import openai
import numpy as np

# OpenAI API 키 설정 (환경변수에서 자동 로드됨)
client = openai.OpenAI(api_key="YOUR_OPENAI_API_KEY")

# 예제 문장
text = "GPT-3.5 embedding example"

# GPT-3.5의 임베딩 모델 사용
response = client.embeddings.create(
    input=text,
    model="text-embedding-ada-002"
)

# 임베딩 벡터 추출
embedding_vector = response.data[0].embedding

# NumPy 배열로 변환
embedding_array = np.array(embedding_vector)

# 전체 임베딩 벡터 출력
print("임베딩 벡터:")
print(embedding_array)

# 임베딩 벡터의 차원 확인
print("\n임베딩 차원:", embedding_array.shape)
