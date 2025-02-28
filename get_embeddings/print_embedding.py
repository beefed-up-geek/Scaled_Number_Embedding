import os
import numpy as np

print("현재 디렉토리:", os.getcwd())
print("현재 디렉토리 목록:", os.listdir("."))

file_path = "llama_embeddings.npz"
if not os.path.exists(file_path):
    print(f"파일 {file_path} 이(가) 존재하지 않습니다!")
else:
    data = np.load(file_path)
    print("데이터 키 목록:", data.files)
    if "005" in data.files:
        embedding_005 = data["005"]
        print("005 임베딩:", embedding_005)
    else:
        print("'005'라는 키가 없습니다!")
