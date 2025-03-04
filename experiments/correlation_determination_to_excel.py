import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm  # tqdm 추가

def analyze_digits(npz_file, embedding_dim, output_filename, digit_length=1):
    """
    '0' ~ '9' (digit_length=1)
    '00' ~ '99' (digit_length=2)
    '000' ~ '999' (digit_length=3)
    '0000' ~ '9999' (digit_length=4)
    
    범위 숫자에 대해 각 임베딩 차원별 선형회귀 및 상관계수 / 결정계수 계산
    결과를 output_filename.xlsx에 저장
    """
    # 1. .npz 파일 로드
    data = np.load(npz_file)
    
    # 2. 키값 설정 및 존재하는 키만 선택
    keys = [f"{i:0{digit_length}d}" for i in range(10**digit_length)]
    valid_keys = [k for k in keys if k in data]
    
    # 3. X(정수 변환), Y(차원별 임베딩 값) 준비
    X_list = []
    Y_lists = [[] for _ in range(embedding_dim)]
    
    for k in valid_keys:
        num_val = int(k)  # "000" -> 0, "001" -> 1, ...
        X_list.append(num_val)
        
        embedding_vec = data[k]  # shape: (embedding_dim,)
        for d in range(embedding_dim):
            Y_lists[d].append(embedding_vec[d])
    
    # 4. 선형회귀 입력으로 X_array 준비
    X_array = np.array(X_list).reshape(-1, 1)
    results = []  # (dimension, correlation, r2)을 담을 리스트
    
    # 5. 각 임베딩 차원별 선형회귀 및 correlation, R^2 계산
    for d in tqdm(range(embedding_dim), desc="Processing Dimensions", unit="dim"):
        Y_array = np.array(Y_lists[d])
        
        # 상관계수(피어슨)
        corr, _ = pearsonr(X_array.flatten(), Y_array)
        
        # 선형회귀
        lr = LinearRegression()
        lr.fit(X_array, Y_array)
        Y_pred = lr.predict(X_array)
        
        # 결정계수
        r2 = r2_score(Y_array, Y_pred)
        
        # dimension은 1-based로 저장
        results.append([d+1, corr, r2])
    
    # 6. 결과 DataFrame 생성
    df = pd.DataFrame(results, columns=["Dimension", "Correlation", "R^2"])
    
    # 7. 결과를 엑셀로 저장
    df.to_excel(output_filename, index=False)
    print(f"Analysis complete. Results saved to {output_filename}")
