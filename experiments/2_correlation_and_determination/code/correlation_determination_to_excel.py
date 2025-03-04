import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm

def analyze_digits(csv_file, output_filename, digit_length=1):
    """
    '0' ~ '9' (digit_length=1)
    '00' ~ '99' (digit_length=2)
    '000' ~ '999' (digit_length=3)
    '0000' ~ '9999' (digit_length=4)
    
    범위 숫자에 대해 각 임베딩 차원별 선형회귀 및 상관계수 / 결정계수 / 표준편차 계산
    결과를 output_filename.csv에 저장
    """
    # 1. CSV 파일 로드
    data = pd.read_csv(csv_file)
    
    # 2. 키값 설정 및 존재하는 키만 선택
    keys = [f"{i:0{digit_length}d}" for i in range(10**digit_length)]
    valid_data = data[data["key"].astype(str).isin(keys)]
    
    # 3. X(정수 변환), Y(차원별 임베딩 값) 준비
    X_list = valid_data["key"].astype(int).values.reshape(-1, 1)
    embedding_columns = [col for col in data.columns if col.startswith("dim_")]
    embedding_dim = len(embedding_columns)
    results = []  # (dimension, correlation, r2, std_dev)을 담을 리스트
    
    # 4. 각 임베딩 차원별 선형회귀 및 correlation, R^2, 표준편차 계산
    for d in tqdm(range(embedding_dim), desc="Processing Dimensions", unit="dim"):
        Y_array = valid_data[embedding_columns[d]].values
        
        # 상관계수(피어슨)
        corr, _ = pearsonr(X_list.flatten(), Y_array)
        
        # 선형회귀
        lr = LinearRegression()
        lr.fit(X_list, Y_array)
        Y_pred = lr.predict(X_list)
        
        # 결정계수
        r2 = r2_score(Y_array, Y_pred)

        # 표준편차 계산
        std_dev = np.std(Y_array, ddof=1)  # 샘플 표준편차 (Bessel's correction 적용)
        
        # dimension은 1-based로 저장
        results.append([d+1, corr, r2, std_dev])
    
    # 5. 결과 DataFrame 생성 및 CSV 저장
    df = pd.DataFrame(results, columns=["Dimension", "Correlation", "R^2", "Std_Dev"])
    df.to_csv(output_filename, index=False)
    print(f"Analysis complete. Results saved to {output_filename}")
