import os
from correlation_determination_to_excel import analyze_digits

npz_file = "../get_embeddings/number_embeddings/gemma_embeddings.npz"
embedding_dim = 2048
output_filename = "gemma_1.xlsx"
digit_length = 1

analyze_digits(npz_file, embedding_dim, output_filename, digit_length)
print(f"Result saved")
