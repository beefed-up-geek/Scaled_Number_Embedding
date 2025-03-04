import os
from correlation_determination_to_excel import analyze_digits

xl_file = "../get_embeddings/number_embeddings/qwen_embeddings.csv"
output_filename = "qwen_1.csv"
digit_length = 1

analyze_digits(xl_file, output_filename, digit_length)
print(f"Result saved")
