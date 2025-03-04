import os
from correlation_determination_to_excel import analyze_digits
'''
bert_embeddings.csv      | 3 BPE
deepseek_embeddings.csv  | Digit
gemma_embeddings.csv     | Digit
gpt2_embeddings.csv      | 4 BPE
llama_embeddings.csv     | 3 BPE
qwen_embeddings.csv      | Digit
'''

directory = "../../../get_embeddings/number_embeddings/{}"
analyze_digits(directory.format("bert_embeddings.csv"), "../bert_1.csv", 1)
analyze_digits(directory.format("bert_embeddings.csv"), "../bert_2.csv", 2)
analyze_digits(directory.format("bert_embeddings.csv"), "../bert_3.csv", 3)
analyze_digits(directory.format("deepseek_embeddings.csv"), "../deepseek_1.csv", 1)
analyze_digits(directory.format("gemma_embeddings.csv"), "../gemma_1.csv", 1)
analyze_digits(directory.format("gpt2_embeddings.csv"), "../gpt2_1.csv", 1)
analyze_digits(directory.format("gpt2_embeddings.csv"), "../gpt2_2.csv", 2)
analyze_digits(directory.format("gpt2_embeddings.csv"), "../gpt2_3.csv", 3)
analyze_digits(directory.format("gpt2_embeddings.csv"), "../gpt2_4.csv", 4)
analyze_digits(directory.format("llama_embeddings.csv"), "../llama_1.csv", 1)
analyze_digits(directory.format("llama_embeddings.csv"), "../llama_2.csv", 2)
analyze_digits(directory.format("llama_embeddings.csv"), "../llama_3.csv", 3)
analyze_digits(directory.format("qwen_embeddings.csv"), "../qwen_1.csv", 1)
