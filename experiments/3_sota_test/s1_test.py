import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("simplescaling/s1.1-7B")
model = AutoModelForCausalLM.from_pretrained("simplescaling/s1.1-7B")

# Ensure model is on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Test function for model inference
def test_model(prompt, max_length=1000, temperature=0.7, top_k=50):
    # Tokenize input text
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=max_length, 
            temperature=temperature, 
            top_k=top_k,
            do_sample=True
        )

    # Decode generated text
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output

  
if __name__ == "__main__":
    prompt = "3412341+2634574=?"
    result = test_model(prompt)
    print("Generated Text:", result)