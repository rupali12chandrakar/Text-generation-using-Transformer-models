
# 🧠 Text Generation using Transformer Models (GPT-2)

Welcome to my simple and powerful **Text Generation** project using **Hugging Face Transformers** and **GPT-2**! This project demonstrates how large language models can generate human-like text based on a short input prompt.

---

## 🔍 Overview

This project uses the pre-trained `gpt2` model from Hugging Face to generate paragraphs of text from a single input line. You can run the entire project in **Google Colab** without any installation hassle.

---

## 📌 Features

- Input a custom prompt
- Generate natural language text
- Customize parameters: `max_length`, `temperature`, `top_k`, `top_p`
- Easily extendable to fine-tuned models

---

## 🚀 Technologies Used

- Python 🐍
- Transformers Library by Hugging Face 🤗
- PyTorch 🔥
- Google Colab ☁️

---

## 🛠️ Setup Instructions

1. **Install Required Libraries**
```bash
pip install transformers torch
Run the Code

python
Copy
Edit
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load model & tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Input prompt
prompt = "Once upon a time in Chhattisgarh,"
inputs = tokenizer.encode(prompt, return_tensors="pt")

# Generate text
outputs = model.generate(
    inputs,
    max_length=100,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.9,
    num_return_sequences=1
)

# Output
generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated)
💡 Sample Output
css
Copy
Edit
Once upon a time in Chhattisgarh, a young innovator discovered the magic of artificial intelligence and started building creative projects...
🧠 Learning Outcomes
Understood the GPT-2 transformer architecture

Learned how to use Hugging Face’s API for text generation

Tuned parameters for creative output control

📎 Try it on Google Colab
