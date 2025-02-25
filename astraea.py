# File: astraea.py
"""
Astraea Inference Script (Standalone Free-Text Generation Version)

This script loads the trained AstraeaTernaryNet student model,
a free-text decoder, and a self-contained text encoder that replaces
the teacher model. The text encoder uses GPT-2 and a learned linear layer
to produce 4096-dimensional embeddings from raw text. These embeddings are
projected and fed to the student model, whose 32-dimensional output is then used
by the decoder to generate a free-form text response.

Commands:
  /test  - Run a predefined set of 9 test questions.
  /exit  - Exit the interpreter.
Any other input is treated as a query.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel

# -----------------------------
# Initialize the GPT-2 tokenizer (for text encoder and decoding)
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.bos_token is None:
    tokenizer.bos_token = "Ġ"
if tokenizer.eos_token is None:
    tokenizer.eos_token = ""

# -----------------------------
# Define the Student Model: AstraeaTernaryNet
# -----------------------------
class AstraeaTernaryNet(nn.Module):
    def __init__(self, input_size=1024, hidden1=3072, hidden2=2048, hidden3=1536, hidden4=768, output_size=32):
        super(AstraeaTernaryNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1, bias=False)
        self.fc2 = nn.Linear(hidden1, hidden2, bias=False)
        self.fc3 = nn.Linear(hidden2, hidden3, bias=False)
        self.fc4 = nn.Linear(hidden3, hidden4, bias=False)
        self.fc5 = nn.Linear(hidden4, output_size, bias=False)
        self.relu = nn.ReLU()
        # Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.fc5.weight)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# -----------------------------
# Define the Free Text Decoder (LSTM-based)
# -----------------------------
class FreeTextDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=1, max_len=50):
        super(FreeTextDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.init_fc = nn.Linear(32, hidden_dim)  # maps student embedding (32-dim) to initial LSTM hidden state
        self.max_len = max_len

    def forward(self, student_embedding, target_ids):
        batch_size = student_embedding.size(0)
        h0 = self.init_fc(student_embedding).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        embeds = self.embedding(target_ids)
        outputs, _ = self.lstm(embeds, (h0, c0))
        logits = self.fc(outputs)
        return logits

    def generate(self, student_embedding, device, temperature=1.0):
        batch_size = student_embedding.size(0)
        h0 = self.init_fc(student_embedding).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        input_token = torch.tensor([tokenizer.bos_token_id] * batch_size, device=device).unsqueeze(1)
        generated = []
        hidden, cell = h0, c0
        for _ in range(self.max_len):
            embeds = self.embedding(input_token)
            output, (hidden, cell) = self.lstm(embeds, (hidden, cell))
            logits = self.fc(output.squeeze(1))
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated.append(next_token)
            input_token = next_token
        generated = torch.cat(generated, dim=1)
        return generated

# -----------------------------
# Define a Text Encoder to Replace the Teacher Model
# -----------------------------
class TextEncoder(nn.Module):
    def __init__(self, pretrained_model="gpt2", target_dim=4096):
        super(TextEncoder, self).__init__()
        # Load a pretrained GPT-2 model for encoding.
        self.encoder = AutoModel.from_pretrained(pretrained_model)
        # We'll average pool over the sequence dimension.
        # Map from GPT-2's hidden size (typically 768) to target_dim (4096)
        self.linear = nn.Linear(self.encoder.config.hidden_size, target_dim)
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # outputs[0]: [batch, seq_len, hidden_size]
        # Average pool over the sequence length.
        pooled = outputs[0].mean(dim=1)  # [batch, hidden_size]
        # Map to target dimension.
        out = self.linear(pooled)  # [batch, target_dim]
        return out

def load_text_encoder():
    device = torch.device("cpu")
    encoder = TextEncoder(pretrained_model="gpt2", target_dim=4096).to(device)
    encoder.eval()
    return encoder

# -----------------------------
# Projection Layer Setup
# -----------------------------
# In training, we projected teacher embeddings (4096-dim) to student input (1024-dim).
def get_proj_in():
    return nn.Linear(4096, 1024, bias=False)

# -----------------------------
# Model Loading Functions
# -----------------------------
def load_student_model(checkpoint_path="astraea_distilled.pth"):
    device = torch.device("cpu")
    model = AstraeaTernaryNet().to(device)
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        print(f"Loaded student model from {checkpoint_path}")
    except FileNotFoundError:
        print(f"Error: {checkpoint_path} not found. Please train the model first.")
        exit(1)
    return model

def load_decoder():
    device = torch.device("cpu")
    decoder = FreeTextDecoder(vocab_size=tokenizer.vocab_size, embed_dim=128, hidden_dim=256, num_layers=1, max_len=50).to(device)
    # Optionally load a decoder checkpoint here.
    return decoder

# -----------------------------
# Inference Function (Standalone Free-Text Generation)
# -----------------------------
def process_query(model, decoder, text_encoder, query):
    """
    Processes a query:
      - If arithmetic (only digits/operators), evaluates it.
      - Otherwise, uses the text encoder to generate a 4096-dim embedding from raw text,
        projects it to 1024-dim, passes through the student model to get a 32-dim vector,
        and then uses the decoder to generate free-form text.
    """
    # Check if query is arithmetic.
    stripped = query.strip().replace(" ", "")
    if stripped.replace("+", "").replace("-", "").replace("*", "").replace("/", "").isdigit():
        try:
            answer = str(eval(query))
            return None, answer
        except Exception:
            return None, "Error in arithmetic evaluation"
    
    # Use the text encoder to get a 4096-dim embedding.
    encoding = tokenizer(query, return_tensors="pt")
    teacher_embed = text_encoder(encoding["input_ids"], attention_mask=encoding.get("attention_mask"))
    teacher_embed = teacher_embed.squeeze(0)
    teacher_embed = teacher_embed / teacher_embed.norm()
    
    # Project to 1024-dim.
    proj_in = get_proj_in()
    input_tensor = proj_in(teacher_embed)
    
    # Pass through the student model.
    with torch.no_grad():
        student_output = model(input_tensor.unsqueeze(0))  # [1, 32]
        student_output = student_output / student_output.norm(dim=1, keepdim=True)
    
    # Generate a free-text response.
    with torch.no_grad():
        generated_ids = decoder.generate(student_output, device=torch.device("cpu"), temperature=1.0)
    generated_ids = generated_ids[0].tolist()
    if tokenizer.eos_token_id in generated_ids:
        generated_ids = generated_ids[:generated_ids.index(tokenizer.eos_token_id)]
    decoded_text = tokenizer.decode(generated_ids, clean_up_tokenization_spaces=True)
    
    return student_output.cpu().numpy(), decoded_text

# -----------------------------
# Interactive Mode
# -----------------------------
def run_interactive(model, decoder, text_encoder):
    print("Enter your queries (type '/test' to run test suite, '/exit' to quit):")
    while True:
        query = input("> ")
        if query.lower() == "/exit":
            print("Exiting...")
            break
        elif query.lower() == "/test":
            run_test(model, decoder, text_encoder)
            continue
        try:
            output, answer = process_query(model, decoder, text_encoder, query)
            print(f"Query: {query}")
            if output is not None:
                print(f"Student embedding (raw vector): {output}")
            print(f"Decoded answer: {answer}\n")
        except Exception as e:
            print(f"Error processing query: {e}\n")

# -----------------------------
# Test Mode (Runs a Predefined Set of 9 Questions)
# -----------------------------
def run_test(model, decoder, text_encoder):
    test_questions = [
        "Hello",
        "Can you help me?",
        "That's odd",
        "2+2",
        "5+1",
        "3-1",
        "2*2",
        "2/2",
        "What is the capital of Australia?"
    ]
    print("\nThe following test questions will be run:")
    for i, q in enumerate(test_questions, start=1):
        print(f"{i}. {q}")
    print("\nRunning test questions...\n")
    for q in test_questions:
        output, answer = process_query(model, decoder, text_encoder, q)
        print(f"Query: {q}")
        if output is not None:
            print(f"Student embedding (raw vector): {output}")
        print(f"Decoded answer: {answer}\n")

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    student_model = load_student_model("astraea_distilled.pth")
    decoder = load_decoder()
    text_encoder = load_text_encoder()
    run_interactive(student_model, decoder, text_encoder)
