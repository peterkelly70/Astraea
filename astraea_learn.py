# File: astraea_learning.py
"""
Astraea Learning Script

This script offers multiple training modules for the Astraea student model:
  1. SQuAD Fine-Tuning (embedding-only distillation)
  2. Arithmetic Fine-Tuning
  3. Math Book Fine-Tuning (runs on every unseen book in the "books" subdirectory)
  4. Free Text QA Fine-Tuning (trains a lightweight decoder to generate free-text answers)
  5. RL Fine-Tuning (uses REINFORCE with a multi-objective reward based on SQuAD data)

After each training module, the script appends a log entry to "training_log.txt".
For book training, processed book filenames are recorded in "books_trained.txt".
This allows you to continue training later and know what modules have been completed.
"""

import os
import re
import random
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import ollama
from transformers import AutoTokenizer, AutoModel

# Initialize GPT-2 tokenizer for both decoding and text encoding.
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.bos_token is None:
    tokenizer.bos_token = "Ġ"
if tokenizer.eos_token is None:
    tokenizer.eos_token = ""

# -----------------------------
# Helper: Ternary Quantization
# -----------------------------
def quantize_to_ternary(w, scale=1.0):
    thresh = scale * torch.mean(torch.abs(w))
    return torch.where(w > thresh, torch.tensor(1.0),
                       torch.where(w < -thresh, torch.tensor(-1.0), torch.tensor(0.0)))

# -----------------------------
# Student Model: AstraeaTernaryNet
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
# Decoder: LSTM-based Free Text Decoder
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
        h0 = self.init_fc(student_embedding).unsqueeze(0)  # [1, batch, hidden_dim]
        c0 = torch.zeros_like(h0)
        embeds = self.embedding(target_ids)  # [batch, seq_len, embed_dim]
        outputs, _ = self.lstm(embeds, (h0, c0))
        logits = self.fc(outputs)  # [batch, seq_len, vocab_size]
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
# Text Encoder (Standalone) to Replace the Teacher Model
# -----------------------------
class TextEncoder(nn.Module):
    def __init__(self, pretrained_model="gpt2", target_dim=4096):
        super(TextEncoder, self).__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model)
        self.linear = nn.Linear(self.encoder.config.hidden_size, target_dim)
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs[0].mean(dim=1)  # average pool over sequence
        out = self.linear(pooled)
        return out

def load_text_encoder():
    device = torch.device("cpu")
    encoder = TextEncoder(pretrained_model="gpt2", target_dim=4096).to(device)
    encoder.eval()
    return encoder

# -----------------------------
# Projection Layer Setup
# -----------------------------
def get_proj_in():
    return nn.Linear(4096, 1024, bias=False)

# -----------------------------
# Data Loading Function (SQuAD)
# -----------------------------
def load_squad_data(filename="squad_train.csv"):
    try:
        df = pd.read_csv(filename)
        print(f"Loaded {len(df)} SQuAD pairs from {filename}")
    except Exception as e:
        print("SQuAD CSV not found. Loading SQuAD dataset...")
        squad = load_dataset("squad_v2")["train"]
        questions = [item["question"] for item in squad]
        answers = [item["answers"]["text"][0] if item["answers"]["text"] else "Unanswerable" for item in squad]
        df = pd.DataFrame({"question": questions, "answer": answers})
        df.to_csv(filename, index=False)
        print(f"Saved {len(df)} SQuAD pairs to {filename}")
    return df

# -----------------------------
# Reward Function for RL Fine-Tuning
# -----------------------------
def compute_reward(generated_text, ground_truth, text_encoder):
    """
    Computes reward as the cosine similarity between the text encoder embeddings
    of the generated answer and the ground truth answer.
    """
    # Encode ground truth
    gt_encoding = tokenizer(ground_truth, return_tensors="pt")
    gen_encoding = tokenizer(generated_text, return_tensors="pt")
    with torch.no_grad():
        gt_embed = text_encoder(gt_encoding["input_ids"]).squeeze(0)
        gen_embed = text_encoder(gen_encoding["input_ids"]).squeeze(0)
    gt_embed = gt_embed / gt_embed.norm()
    gen_embed = gen_embed / gen_embed.norm()
    cosine_sim = torch.dot(gt_embed, gen_embed)
    return cosine_sim.item()

# -----------------------------
# RL Fine-Tuning Module (REINFORCE)
# -----------------------------
def train_rl_finetune(model, decoder, text_encoder, num_epochs=5, batch_size=16):
    """
    Fine-tunes the decoder using reinforcement learning with a reward based on:
      - Similarity to the ground truth answer (from SQuAD) using the text encoder.
    Uses a simple REINFORCE update on the decoder parameters.
    """
    df = load_squad_data()
    optimizer = optim.Adam(decoder.parameters(), lr=1e-5)  # small LR for RL fine-tuning
    for epoch in range(num_epochs):
        total_loss = 0.0
        batch = df.sample(batch_size)
        questions = batch["question"].tolist()
        ground_truths = batch["answer"].tolist()
        for q, gt in zip(questions, ground_truths):
            # Encode question with text encoder.
            encoding = tokenizer(q, return_tensors="pt")
            teacher_embed = text_encoder(encoding["input_ids"], attention_mask=encoding.get("attention_mask"))
            teacher_embed = teacher_embed.squeeze(0)
            teacher_embed = teacher_embed / teacher_embed.norm()
            proj_in_layer = get_proj_in()  # temporary projection
            input_tensor = proj_in_layer(teacher_embed)
            with torch.no_grad():
                student_output = model(input_tensor.unsqueeze(0))  # [1, 32]
                student_output = student_output / student_output.norm(dim=1, keepdim=True)
            # Generate response using the decoder (sampling).
            generated_ids = decoder.generate(student_output, device=torch.device("cpu"), temperature=1.0)
            # Compute log probabilities via teacher-forced forward pass.
            generated_ids_tensor = torch.tensor(generated_ids, dtype=torch.long)  # [1, seq_len]
            logits = decoder(student_output, generated_ids_tensor[:, :-1])
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            target_tokens = generated_ids_tensor[:, 1:]
            token_log_probs = log_probs.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)
            sequence_log_prob = token_log_probs.mean()
            # Decode generated text.
            gen_ids = generated_ids[0].tolist()
            if tokenizer.eos_token_id in gen_ids:
                gen_ids = gen_ids[:gen_ids.index(tokenizer.eos_token_id)]
            generated_text = tokenizer.decode(gen_ids, clean_up_tokenization_spaces=True)
            # Compute reward.
            reward = compute_reward(generated_text, gt, text_encoder)
            # REINFORCE loss: negative reward times log probability.
            loss = -reward * sequence_log_prob
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / batch_size
        print(f"Epoch {epoch+1}/{num_epochs}, Avg RL Loss: {avg_loss:.4f}")
    print("RL fine-tuning completed.")

# -----------------------------
# Logging Helpers
# -----------------------------
LOG_FILE = "training_log.txt"
BOOKS_TRAINED_FILE = "books_trained.txt"

def log_training(module_name, epochs):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp}: {module_name} completed ({epochs} epochs)\n"
    with open(LOG_FILE, "a") as f:
        f.write(log_entry)
    print(f"Logged training: {module_name}")

def get_unseen_books():
    os.makedirs("books", exist_ok=True)
    all_books = [os.path.join("books", f) for f in os.listdir("books") if f.endswith((".tex", ".txt"))]
    if os.path.exists(BOOKS_TRAINED_FILE):
        with open(BOOKS_TRAINED_FILE, "r") as f:
            seen = set(line.strip() for line in f.readlines())
    else:
        seen = set()
    unseen = [book for book in all_books if os.path.basename(book) not in seen]
    return unseen

def mark_book_as_trained(book_path):
    with open(BOOKS_TRAINED_FILE, "a") as f:
        f.write(os.path.basename(book_path) + "\n")
    print(f"Marked {os.path.basename(book_path)} as trained.")

# -----------------------------
# Other Training Modules (SQuAD, Arithmetic, Math Book, Free Text QA)
# -----------------------------
def train_on_squad(model, num_epochs=5, batch_size=32):
    df = load_squad_data()
    optimizer = optim.Adam(list(model.parameters()) + list(get_proj_in().parameters()), lr=0.001)
    criterion = nn.MSELoss()
    model.train()
    with tqdm(total=num_epochs, desc="SQuAD Training", unit="epoch") as pbar:
        for epoch in range(num_epochs):
            batch = df.sample(batch_size)
            questions = batch["question"].tolist()
            answers = batch["answer"].tolist()
            q_embeds = [ollama.embeddings(model="mistral:7b-instruct-q4_0", prompt=q)["embedding"] for q in questions]
            q_embeds = torch.tensor(q_embeds, dtype=torch.float32)
            q_embeds = q_embeds / torch.norm(q_embeds, dim=1, keepdim=True)
            inputs = get_proj_in()(q_embeds).to(torch.device("cpu"))
            a_embeds = [ollama.embeddings(model="mistral:7b-instruct-q4_0", prompt=a)["embedding"] for a in answers]
            a_embeds = torch.tensor(a_embeds, dtype=torch.float32)
            a_embeds = a_embeds / torch.norm(a_embeds, dim=1, keepdim=True)
            targets = proj_out(a_embeds)
            targets = targets / targets.norm(dim=1, keepdim=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs / outputs.norm(dim=1, keepdim=True)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})
            pbar.update(1)
    log_training("SQuAD Fine-Tuning", num_epochs)
    print("SQuAD fine-tuning completed.")

def train_on_arithmetic(model, num_epochs=5, batch_size=32):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    model.train()
    with tqdm(total=num_epochs, desc="Arithmetic Training", unit="epoch") as pbar:
        for epoch in range(num_epochs):
            questions, answers = [], []
            for _ in range(batch_size):
                a = random.randint(1, 100)
                b = random.randint(1, 100)
                op = random.choice(["+", "-", "*", "/"])
                if op == "/":
                    a = a * b
                    q = f"{a} / {b}"
                    a_text = str(a // b)
                else:
                    q = f"{a} {op} {b}"
                    a_text = str(eval(q))
                questions.append(q)
                answers.append(a_text)
            q_embeds = [ollama.embeddings(model="mistral:7b-instruct-q4_0", prompt=q)["embedding"] for q in questions]
            q_embeds = torch.tensor(q_embeds, dtype=torch.float32)
            q_embeds = q_embeds / torch.norm(q_embeds, dim=1, keepdim=True)
            inputs = get_proj_in()(q_embeds).to(torch.device("cpu"))
            a_embeds = [ollama.embeddings(model="mistral:7b-instruct-q4_0", prompt=a)["embedding"] for a in answers]
            a_embeds = torch.tensor(a_embeds, dtype=torch.float32)
            a_embeds = a_embeds / torch.norm(a_embeds, dim=1, keepdim=True)
            targets = proj_out(a_embeds)
            targets = targets / targets.norm(dim=1, keepdim=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs / outputs.norm(dim=1, keepdim=True)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})
            pbar.update(1)
    log_training("Arithmetic Fine-Tuning", num_epochs)
    print("Arithmetic fine-tuning completed.")

def train_on_math_book_file(model, book_path, num_epochs=5, batch_size=16):
    try:
        with open(book_path, "r", encoding="utf-8") as f:
            content = f.read()
        if book_path.endswith(".tex"):
            from pylatexenc.latex2text import LatexNodes2Text
            content = LatexNodes2Text().latex_to_text(content)
        paragraphs = content.split("\n\n")
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 50]
    except Exception as e:
        print(f"Error loading {book_path}: {e}")
        return
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    model.train()
    with tqdm(total=num_epochs, desc=f"Book Training ({os.path.basename(book_path)})", unit="epoch") as pbar:
        for epoch in range(num_epochs):
            batch = random.sample(paragraphs, min(batch_size, len(paragraphs)))
            q_embeds = [ollama.embeddings(model="mistral:7b-instruct-q4_0", prompt=p)["embedding"] for p in batch]
            q_embeds = torch.tensor(q_embeds, dtype=torch.float32)
            q_embeds = q_embeds / torch.norm(q_embeds, dim=1, keepdim=True)
            inputs = get_proj_in()(q_embeds).to(torch.device("cpu"))
            targets = inputs
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs / outputs.norm(dim=1, keepdim=True)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})
            pbar.update(1)
    log_training(f"Math Book Fine-Tuning ({os.path.basename(book_path)})", num_epochs)
    mark_book_as_trained(book_path)
    print(f"Math book training on {book_path} completed.")

def train_on_all_unseen_books(model, num_epochs=5, batch_size=16):
    unseen_books = get_unseen_books()
    if not unseen_books:
        print("No unseen books found in the 'books' directory.")
        return
    for book in unseen_books:
        train_on_math_book_file(model, book, num_epochs=num_epochs, batch_size=batch_size)

def train_free_text_qa(model, decoder, num_epochs=5, batch_size=32, lambda_embed=0.5):
    df = load_squad_data()
    optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()) +
                             list(get_proj_in().parameters()) + list(proj_out.parameters()), lr=0.001)
    mse_criterion = nn.MSELoss()
    ce_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100)
    model.train()
    decoder.train()
    with tqdm(total=num_epochs, desc="Free Text QA Training", unit="epoch") as pbar:
        for epoch in range(num_epochs):
            batch = df.sample(batch_size)
            questions = batch["question"].tolist()
            answers = batch["answer"].tolist()
            q_embeds = [ollama.embeddings(model="mistral:7b-instruct-q4_0", prompt=q)["embedding"] for q in questions]
            q_embeds = torch.tensor(q_embeds, dtype=torch.float32)
            q_embeds = q_embeds / torch.norm(q_embeds, dim=1, keepdim=True)
            inputs = get_proj_in()(q_embeds).to(torch.device("cpu"))
            student_out = model(inputs)
            student_out = student_out / student_out.norm(dim=1, keepdim=True)
            target_texts = [tokenizer.bos_token + a + tokenizer.eos_token for a in answers]
            target_encodings = tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True)
            target_ids = target_encodings["input_ids"].to(torch.device("cpu"))
            logits = decoder(student_out, target_ids[:, :-1])
            loss_ce = ce_criterion(logits.reshape(-1, logits.size(-1)), target_ids[:, 1:].reshape(-1))
            a_embeds = [ollama.embeddings(model="mistral:7b-instruct-q4_0", prompt=a)["embedding"] for a in answers]
            a_embeds = torch.tensor(a_embeds, dtype=torch.float32).to(torch.device("cpu"))
            a_embeds = a_embeds / torch.norm(a_embeds, dim=1, keepdim=True)
            teacher_answer_embeds = proj_out(a_embeds)
            teacher_answer_embeds = teacher_answer_embeds / teacher_answer_embeds.norm(dim=1, keepdim=True)
            loss_mse = mse_criterion(student_out, teacher_answer_embeds)
            loss = loss_ce + lambda_embed * loss_mse
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item(), "CE": loss_ce.item(), "MSE": loss_mse.item()})
            pbar.update(1)
    log_training("Free Text QA Fine-Tuning", num_epochs)
    print("Free Text QA fine-tuning completed.")

def train_rl_finetune(model, decoder, text_encoder, num_epochs=5, batch_size=16):
    """
    RL Fine-Tuning using REINFORCE.
    For each sample from SQuAD, we:
      - Encode the question with the text encoder.
      - Project and pass through the student model.
      - Use the decoder to generate a free-text response.
      - Compute a reward as the cosine similarity between the text encoder embeddings of
        the generated answer and the ground-truth answer.
      - Update the decoder parameters with a policy gradient loss.
    """
    df = load_squad_data()
    optimizer = optim.Adam(decoder.parameters(), lr=1e-5)
    for epoch in range(num_epochs):
        total_loss = 0.0
        batch = df.sample(batch_size)
        questions = batch["question"].tolist()
        ground_truths = batch["answer"].tolist()
        for q, gt in zip(questions, ground_truths):
            # Encode question using text encoder.
            encoding = tokenizer(q, return_tensors="pt")
            teacher_embed = text_encoder(encoding["input_ids"], attention_mask=encoding.get("attention_mask"))
            teacher_embed = teacher_embed.squeeze(0)
            teacher_embed = teacher_embed / teacher_embed.norm()
            proj_in_layer = get_proj_in()
            input_tensor = proj_in_layer(teacher_embed)
            with torch.no_grad():
                student_output = model(input_tensor.unsqueeze(0))
                student_output = student_output / student_output.norm(dim=1, keepdim=True)
            # Generate response from decoder.
            generated_ids = decoder.generate(student_output, device=torch.device("cpu"), temperature=1.0)
            # Compute log probabilities via teacher forcing.
            generated_ids_tensor = torch.tensor(generated_ids, dtype=torch.long)
            logits = decoder(student_output, generated_ids_tensor[:, :-1])
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            target_tokens = generated_ids_tensor[:, 1:]
            token_log_probs = log_probs.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)
            sequence_log_prob = token_log_probs.mean()
            # Decode generated sequence.
            gen_ids = generated_ids[0].tolist()
            if tokenizer.eos_token_id in gen_ids:
                gen_ids = gen_ids[:gen_ids.index(tokenizer.eos_token_id)]
            generated_text = tokenizer.decode(gen_ids, clean_up_tokenization_spaces=True)
            # Compute reward.
            reward = compute_reward(generated_text, gt, text_encoder)
            # REINFORCE loss.
            loss = -reward * sequence_log_prob
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / batch_size
        print(f"RL Epoch {epoch+1}/{num_epochs}, Avg RL Loss: {avg_loss:.4f}")
    log_training("RL Fine-Tuning", num_epochs)
    print("RL fine-tuning completed.")

def compute_reward(generated_text, ground_truth, text_encoder):
    # Encode both generated text and ground truth using the text encoder.
    gt_encoding = tokenizer(ground_truth, return_tensors="pt")
    gen_encoding = tokenizer(generated_text, return_tensors="pt")
    with torch.no_grad():
        gt_embed = text_encoder(gt_encoding["input_ids"]).squeeze(0)
        gen_embed = text_encoder(gen_encoding["input_ids"]).squeeze(0)
    gt_embed = gt_embed / gt_embed.norm()
    gen_embed = gen_embed / gen_embed.norm()
    cosine_sim = torch.dot(gt_embed, gen_embed)
    return cosine_sim.item()

# -----------------------------
# Logging Helpers for Training Modules
# -----------------------------
LOG_FILE = "training_log.txt"
BOOKS_TRAINED_FILE = "books_trained.txt"

def log_training(module_name, epochs):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp}: {module_name} completed ({epochs} epochs)\n"
    with open(LOG_FILE, "a") as f:
        f.write(log_entry)
    print(f"Logged training: {module_name}")

def get_unseen_books():
    os.makedirs("books", exist_ok=True)
    all_books = [os.path.join("books", f) for f in os.listdir("books") if f.endswith((".tex", ".txt"))]
    if os.path.exists(BOOKS_TRAINED_FILE):
        with open(BOOKS_TRAINED_FILE, "r") as f:
            seen = set(line.strip() for line in f.readlines())
    else:
        seen = set()
    unseen = [book for book in all_books if os.path.basename(book) not in seen]
    return unseen

def mark_book_as_trained(book_path):
    with open(BOOKS_TRAINED_FILE, "a") as f:
        f.write(os.path.basename(book_path) + "\n")
    print(f"Marked {os.path.basename(book_path)} as trained.")

# -----------------------------
# Main Menu
# -----------------------------
def main_menu():
    print("\nAstraea Learning - Training Modules:")
    print("1: SQuAD Fine-Tuning")
    print("2: Arithmetic Fine-Tuning")
    print("3: Math Book Fine-Tuning (all unseen books)")
    print("4: Free Text QA Fine-Tuning")
    print("5: RL Fine-Tuning")
    print("6: Exit")
    choice = input("Enter module number: ")
    return choice.strip()

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # Load student model checkpoint if available.
    student_model = AstraeaTernaryNet().to(torch.device("cpu"))
    checkpoint = "astraea_distilled.pth"
    if os.path.exists(checkpoint):
        try:
            student_model.load_state_dict(torch.load(checkpoint, map_location=torch.device("cpu")))
            student_model.eval()
            print(f"Loaded student model from {checkpoint}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    else:
        print("No checkpoint found; training will start from scratch.")
    # Main loop for training modules
    while True:
        choice = main_menu()
        if choice == "1":
            train_on_squad(student_model, num_epochs=5, batch_size=32)
        elif choice == "2":
            train_on_arithmetic(student_model, num_epochs=5, batch_size=32)
        elif choice == "3":
            train_on_all_unseen_books(student_model, num_epochs=5, batch_size=16)
        elif choice == "4":
            from transformers import AutoTokenizer  # ensure tokenizer is available
            train_free_text_qa(student_model, load_decoder(), num_epochs=5, batch_size=32, lambda_embed=0.5)
        elif choice == "5":
            # RL Fine-Tuning
            train_rl_finetune(student_model, load_decoder(), load_text_encoder(), num_epochs=5, batch_size=16)
        elif choice == "6":
            print("Exiting training module.")
            break
        else:
            print("Invalid choice. Try again.")
        cont = input("Do you want to run another training module? (y/n): ").strip().lower()
        if cont != "y":
            break
    print("Training session completed.")
