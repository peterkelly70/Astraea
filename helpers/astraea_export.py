# File: export_astraea.py
"""
Export Astraea Model Script

This script loads the trained AstraeaTernaryNet model from a checkpoint,
exports it to ONNX format, creates a manifest JSON file with model metadata,
and packages these files along with the required GPT-2 tokenizer files.
This package can then be used for deployment (e.g., with Ollama).
"""

import os
import json
import torch
import torch.nn as nn
import shutil

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
# Model Loading Function
# -----------------------------
def load_model(checkpoint_path="astraea_distilled.pth", device="cpu"):
    model = AstraeaTernaryNet().to(device)
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Loaded model from {checkpoint_path}")
    else:
        print("Checkpoint not found. Exiting.")
        exit(1)
    return model

# -----------------------------
# ONNX Export Function
# -----------------------------
def export_to_onnx(model, onnx_filepath, device="cpu"):
    # Create a dummy input matching the model's expected input (batch size 1, 1024-dim)
    dummy_input = torch.randn(1, 1024, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_filepath,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11
    )
    print(f"Model exported to {onnx_filepath}")

# -----------------------------
# Manifest Creation Function
# -----------------------------
def create_manifest(manifest_filepath):
    manifest = {
        "model_name": "AstraeaTernaryNet",
        "version": "1.0",
        "description": "A distilled ternary network for QA and arithmetic tasks.",
        "input": {
            "name": "input",
            "shape": [None, 1024],
            "dtype": "float32"
        },
        "output": {
            "name": "output",
            "shape": [None, 32],
            "dtype": "float32"
        },
        "dependencies": {
            "framework": "PyTorch",
            "opset_version": 11
        }
    }
    with open(manifest_filepath, "w") as f:
        json.dump(manifest, f, indent=4)
    print(f"Manifest written to {manifest_filepath}")

# -----------------------------
# Copy Tokenizer Files Function
# -----------------------------
def copy_tokenizer_files(destination_dir):
    # Default Hugging Face cache for transformers on Linux/macOS
    source_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "transformers")
    required_files = ["tokenizer_config.json", "config.json", "vocab.json", "merges.txt", "tokenizer.json"]
    os.makedirs(destination_dir, exist_ok=True)
    for filename in required_files:
        src_file = os.path.join(source_dir, filename)
        if os.path.exists(src_file):
            shutil.copy(src_file, destination_dir)
            print(f"Copied {filename} to {destination_dir}")
        else:
            print(f"Warning: {filename} not found in {source_dir}")

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # Set up package directory
    package_dir = "astraea_model"
    os.makedirs(package_dir, exist_ok=True)
    
    # Define file paths in package
    onnx_filepath = os.path.join(package_dir, "astraea.onnx")
    manifest_filepath = os.path.join(package_dir, "manifest.json")
    tokenizer_dir = os.path.join(package_dir, "tokenizer")
    
    device = "cpu"  # Change to "cuda" if desired and available
    model = load_model("astraea_distilled.pth", device=device)
    
    # Export the model to ONNX and create manifest in the package directory
    export_to_onnx(model, onnx_filepath, device=device)
    create_manifest(manifest_filepath)
    
    # Copy the required GPT-2 tokenizer files to the package's tokenizer subdirectory
    copy_tokenizer_files(tokenizer_dir)
    
    print("Export complete. Your package directory 'astraea_model' now contains the model, manifest, and tokenizer files.")
