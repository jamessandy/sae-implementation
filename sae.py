import os
import json
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from transformers import GPT2Model, GPT2Tokenizer
from datasets import load_dataset

from tqdm import tqdm
import wandb


# Global device setting 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    try:
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except Exception as e:
        print(f"Could not get GPU memory details: {e}")


print("\n Logging in to Weights & Biases")
os.environ['WANDB_API_KEY'] = ''
wandb.login()
print("\n Environment Setup Complete ")



class DataManager:
    def __init__(self, dataset_name="openwebtext", max_samples=50000):
        self.dataset_name = dataset_name
        self.max_samples = max_samples
        self.texts = []
    def load_openwebtext_sample(self):
        print(f"Loading OpenWebText dataset (max {self.max_samples} samples)...")
        try:
            dataset = load_dataset(self.dataset_name, split="train", streaming=True)
            texts = []
            for i, example in enumerate(dataset):
                if i >= self.max_samples:
                    break
                text = example['text'].strip()
                if len(text) > 100:
                    texts.append(text)
                if (i + 1) % 10000 == 0:
                    print(f"Loaded {i+1} samples...")
            self.texts = texts[:self.max_samples]
            print(f"Successfully loaded {len(self.texts)} text samples.")
        except Exception as e:
            print(f"Error loading OpenWebText: {e}")
            print("Falling back to dummy data...")
            self.create_dummy_dataset()
    def create_dummy_dataset(self):
        print("Creating dummy dataset...")
        dummy_texts = [
            "The quick brown fox jumps over the lazy dog. This is a placeholder sentence for testing purposes." for _ in range(20)
        ]
        self.texts = [item for sublist in [dummy_texts] * (self.max_samples // 20 + 1) for item in sublist][:self.max_samples]
        print(f"Created {len(self.texts)} dummy text samples")

class GPT2ActivationExtractor:
    def __init__(self, model_name="gpt2"):
        print(f"\nLoading {model_name} model and tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = self.model.to(device)
        self.model.eval()
        print(f"Model loaded. Hidden size: {self.model.config.hidden_size}")
    def extract_activations(self, texts: List[str], layer_idx: int, max_length: int, batch_size: int) -> np.ndarray:
        print(f"\nExtracting activations from layer {layer_idx}...")
        all_activations = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Extracting Activations"):
                batch_texts = texts[i:i+batch_size]
                inputs = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs, output_hidden_states=True)
                layer_activations = outputs.hidden_states[layer_idx]
                attention_mask = inputs['attention_mask']

                for j in range(layer_activations.shape[0]):
                    valid_mask = attention_mask[j].bool()
                    valid_activations = layer_activations[j][valid_mask]
                    all_activations.append(valid_activations.cpu().numpy())
        result = np.vstack(all_activations)
        print(f"Extracted {result.shape[0]} activation vectors of dimension {result.shape[1]}")
        return result

class SimpleSAE(nn.Module):
    def __init__(self, input_dim: int, expansion_factor: int, sparsity_coeff: float):
        super().__init__()
        self.hidden_dim = input_dim * expansion_factor
        self.sparsity_coeff = sparsity_coeff
        self.encoder = nn.Linear(input_dim, self.hidden_dim)
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity='relu')
        nn.init.zeros_(self.encoder.bias)
    def encode(self, x):
        return F.relu(self.encoder(x))
    def decode(self, h):
        return F.linear(h, self.encoder.weight.t(), self.decoder_bias)
    def forward(self, x):
        h = self.encode(x)
        x_recon = self.decode(h)
        return x_recon, h
    def compute_loss(self, x, x_recon, h):
        recon_loss = F.mse_loss(x_recon, x)
        sparsity_loss = torch.mean(torch.abs(h))
        total_loss = recon_loss + self.sparsity_coeff * sparsity_loss
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'sparsity_loss': sparsity_loss,
            'fraction_active': torch.mean((h > 0).float())
        }

def train_sae(activations: np.ndarray, config: Dict):
    print("\n Setting up SAE training")
    activations_tensor = torch.FloatTensor(activations)
    train_size = int(0.9 * len(activations_tensor))
    val_size = len(activations_tensor) - train_size
    train_data, val_data = random_split(activations_tensor, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], pin_memory=True)
    
    input_dim = activations.shape[1]
    sae = SimpleSAE(input_dim, config['expansion_factor'], config['sparsity_coeff']).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=config['lr'])
    model_path = "sae_gpt2.pt"

    print(f"Starting W&B run for SAE Training...")
    with wandb.init(project="sae-gpt2-successful-run", config=config) as run:
        print(f"Training SAE: Input Dim={input_dim} -> Hidden Dim={sae.hidden_dim}")
        print(f"Total Parameters: {sum(p.numel() for p in sae.parameters()):,}")

        for epoch in range(config['epochs']):
            sae.train()
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
                batch = batch.to(device)
                x_recon, h = sae(batch)
                losses = sae.compute_loss(batch, x_recon, h)
                
                optimizer.zero_grad()
                losses['total_loss'].backward()
                optimizer.step()
            
            sae.eval()
            val_losses_total, val_losses_recon, val_sparsities = [], [], []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                    batch = batch.to(device)
                    x_recon, h = sae(batch)
                    losses = sae.compute_loss(batch, x_recon, h)
                    val_losses_total.append(losses['total_loss'].item())
                    val_losses_recon.append(losses['recon_loss'].item())
                    val_sparsities.append(losses['fraction_active'].item())
            
            avg_val_loss = np.mean(val_losses_total)
            avg_recon_loss = np.mean(val_losses_recon)
            avg_sparsity = np.mean(val_sparsities)

            run.log({
                "epoch": epoch + 1,
                "val_loss": avg_val_loss,
                "recon_loss": avg_recon_loss,
                "sparsity": avg_sparsity
            })
            
            if (epoch + 1) % 10 == 0 or (epoch + 1) == config['epochs']:
                print(f"Epoch {epoch+1:3d}: Val Loss={avg_val_loss:.4f}, Recon Loss={avg_recon_loss:.4f}, Sparsity={avg_sparsity:.4f}")

        print("\nLogging final model artifact to W&B...")
        model_artifact = wandb.Artifact("sae_model", type="model", metadata=config)
        torch.save(sae.state_dict(), model_path)
            
        model_artifact.add_file(model_path)
        run.log_artifact(model_artifact)
        print(f"Final model saved to {model_path} and logged to W&B.")
        
    return sae


def find_and_visualize_maes(sae_model: 'SimpleSAE', data_manager: 'DataManager', extractor: 'GPT2ActivationExtractor', config: Dict, num_top_examples: int = 5):
    print(f"\n--- Finding Maximum Activating Examples (MAEs) for {num_top_examples} features ---")
    sae_model.eval() 
    
    
    top_examples_per_feature = {i: [] for i in range(sae_model.hidden_dim)}
    
    mae_batch_size = config.get('mae_batch_size', 32)

    total_tokens_processed = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(data_manager.texts), mae_batch_size), desc="Processing texts for MAEs"):
            batch_texts = data_manager.texts[i : i + mae_batch_size]
            
            inputs = extractor.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=config['max_length'])
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = extractor.model(**inputs, output_hidden_states=True)
            layer_activations = outputs.hidden_states[config['layer_idx']]
            attention_mask = inputs['attention_mask']

            valid_mask = attention_mask.bool()
            flat_activations = layer_activations[valid_mask]

            if flat_activations.numel() == 0:
                continue

            h = sae_model.encode(flat_activations)

            batch_tokenized_inputs = extractor.tokenizer(batch_texts, add_special_tokens=False, padding=False, truncation=True, max_length=config['max_length'])
            
            token_idx_in_flat_activations = 0
            for text_idx_in_batch in range(len(batch_texts)):
                current_token_ids = batch_tokenized_inputs.input_ids[text_idx_in_batch]
                current_attention_mask = attention_mask[text_idx_in_batch]

                for token_local_idx in range(len(current_token_ids)):
                    if current_attention_mask[token_local_idx].item() == 1:
                        if token_idx_in_flat_activations >= h.shape[0]:
                            break

                        token_activation_vector = h[token_idx_in_flat_activations, :].cpu().numpy()
                        
                        start_context_idx = max(0, token_local_idx - 5)
                        end_context_idx = min(len(current_token_ids), token_local_idx + 6)
                        context_token_ids = current_token_ids[start_context_idx:end_context_idx]
                        context_text = extractor.tokenizer.decode(context_token_ids, skip_special_tokens=True)
                        
                        for feature_idx in range(sae_model.hidden_dim):
                            feature_activation = token_activation_vector[feature_idx]
                            
                            if feature_activation > 0:
                                top_examples_per_feature[feature_idx].append((feature_activation, context_text))
                        
                        token_idx_in_flat_activations += 1
            
            total_tokens_processed += flat_activations.shape[0]

            
            if total_tokens_processed % pruning_interval == 0 and total_tokens_processed > 0:
                print(f"Pruning MAE examples to conserve memory at {total_tokens_processed} tokens...")
                for fid in range(sae_model.hidden_dim):
                    # Keep num_top_examples * 2 as a buffer to ensure enough candidates for final selection
                    top_examples_per_feature[fid] = sorted(top_examples_per_feature[fid], key=lambda x: x[0], reverse=True)[:num_top_examples * 2]


    print(f"Processed {total_tokens_processed} tokens for MAE extraction.")

  
    feature_total_activations = {fid: sum(item[0] for item in examples) for fid, examples in top_examples_per_feature.items()}
    sorted_features_by_total_activation = sorted(feature_total_activations.items(), key=lambda item: item[1], reverse=True)
    num_features_to_display = min(20, sae_model.hidden_dim)
    
    print(f"\nDisplaying MAEs for the top {num_features_to_display} overall most active features:")
    for i, (feature_idx, total_act) in enumerate(sorted_features_by_total_activation[:num_features_to_display]):
        if feature_idx not in top_examples_per_feature or not top_examples_per_feature[feature_idx]:
            continue

        print(f"\n--- Feature {feature_idx} (Total Activation: {total_act:.2f}) ---")
        sorted_examples = sorted(top_examples_per_feature[feature_idx], key=lambda x: x[0], reverse=True)
        
        for k, (activation_value, text_segment) in enumerate(sorted_examples[:num_top_examples]):
            print(f"  {k+1}. (Act: {activation_value:.4f}) -> \"{text_segment}\"")

    print("\n MAE Analysis Complete ")



def main():
    print("Starting SAE Experiment")
    
    config = {
        'dataset_samples': 20000,
        'layer_idx': 6,
        'expansion_factor': 8,
        'sparsity_coeff': 1e-3,
        'batch_size': 512,
        'lr': 1e-3,
        'epochs': 100,
        'max_length': 128,
        'model_name': 'gpt2',
        'mae_batch_size': 32,
        'num_mae_examples': 5
    }
    
    print("\n Experiment Configuration ")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    data_manager = DataManager(max_samples=config['dataset_samples'])
    data_manager.load_openwebtext_sample()
    
    extractor = GPT2ActivationExtractor(config['model_name'])
    
    activations_for_training = extractor.extract_activations(
        texts=data_manager.texts,
        layer_idx=config['layer_idx'],
        max_length=config['max_length'],
        batch_size=32
    )
    
    sae = train_sae(activations_for_training, config)
    
    
    
    del activations_for_training
    
    torch.cuda.empty_cache() 
    

    
    find_and_visualize_maes(
        sae_model=sae,
        data_manager=data_manager,
        extractor=extractor,
        config=config,
        num_top_examples=config['num_mae_examples']
    )
    
    print("\n Experiment complete and results logged to W&B.")

if __name__ == "__main__":
    main()
