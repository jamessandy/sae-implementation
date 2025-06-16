# Sparse Autoencoder for LLM Feature Interpretability

This repository contains an independent implementation of Sparse Autoencoders (SAEs) designed to extract interpretable features from the internal activations of large language models (LLMs). This project is inspired by the paper "[A. Name et al., 'Sparse Autoencoders Find Highly Interpretable Features in Language Models,' Year.]" (Remember to replace this with the actual citation of the paper you are implementing).

The goal of this implementation is to combat **polysemanticity** (when individual neurons activate in different unrelated contexts) in LLMs by using SAEs to decompose complex activations into simpler, more **monosemantic** features.

## Key Features

* **Activation Extraction:** Extracts hidden state activations from a pre-trained GPT-2 model.
* **Sparse Autoencoder (SAE) Training:** Implements and trains a simple sparse autoencoder with an L1 sparsity penalty.
* **Weights & Biases Integration:** Logs training progress and artifacts to Weights & Biases for experiment tracking.
* **Feature Analysis:** Identifies and displays **Maximum Activating Examples (MAEs)** for learned features, aiding in their interpretability.

## Implementation Details

* **LLM Used:** GPT-2 (`gpt2`)
* **Dataset:** OpenWebText (a sample of 20,000 texts for SAE training, and 5,000 for MAE analysis)
* **SAE Architecture:**
    * **Encoder:** `Linear` layer followed by `ReLU` activation.
    * **Decoder:** `Linear` layer using the transpose of the encoder's weight matrix, with an independent bias.
    * **Overcompleteness:** The hidden dimension of the SAE is 8 times the input activation dimension.
* **Loss Function:** Mean Squared Error (MSE) for reconstruction loss combined with L1 norm for sparsity loss.
* **Optimizer:** Adam
* **Key Hyperparameters:**
    * Learning Rate: `1e-3`
    * Sparsity Coefficient ($\lambda$): `1e-3`
    * Batch Size: `512`
    * Epochs: `100`

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed.

### Installation

1.  Clone this repository:
    ```bash
    git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)
    cd your-repo-name
    ```
    (Replace `yourusername` and `your-repo-name` with your actual GitHub details if you move this to a full repository.)

2.  Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3.  Set your Weights & Biases API key. You can find this on your Weights & Biases profile page.
    ```bash
    export WANDB_API_KEY='YOUR_WANDB_API_KEY'
    # Or set it directly in your sae.py script as you currently have (os.environ['WANDB_API_KEY'] = 'e')
    # For security, storing it as an environment variable is generally preferred for public repos.
    ```

### Running the Experiment

To run the full experiment, including data loading, activation extraction, SAE training, and feature analysis:

```bash
python sae.py
