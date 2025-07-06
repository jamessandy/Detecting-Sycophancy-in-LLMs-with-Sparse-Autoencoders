# Detecting-Sycophancy-in-LLMs-with-Sparse-Autoencoders

This repository contains the code and resources for a mechanistic interpretability project aimed at identifying and analyzing features related to sycophantic behavior in the GPT-2 language model.

## Abstract

This work uses a Sparse Autoencoder (SAE) to find interpretable features in a 124M parameter GPT-2 model. We identify a specific feature that causally governs the model's sycophantic responses. Through direct interventions (causal ablation and boosting), we demonstrate control over this abstract social behavior, providing a proof-of-concept for dissecting and editing high-level properties of LLMs.

## How to Run

1.  **Clone the repository:**
   ```bash
git clone https://github.com/jamessandy/Detecting-Sycophancy-in-LLMs-with-Sparse-Autoencoders.git
cd Detecting-Sycophancy-in-LLMs-with-Sparse-Autoencoders
```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the experiment:**
    ```bash
    python main.py
    ```
    You will be prompted to enter your own Weights & Biases API key.

## Results

Our key finding was the identification of Feature #744, which strongly correlates with sycophantic behavior. Intervening on this feature allowed us to suppress or amplify sycophancy in the model's outputs.

