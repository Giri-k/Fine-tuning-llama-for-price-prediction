# Fine-Tuning Llama 3.1 8B for Price Prediction

This repository contains the code and resources for fine-tuning the
**Llama 3.1 8B** model for price prediction.\
The goal of this project is to accurately predict the price of a product
based on its description, using a quantized model for efficient
performance.

------------------------------------------------------------------------

## Key Features 🚀

-   **Efficient Fine-Tuning**: The Llama 3.1 8B model is fine-tuned
    using **QLoRA with PEFT (Parameter-Efficient Fine-Tuning)**, a
    technique that drastically reduces the computational resources
    required for training.\
-   **Model Quantization**: The model is quantized to **4-bit**,
    significantly reducing the model size from **32.1 GB to 4.9 GB**.\
-   **High Performance**: The fine-tuned model achieves a **prediction
    error of only \$85**, a significant improvement from the base
    model's error of **\$360** on the same task.\
-   **Compact Adapter**: The resulting PEFT adapter is only **27.3 MB**,
    making the fine-tuned model lightweight, portable, and easy to
    deploy.

------------------------------------------------------------------------

## Model & Data Details 📦

-   **Base Model**: Llama 3.1 8B
-   **Fine-Tuning Technique**: QLoRA with PEFT
-   **Quantization**: 4-bit
-   **Dataset**: Processed version of Amazon product data, available on
    Hugging Face at
    [`ed-donner/pricer-data`](https://huggingface.co/ed-donner/pricer-data).

------------------------------------------------------------------------

## Performance 📊

The fine-tuned model's performance was evaluated based on the
**prediction error**, which is the absolute difference between the
predicted price and the actual price.

  Model                     Prediction Error
  ------------------------- ------------------
  Base Llama 3.1 8B - $360
  
  Fine-Tuned Llama 3.1 8B - $85

The fine-tuned model not only shows a **massive improvement** over the
base model but also **outperforms several larger frontier models** in
terms of accuracy on this specific task.

------------------------------------------------------------------------

## Usage Guide

### Model Training

- Open `Fine_tuning_llama.ipynb`.
- Load and preprocess the dataset.
- Configure model and training hyperparameters.
- Run training cells to fine-tune the LLaMA model.
- Save the fine-tuned model checkpoint.

### Evaluation

- Open `Testing_the_finetuned_model.ipynb`.
- Load the saved checkpoint.
- Run predictions on test data.
- Evaluate with metrics such as MAE, RMSE, and MAPE.
- Visualize predictions vs. actual values.

------------------------------------------------------------------------

