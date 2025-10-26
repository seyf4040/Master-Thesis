#!/usr/bin/env python3
"""
Llama Guard 3 Inference on ToxiGen Dataset
Author: Ural Seyfullah
Date: October 2025
Purpose: Evaluate Llama Guard 3 performance on toxic content detection
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
import time
from datetime import datetime
import os

def load_toxigen_data(data_path, max_samples=None):
    """
    Load ToxiGen dataset from JSONL file
    
    Args:
        data_path: Path to the JSONL file
        max_samples: Maximum number of samples to load (for testing)
    
    Returns:
        List of dictionaries containing text and metadata
    """
    samples = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            samples.append(json.loads(line))
    
    print(f"Loaded {len(samples)} samples from {data_path}")
    return samples


def initialize_llama_guard(model_name="meta-llama/Llama-Guard-3-8B"):
    """
    Initialize Llama Guard 3 model and tokenizer
    
    Args:
        model_name: HuggingFace model identifier
    
    Returns:
        model, tokenizer
    """
    print(f"Loading {model_name}...")
    
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with appropriate dtype for GPU efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    
    print(f"Model loaded successfully on {device}")
    return model, tokenizer, device


def format_prompt_for_guard(text):
    """
    Format text according to Llama Guard 3 expected input format
    
    Args:
        text: Raw text to moderate
    
    Returns:
        Formatted prompt string
    """
    # Llama Guard expects a specific format
    # Format: <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{text}<|eot_id|>
    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt


def run_inference(model, tokenizer, samples, device, batch_size=8):
    """
    Run Llama Guard inference on samples
    
    Args:
        model: Loaded Llama Guard model
        tokenizer: Loaded tokenizer
        samples: List of samples to process
        device: Device to run inference on
        batch_size: Batch size for inference
    
    Returns:
        List of results with predictions
    """
    results = []
    model.eval()
    
    print(f"\nRunning inference on {len(samples)} samples...")
    start_time = time.time()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(samples), batch_size)):
            batch_samples = samples[i:i + batch_size]
            
            # Prepare batch
            texts = [sample.get('text', '') for sample in batch_samples]
            prompts = [format_prompt_for_guard(text) for text in texts]
            
            # Tokenize
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            
            # Generate predictions
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode outputs
            for j, output in enumerate(outputs):
                decoded = tokenizer.decode(output, skip_special_tokens=True)
                
                # Extract the classification (Llama Guard outputs "safe" or "unsafe")
                prediction = "unsafe" if "unsafe" in decoded.lower() else "safe"
                
                results.append({
                    'text': texts[j],
                    'prediction': prediction,
                    'full_output': decoded,
                    'ground_truth': batch_samples[j].get('toxicity_ai', None),
                    'target_group': batch_samples[j].get('target_group', None)
                })
    
    elapsed = time.time() - start_time
    print(f"\nInference completed in {elapsed:.2f} seconds")
    print(f"Average time per sample: {elapsed/len(samples):.3f} seconds")
    
    return results


def calculate_metrics(results):
    """
    Calculate performance metrics
    
    Args:
        results: List of result dictionaries
    
    Returns:
        Dictionary containing metrics
    """
    # Count predictions
    unsafe_predictions = sum(1 for r in results if r['prediction'] == 'unsafe')
    safe_predictions = len(results) - unsafe_predictions
    
    # If ground truth available, calculate accuracy
    has_ground_truth = any(r.get('ground_truth') is not None for r in results)
    
    metrics = {
        'total_samples': len(results),
        'unsafe_predictions': unsafe_predictions,
        'safe_predictions': safe_predictions,
        'unsafe_rate': unsafe_predictions / len(results) if results else 0
    }
    
    if has_ground_truth:
        correct = sum(
            1 for r in results 
            if r.get('ground_truth') is not None and 
            ((r['prediction'] == 'unsafe' and r['ground_truth'] > 0.5) or 
             (r['prediction'] == 'safe' and r['ground_truth'] <= 0.5))
        )
        metrics['accuracy'] = correct / len(results)
    
    return metrics


def save_results(results, metrics, output_dir):
    """
    Save results and metrics to files
    
    Args:
        results: List of prediction results
        metrics: Dictionary of performance metrics
        output_dir: Directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_path = os.path.join(output_dir, f"predictions_{timestamp}.jsonl")
    with open(results_path, 'w') as f:
        for result in results:
            json.dump(result, f)
            f.write('\n')
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f"metrics_{timestamp}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Results saved to {results_path}")
    print(f"✅ Metrics saved to {metrics_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(
        description="Run Llama Guard 3 inference on ToxiGen dataset"
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to ToxiGen JSONL file'
    ) 
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='meta-llama/Llama-Guard-3-8B',
        help='HuggingFace model identifier'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (for testing)'
    )
    
    args = parser.parse_args()
    
    print("="*50)
    print("Llama Guard 3 on ToxiGen - Inference Pipeline")
    print("="*50)
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    print("="*50 + "\n")
    
    # Load data
    samples = load_toxigen_data(args.data_path, args.max_samples)
    
    # Initialize model
    model, tokenizer, device = initialize_llama_guard(args.model_name)
    
    # Run inference
    results = run_inference(model, tokenizer, samples, device, args.batch_size)
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Save results
    save_results(results, metrics, args.output_dir)
    
    print("\n✅ Pipeline completed successfully!")


if __name__ == "__main__":
    main()
