import os
import json
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI #openai latest version

load_dotenv()  # Loads variables from .env

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_response(prompt, params):
    """Generate LLM response with specified parameters"""
    try:
        # Filter out unsupported parameters
        supported_params = {
            k: v for k, v in params.items()
            if k in ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty", "logprobs"]
        }
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",  # Use the correct model
            prompt=prompt,
            **supported_params
        )
        return response
    except Exception as e:
        print(f"API Error: {e}")
        return None

def run_experiments(prompt, configs, max_retries=3):
    """Run experiments with exponential backoff"""
    results = []
    for config in configs:
        for attempt in range(max_retries):
            config_to_send = dict(config)  # make a copy

            response = generate_response(prompt, config_to_send)
            if response:
                result = {
                    "params": config,  # store original config
                    "response": response.choices[0].text.strip()
                }
                # If the original config explicitly requested logprobs, include it in the result.
                if 'logprobs' in config:
                    lp_data = (response.choices[0].logprobs.top_logprobs 
                               if hasattr(response.choices[0].logprobs, 'top_logprobs') else None)
                    if lp_data is not None:
                        result["logprobs"] = lp_data
                results.append(result)
                break
            time.sleep(2 ** attempt)
    return results

def visualize_logprobs(result, max_tokens_to_plot=10):
    """
    Visualize the top 5 log probabilities for each token in the generated response.
    Only the first `max_tokens_to_plot` tokens are visualized for clarity.
    The full generated output is included in the image title.
    """
    if "logprobs" not in result:
        print("No logprobs data available for visualization.")
        return

    logprobs_data = result["logprobs"]
    response_text = result["response"]
    tokens_to_plot = logprobs_data[:max_tokens_to_plot]

    num_tokens = len(tokens_to_plot)
    fig, axs = plt.subplots(num_tokens, 1, figsize=(10, 3 * num_tokens))
    
    if num_tokens == 1:
        axs = [axs]  # ensure axs is iterable

    # For each token position, plot a bar chart of the top 5 candidate tokens
    for idx, token_dict in enumerate(tokens_to_plot):
        # Sort candidate tokens by log probability (descending)
        sorted_candidates = sorted(token_dict.items(), key=lambda x: x[1], reverse=True)
        top_candidates = sorted_candidates[:5]
        candidates, log_probs = zip(*top_candidates)
        
        axs[idx].bar(candidates, log_probs, color='skyblue')
        axs[idx].set_title(f"Token {idx+1}")
        axs[idx].set_ylabel("Log Probability")
        axs[idx].set_xlabel("Candidate Token")
    
    plt.suptitle("Logprobs Visualization\nGenerated Output:\n" + response_text, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig("logprobs_analysis.png", bbox_inches="tight")
    plt.close()
    print("Saved logprobs visualization as logprobs_analysis.png")

def main():
    # Experiment Configuration
    BASE_PROMPT = "Explain the importance of artificial intelligence in healthcare:\n\n"
    PARAM_CONFIGS = (
        [{'temperature': t} for t in [0.2, 0.7, 1]] +
        [{'top_p': p} for p in [0.3, 0.5, 0.9]] +
        [{'frequency_penalty': 0.5, 'presence_penalty': 0.5},
         {'frequency_penalty': 1, 'presence_penalty': 1},
         {'frequency_penalty': 2.0, 'presence_penalty': 2.0}] +
        [{'max_tokens': mt} for mt in [50, 100, 150]] +
        [{'logprobs': 5}]
    )

    # Run experiments
    results = run_experiments(BASE_PROMPT, PARAM_CONFIGS)
    
    # Save raw results (only include keys that have data)
    clean_results = []
    for res in results:
        clean_res = {
            "params": res["params"],
            "response": res["response"]
        }
        # Only include logprobs if originally requested
        if "logprobs" in res:
            clean_res["logprobs"] = res["logprobs"]
        clean_results.append(clean_res)
        
    with open('experiment_results.json', 'w') as f:
        json.dump(clean_results, f, indent=2)
    print("Saved experiment results to experiment_results.json")
    
    # Visualization: Only visualize logprobs experiments.
    for res in results:
        if "logprobs" in res:
            visualize_logprobs(res)

if __name__ == "__main__":
    main()
