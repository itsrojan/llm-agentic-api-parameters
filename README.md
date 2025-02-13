# llm-agentic-api-parameters
Assignment: Exploring LLM API Parameters in Python

Objective:

In this assignment, you will write a Python script to interact with a Large Language Model (LLM) using an API (such as OpenAI's GPT or other models). You will experiment with different parameters that influence the probability of next-token prediction and analyze their impacts on the model’s responses.

Key Concepts to Explore:

Temperature:How randomness affects responses
Top-p (Nucleus) Sampling:Probabilistic filtering
Frequency and Presence Penalties:Biasing token selection
Max Tokens:Controlling response length
Logprobs:Analyzing token probabilities

To Use:

Create .env file:

```env
OPENAI_API_KEY=your_api_key_here
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the script:
```bash
python llm_parameter_experiment.py
```
