# llm-agentic-api-parameters
Assignment: Exploring LLM API Parameters in Python

## Objective:

In this assignment, you will write a Python script to interact with a Large Language Model (LLM) using an API (such as OpenAI's GPT or other models). You will experiment with different parameters that influence the probability of next-token prediction and analyze their impacts on the model’s responses.

## Key Concepts to Explore:

Temperature:How randomness affects responses

Top-p (Nucleus) Sampling:Probabilistic filtering

Frequency and Presence Penalties:Biasing token selection

Max Tokens:Controlling response length

Logprobs:Analyzing token probabilities

## To Use:

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

## Takeaway:

This project examines how different AI parameters influence generated responses, demonstrated through examples about AI in healthcare. Each parameter tweaks the output in unique ways:

- **Temperature**: Controls randomness. At 0.2, responses are tight and repetitive ("Improved Diagnosis and Treatment: AI can analyze vast amounts"). At 0.7, they gain slight variety ("Improved diagnostic accuracy"), and at 1.0, they broaden ("Diagnosis and treatment: AI can analyze medical data"), trading focus for exploration.
  
- **Top_p**: Filters word choices by probability. At 0.3, it’s conservative ("Improved Diagnosis and Treatment"), mirroring low temperature. At 0.5, it shifts focus ("Improving patient outcomes"), and at 0.9, it mixes specificity with diversity ("Improved diagnostic accuracy: AI can analyze medical images").

- **Frequency & Presence Penalty**: Reduces repetition. With 0.5 each, output avoids overused terms ("Accurate Diagnoses: AI has the ability"). At 1.0 each, it’s "Improved medical diagnosis and treatment," and at 2.0 each, it’s fresher ("Examining and interpreting medical data"), prioritizing novelty.

- **Max_tokens**: Sets length. At 50 tokens, it’s one concise point ("Improved Diagnostic Capabilities: AI-based algorithms…"). At 100, it covers two ("Personalized treatment plans" added), and at 150, it expands to four, including "Management of chronic diseases" and "Patient assistance," offering depth.

- **Logprobs**: Reveals decision-making. For "Efficient and Accurate Diagnosis," log probabilities show high confidence in "Accurate" (-0.29) over "Faster" (-7.03), giving insight into word choice priorities.

### Conclusion

Parameters shape AI outputs: low temperature/top_p for precision, high penalties for originality, and more tokens for elaboration. 

For healthcare contexts needing clarity, moderate settings (e.g., temperature 0.7, max_tokens 100) with slight penalties balance accuracy and insight. 

Adjust based on your goals—focus, creativity, or completeness.
