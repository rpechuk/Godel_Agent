#!/bin/bash

# Load the model into Ollama
ollama create gpt-4o -f "./modelfile"
# Load the model into Ollama
ollama create gpt-4o-mini -f "./modelfile"
# Load the model into Ollama
ollama pull llama3.2
ollama cp llama3.2 gpt-3.5-turbo
echo "Model creation complete"
# Load the embedding into Ollama
ollama pull nomic-embed-text
echo "Embedding creation complete"
# List the models in Ollama
ollama list