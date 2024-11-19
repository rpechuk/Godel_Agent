#!/bin/bash

# Load the model into Ollama
ollama create gpt-4o -f "./gpt-4o-Modelfile"
# Load the model into Ollama
ollama create gpt-4o-mini -f "./gpt-4o-Modelfile"
# Load the model into Ollama
ollama pull llama3.2
ollama cp llama3.2 gpt-3.5-turbo
# List the models in Ollama
ollama list