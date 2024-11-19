import os
import ast
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import openai

@dataclass
class CodeContext:
    file_path: str
    content: str
    imports: List[str]
    classes: List[str]
    functions: List[str]
    variables: List[str]
    metadata: Dict
    embedding: Optional[List[float]] = None

class StructuredRAG:
    def __init__(self, base_path: str = "."):
        try:
            self.client = openai.OpenAI(api_key='ollama', base_url="http://localhost:11434/v1")
            # Test connection
            self.get_embedding("test")
            print("Successfully connected to Ollama embedding service")
        except Exception as e:
            print(f"Warning: Could not connect to Ollama: {e}")
            print("Will use fallback simple embeddings")
            self.client = None
        
        self.base_path = base_path
        self.contexts: Dict[str, CodeContext] = {}
        self.initialize_contexts()

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using local Ollama model with fallback"""
        try:
            response = self.client.embeddings.create(
                model="nomic-embed-text",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Warning: Embedding service error: {e}")
            print("Using fallback simple embedding...")
            # Simple fallback embedding based on character frequency
            # Not as good as real embeddings but allows testing without Ollama
            char_freq = {}
            for char in text.lower():
                char_freq[char] = char_freq.get(char, 0) + 1
            # Create a simple 384-dimensional embedding (same as nomic-embed-text)
            simple_embedding = [0] * 384
            for i, char in enumerate(char_freq):
                if i < 384:
                    simple_embedding[i] = char_freq[char] / len(text)
            return simple_embedding

    def extract_code_elements(self, content: str) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Extract imports, classes, functions, and variables from code"""
        tree = ast.parse(content)
        imports = []
        classes = []
        functions = []
        variables = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                imports.append(f"from {node.module} import {', '.join(n.name for n in node.names)}")
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables.append(target.id)

        return imports, classes, functions, variables

    def get_file_metadata(self, file_path: str) -> Dict:
        """Extract metadata about the file"""
        return {
            "file_size": os.path.getsize(file_path),
            "last_modified": os.path.getmtime(file_path),
            "extension": os.path.splitext(file_path)[1],
            "directory": os.path.dirname(file_path),
            "is_test": "test" in file_path.lower(),
            "is_main": "main" in file_path.lower(),
        }

    def initialize_contexts(self):
        """Initialize code contexts for all relevant files"""
        for root, _, files in os.walk(self.base_path):
            for file in files:
                if file.endswith(('.py', '.md', '.json')):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if file.endswith('.py'):
                        imports, classes, functions, variables = self.extract_code_elements(content)
                    else:
                        imports, classes, functions, variables = [], [], [], []

                    metadata = self.get_file_metadata(file_path)
                    
                    # Create embedding from file content and metadata
                    embedding_text = f"""
                    File: {file_path}
                    Type: {metadata['extension']}
                    Classes: {', '.join(classes)}
                    Functions: {', '.join(functions)}
                    Imports: {', '.join(imports)}
                    Content: {content[:1000]}  # First 1000 chars for embedding, i think we need the whole file here
                    """
                    embedding = self.get_embedding(embedding_text)

                    self.contexts[file_path] = CodeContext(
                        file_path=file_path,
                        content=content,
                        imports=imports,
                        classes=classes,
                        functions=functions,
                        variables=variables,
                        metadata=metadata,
                        embedding=embedding
                    )

    def find_relevant_contexts(self, query: str, k: int = 3) -> List[CodeContext]:
        """Find k most relevant code contexts for a query"""
        query_embedding = self.get_embedding(query)
        
        # Calculate cosine similarities
        similarities = []
        for file_path, context in self.contexts.items():
            if context.embedding:
                similarity = np.dot(query_embedding, context.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(context.embedding)
                )
                similarities.append((similarity, context))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [context for _, context in similarities[:k]]

    def get_runtime_context(self) -> Dict:
        """Get current runtime context including loaded modules and global variables"""
        import sys
        import gc
        
        runtime_context = {
            "loaded_modules": list(sys.modules.keys()),
            "global_variables": {},
            "active_objects": []
        }

        # Get all objects in memory
        for obj in gc.get_objects():
            try:
                if hasattr(obj, '__class__') and hasattr(obj, '__dict__'):
                    obj_info = {
                        "type": type(obj).__name__,
                        "attributes": list(obj.__dict__.keys())
                    }
                    runtime_context["active_objects"].append(obj_info)
            except:
                continue

        return runtime_context

    def augment_query(self, query: str, k: int = 3) -> str:
        """Augment query with relevant code contexts and runtime information"""
        relevant_contexts = self.find_relevant_contexts(query, k)
        runtime_context = self.get_runtime_context()
        
        augmented_query = f"""Query: {query}

        Runtime Context:
        {json.dumps(runtime_context, indent=2)}

        Relevant Code Contexts:
        """

        for ctx in relevant_contexts:
            augmented_query += f"""
        File: {ctx.file_path}
        Classes: {', '.join(ctx.classes)}
        Functions: {', '.join(ctx.functions)}
        Imports: {', '.join(ctx.imports)}
        Metadata: {json.dumps(ctx.metadata, indent=2)}

        Relevant Content:
        {ctx.content[:1000]}  # First 1000 chars of content

        ---
        """
        
        return augmented_query 