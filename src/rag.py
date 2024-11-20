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
    metadata: Dict
    imports: Optional[List[str]] = None
    classes: Optional[List[str]] = None
    functions: Optional[List[str]] = None
    variables: Optional[List[str]] = None
    embedding: Optional[List[float]] = None

    def __str__(self):
        ret = f"File: {self.file_path}\nMetadata: {json.dumps(self.metadata, indent=2)}"
        # Type: {self.metadata['extension']}

        if self.classes:
            ret += f"\nClasses: {', '.join(self.classes)}"
        if self.functions:
            ret += f"\nFunctions: {', '.join(self.functions)}"
        if self.imports:
            ret += f"\nImports: {', '.join(self.imports)}"
        if self.variables:
            ret += f"\nVariables: {', '.join(self.variables)}"

        ret += f"\nContent:\n{self.content}"

        return ret

class StructuredRAG:
    ignore = ['__pycache__', 'task', 'test']

    def __init__(self, base_path: str = "."):
        self.client = openai.OpenAI(api_key='ollama', base_url="http://localhost:11434/v1")
        self.base_path = base_path
        self.contexts: Dict[str, CodeContext] = {}
        self.initialize_contexts()
        print("Initiliazed RAG with files:")
        for file in self.contexts.values():
            print(file.file_path)

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using local Ollama model"""
        response = self.client.embeddings.create(
            model="nomic-embed-text",
            input=text
        )
        return response.data[0].embedding
    
    def initialize_contexts(self):
        """Initialize code contexts for all relevant files"""
        self.update_contexts()

    def extract_code_elements(self, content: str) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Extract imports, classes, functions, and variables from code"""
        tree = ast.parse(content, type_comments=True)
        imports = []
        classes = []
        functions = []
        variables = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) or isinstance(node, ast.Import):
                import_dict = {}
                if isinstance(node, ast.ImportFrom):
                    import_dict['module'] = node.module
                if node.names:
                    import_dict['names'] = [n.name for n in node.names]
                imports.append(str(import_dict))
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
            "last_updated": os.path.getmtime(file_path),
        }

    def find_relevant_contexts(self, query: str, k: int = 3) -> List[CodeContext]:
        """Find k most relevant code contexts for a query"""
        query_embedding = self.get_embedding(query)
        
        # Calculate cosine similarities
        similarities = []
        for context in self.contexts.values():
            if context.embedding:
                # Cosine similarity
                similarity = np.dot(query_embedding, context.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(context.embedding)
                )
                similarities.append((similarity, context))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [context for _, context in similarities[:k]]

    def augment_query(self, query: str, k: int = 3) -> str:
        """Augment query with relevant code contexts and runtime information"""
        relevant_contexts = self.find_relevant_contexts(query, k)
        
        return f"# Relevant Code Contexts:\n{'\n---\n'.join(str(ctx) for ctx in relevant_contexts)}\n---\n# Your Goal:\nThe previous context is provided in order of relevancy from most to least relavant. Use the context to inform your answer to the query and always answer to the best of your ability using the context.\n---\n# Query:\n{query}" 
    
    def get_context(self, query: str, k: int = 3) -> str:
        """Get the relevant context for a query"""
        return '\n---\n'.join(str(ctx) for ctx in self.find_relevant_contexts(query, k))
    
    def update_contexts(self):
        """Update storedcontexts with new files or changes"""
        for root, _, files in os.walk(self.base_path):
            for file in files:
                if file.endswith(('.py', '.md', '.json')):
                    file_path = os.path.join(root, file)
                    if any(ig in file_path for ig in self.ignore):
                        continue

                    metadata = self.get_file_metadata(file_path)

                    if file_path in self.contexts:
                        if metadata['last_modified'] == self.contexts[file_path].metadata['last_modified']:
                            continue

                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if file.endswith('.py'):
                        imports, classes, functions, variables = self.extract_code_elements(content)
                    else:
                        imports, classes, functions, variables = None, None, None, None


                    self.contexts[file_path] = CodeContext(
                        file_path=file_path,
                        content=content,
                        imports=imports,
                        classes=classes,
                        functions=functions,
                        variables=variables,
                        metadata=metadata,
                    )
                    embedding = self.get_embedding(str(self.contexts[file_path]))
                    self.contexts[file_path].embedding = embedding