from rag import StructuredRAG
from agent_module import Agent
import pprint

def test_rag_initialization():
    """Test RAG system initialization and file indexing"""
    rag = StructuredRAG()
    print("\n=== Indexed Files ===")
    for context in rag.contexts.values():
        print(str(context), end="\n\n")
        

def test_context_retrieval():
    """Test context retrieval for specific queries"""
    rag = StructuredRAG()
    
    test_queries = [
        "How does the solver function work?",
        "What are the available agent actions?",
        "How is task evaluation implemented?",
    ]
    
    print("\n=== Context Retrieval Test ===")
    for query in test_queries:
        print(f"\nQuery: {query}")
        contexts = rag.find_relevant_contexts(query, k=2)
        for ctx in contexts:
            print(str(ctx), end="\n\n")

def test_query_augmentation():
    """Test query augmentation with code and runtime context"""
    rag = StructuredRAG()
    
    test_query = "How can I modify the solver function to improve performance?"
    augmented_query = rag.augment_query(test_query)

    print("\n=== Query Augmentation Test ===")
    print(f"Original query: {test_query}")
    print("\nAugmented query preview:")
    print(augmented_query)

def test_agent_integration():
    """Test RAG integration with the Agent class"""
    agent = Agent()
    
    test_message = {
        "role": "user",
        "content": "How does the solver function handle math problems?"
    }
    
    print("\n=== Agent Integration Test ===")
    response = agent.action_call_llm(
        model="gpt-3.5-turbo",
        messages=[test_message],
        temperature=0.7,
        max_completion_tokens=500
    )
    
    print("\nLLM Response:")
    pprint.pp(response)

def main():
    """Run all tests"""
    print("Starting RAG system tests...")
    
    # print("\nTest 1: Initialization")
    # test_rag_initialization()
    
    # print("\nTest 2: Context Retrieval")
    # test_context_retrieval()
    
    print("\nTest 3: Query Augmentation")
    test_query_augmentation()
    
    # print("\nTest 4: Agent Integration")
    # test_agent_integration()

if __name__ == "__main__":
    main() 