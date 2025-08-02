#!/usr/bin/env python3
"""
Debug script to diagnose RAG system issues
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

from multi_spec_rag import MultiSpecRAGSystem
from config import *
import ollama

def test_vector_search():
    """Test if vector search is finding relevant content"""
    print("🔍 Testing Vector Search...")
    
    # Initialize RAG system
    rag_system = MultiSpecRAGSystem(str(DB_DIR))
    
    # Check what specs are loaded
    print("\n📚 Loaded Collections:")
    for spec_type in rag_system.collections:
        collection = rag_system.collections[spec_type]
        count = collection.count()
        print(f"  {spec_type}: {count} chunks")
        
        if count == 0:
            print(f"  ⚠️ WARNING: {spec_type} has no chunks!")
    
    # Test search with a simple question
    test_questions = [
        "interface",
        "module", 
        "wire",
        "signal",
        "USB",
        "ethernet"
    ]
    
    for question in test_questions:
        print(f"\n🔍 Testing search for: '{question}'")
        
        try:
            # Test smart query
            results = rag_system.smart_query(question)
            
            if not results:
                print("  ❌ No results found")
                continue
                
            for spec_type, spec_results in results.items():
                print(f"  📖 {spec_type}: {len(spec_results)} results")
                
                if spec_results:
                    best_result = spec_results[0]
                    print(f"    Score: {best_result['similarity_score']:.3f}")
                    print(f"    Section: {best_result['metadata']['section']}")
                    print(f"    Text preview: {best_result['text'][:150]}...")
                    
        except Exception as e:
            print(f"  ❌ Error: {e}")

# def test_ollama_models():
#     """Test which Ollama models are working"""
#     print("\n🤖 Testing Ollama Models...")
    
#     # Get available models
#     try:
#         models = ollama.list()
#         available_models = [model['name'] for model in models['models']]
#         print(f"Available models: {available_models}")
#     except Exception as e:
#         print(f"❌ Error listing models: {e}")
#         return
    
#     # Test each model
#     test_models = ['llama3.2:1b', 'llama3.2:3b', 'llama3.1:8b']
    
#     for model_name in test_models:
#         if model_name in available_models:
#             print(f"\n🧪 Testing {model_name}...")
#             try:
#                 response = ollama.chat(
#                     model=model_name, 
#                     messages=[{'role': 'user', 'content': 'What is SystemVerilog? Answer in one sentence.'}]
#                 )
#                 print(f"  ✅ Works! Response: {response['message']['content'][:100]}...")
#             except Exception as e:
#                 print(f"  ❌ Failed: {e}")
#         else:
#             print(f"  ⚠️ {model_name} not available")

def test_ollama_models():
    """Test which Ollama models are working"""
    print("\n🤖 Testing Ollama Models...")
    
    # Get available models
    try:
        models_response = ollama.list()
        
        # Extract model names from the Model objects
        available_models = []
        if hasattr(models_response, 'models'):
            # models_response.models is a list of Model objects
            for model_obj in models_response.models:
                if hasattr(model_obj, 'model'):
                    available_models.append(model_obj.model)
        
        print(f"Available models: {available_models}")
        
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        print("Make sure Ollama is running with: ollama serve")
        return
    
    # Test each model
    test_models = ['llama3.2:1b', 'llama3.2:3b', 'llama3.1:8b']
    
    for model_name in test_models:
        if model_name in available_models:
            print(f"\n🧪 Testing {model_name}...")
            try:
                response = ollama.chat(
                    model=model_name, 
                    messages=[{'role': 'user', 'content': 'What is SystemVerilog? Answer in one sentence.'}]
                )
                print(f"  ✅ Works! Response: {response['message']['content'][:100]}...")
            except Exception as e:
                print(f"  ❌ Failed: {e}")
        else:
            print(f"  ⚠️ {model_name} not available")
            # Show what models are actually available
            if available_models:
                print(f"    Available models: {available_models}")

def test_full_rag_pipeline():
    """Test the complete RAG pipeline with debugging"""
    print("\n🔄 Testing Full RAG Pipeline...")
    
    rag_system = MultiSpecRAGSystem(str(DB_DIR))
    
    test_question = "How do you declare a wire in SystemVerilog?"
    print(f"Question: {test_question}")
    
    # Step 1: Test retrieval
    print("\n1️⃣ Testing retrieval...")
    context_results = rag_system.smart_query(test_question)
    
    if not context_results:
        print("❌ No context retrieved!")
        return
    
    print(f"✅ Retrieved from {len(context_results)} specs")
    
    # Step 2: Build context
    print("\n2️⃣ Building context...")
    context_parts = []
    for spec_type, results in context_results.items():
        print(f"  {spec_type}: {len(results)} chunks")
        context_parts.append(f"\n=== {spec_type.upper()} SPECIFICATION ===")
        for i, result in enumerate(results[:2], 1):
            metadata = result['metadata']
            context_parts.append(
                f"Section {metadata['section']}: {metadata['title']}\n"
                f"{result['text'][:400]}...\n"
            )
    
    context = "\n".join(context_parts)
    print(f"Context length: {len(context)} characters")
    
    # Step 3: Test LLM
    print("\n3️⃣ Testing LLM response...")
    prompt = f"""Based on the following hardware specification excerpts, please answer the question.

CONTEXT FROM SPECIFICATIONS:
{context}

QUESTION: {test_question}

Please provide a comprehensive answer based on the specification context above."""
    
    try:
        # Try with default model
        response = ollama.chat(model="llama3.2:3b", messages=[
            {'role': 'user', 'content': prompt}
        ])
        print(f"✅ LLM Response: {response['message']['content'][:200]}...")
        
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        
        # Try with smaller model
        try:
            print("Trying smaller model...")
            response = ollama.chat(model="llama3.2:1b", messages=[
                {'role': 'user', 'content': prompt}
            ])
            print(f"✅ LLM Response (1b): {response['message']['content'][:200]}...")
        except Exception as e2:
            print(f"❌ Smaller model also failed: {e2}")

def check_spec_loading():
    """Check if specs were loaded correctly"""
    print("\n📄 Checking Specification Loading...")
    
    # Check PDF files
    spec_files = list(SPECS_DIR.glob("*.pdf"))
    print(f"PDF files found: {len(spec_files)}")
    
    for pdf_file in spec_files:
        print(f"  📄 {pdf_file.name} ({pdf_file.stat().st_size / 1024 / 1024:.1f} MB)")
        
        # Try to open and read first page
        try:
            import fitz
            doc = fitz.open(str(pdf_file))
            page = doc[0]
            text = page.get_text()
            print(f"    ✅ Readable, first page has {len(text)} characters")
            if len(text) < 100:
                print(f"    ⚠️ WARNING: Very little text extracted!")
            doc.close()
        except Exception as e:
            print(f"    ❌ Error reading: {e}")

def main():
    print("🚀 RAG System Diagnostic Tool")
    print("=" * 50)
    
    # Run all tests
    check_spec_loading()
    test_ollama_models()
    test_vector_search()
    test_full_rag_pipeline()
    
    print("\n" + "=" * 50)
    print("🏁 Diagnostic Complete!")
    print("\nCommon Issues and Solutions:")
    print("1. No chunks loaded → Check PDF files and reload specs")
    print("2. Low similarity scores → Try different embedding model")
    print("3. Ollama errors → Check if service is running")
    print("4. Poor LLM responses → Try different model or improve prompts")

if __name__ == "__main__":
    main()