#!/usr/bin/env python3
"""
Quick fix script for common RAG issues
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

import ollama
import shutil

def fix_ollama_model():
    """Ensure we're using a working model"""
    print("🔧 Checking Ollama models...")
    
    try:
        models = ollama.list()
        available = [m['name'] for m in models['models']]
        print(f"Available models: {available}")
        
        # Test models in order of preference
        test_order = ['llama3.2:1b', 'llama3.2:3b', 'llama3.1:8b']
        working_model = None
        
        for model in test_order:
            if model in available:
                try:
                    response = ollama.chat(model=model, messages=[
                        {'role': 'user', 'content': 'Say "OK" if you work.'}
                    ])
                    print(f"✅ {model} works!")
                    working_model = model
                    break
                except Exception as e:
                    print(f"❌ {model} failed: {e}")
        
        if working_model:
            # Update config file
            config_path = Path("src/config.py")
            if config_path.exists():
                content = config_path.read_text()
                # Replace the DEFAULT_LLM_MODEL line
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'DEFAULT_LLM_MODEL' in line and '=' in line:
                        lines[i] = f'DEFAULT_LLM_MODEL = "{working_model}"'
                        break
                config_path.write_text('\n'.join(lines))
                print(f"✅ Updated config to use {working_model}")
        else:
            print("❌ No working models found!")
            print("Run: ollama pull llama3.2:1b")
            
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        print("Is Ollama running? Try: ollama serve")

def fix_embedding_model():
    """Switch to a better embedding model"""
    print("\n🔧 Updating embedding model...")
    
    config_path = Path("src/config.py")
    if config_path.exists():
        content = config_path.read_text()
        
        # Update to better model
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'EMBEDDING_MODEL' in line and '=' in line:
                lines[i] = 'EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"  # Better for Q&A'
                break
        
        config_path.write_text('\n'.join(lines))
        print("✅ Updated to Q&A optimized embedding model")
        print("ℹ️ You'll need to reload your specs for this to take effect")

def reset_vector_db():
    """Reset the vector database"""
    print("\n🔧 Resetting vector database...")
    
    db_path = Path("data/vector_db")
    if db_path.exists():
        try:
            shutil.rmtree(db_path)
            print("✅ Vector database reset")
            print("ℹ️ You'll need to reload your specifications")
        except Exception as e:
            print(f"❌ Error resetting database: {e}")
    else:
        print("ℹ️ No vector database found")

def check_specs():
    """Check specification files"""
    print("\n🔧 Checking specification files...")
    
    specs_dir = Path("specs")
    pdf_files = list(specs_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ No PDF files found in specs/ directory")
        print("Add your specification PDFs to the specs/ folder")
        return
    
    print(f"✅ Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        size_mb = pdf.stat().st_size / 1024 / 1024
        print(f"  📄 {pdf.name} ({size_mb:.1f} MB)")
        
        # Quick test if PDF is readable
        try:
            import fitz
            doc = fitz.open(str(pdf))
            page_count = len(doc)
            first_page_text = doc[0].get_text()
            doc.close()
            
            if len(first_page_text) < 50:
                print(f"    ⚠️ WARNING: Very little text extracted from first page")
            else:
                print(f"    ✅ Readable ({page_count} pages)")
                
        except Exception as e:
            print(f"    ❌ Error reading PDF: {e}")

def create_test_config():
    """Create a minimal test configuration"""
    print("\n🔧 Creating optimized test configuration...")
    
    test_config = '''"""Optimized configuration for testing"""
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
SPECS_DIR = PROJECT_ROOT / "specs"
DB_DIR = PROJECT_ROOT / "data" / "vector_db_test"
LOGS_DIR = PROJECT_ROOT / "logs"

# Optimized settings for testing
EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"  # Better for Q&A
DEFAULT_LLM_MODEL = "llama3.2:1b"  # Fastest model

# Smaller chunks for better matching
MAX_CHUNK_SIZE = 600
CHUNK_OVERLAP = 50

# More results for better context
DEFAULT_RESULTS_PER_SPEC = 5
MAX_CONTEXT_LENGTH = 6000

# Create directories
DB_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
'''
    
    Path("src/config_test.py").write_text(test_config)
    print("✅ Created optimized test configuration")
    print("ℹ️ Import this in your RAG system for testing")

def main():
    print("🚀 RAG System Quick Fix Tool")
    print("=" * 40)
    
    fix_ollama_model()
    fix_embedding_model()
    check_specs()
    create_test_config()
    
    print("\n" + "=" * 40)
    print("🏁 Quick fixes applied!")
    print("\nNext steps:")
    print("1. Restart your RAG system")
    print("2. Let it reload specifications with new settings")
    print("3. Test with a simple question")
    print("4. If still not working, run: python debug_rag.py")

if __name__ == "__main__":
    main()