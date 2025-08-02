#!/usr/bin/env python3
"""
Enhanced Multi-Spec RAG System - Windows Version
"""
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from multi_spec_rag import MultiSpecRAGSystem, load_all_specs
from config import *
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'rag_system.log'),
        logging.StreamHandler()
    ]
)

def main():
    print("🚀 Hardware Specifications RAG System (Windows)")
    print("=" * 60)
    
    # Initialize system
    try:
        rag_system = MultiSpecRAGSystem(str(DB_DIR))
        logging.info("RAG system initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize RAG system: {e}")
        return
    
    # Check for spec files
    spec_files = list(SPECS_DIR.glob("*.pdf"))
    if not spec_files:
        print(f"⚠️  No PDF files found in {SPECS_DIR}")
        print("Please add your specification PDFs to the specs/ directory")
        print("Run: python src/download_specs.py for more info")
        return
    
    print(f"📁 Found {len(spec_files)} PDF files:")
    for pdf_file in spec_files:
        print(f"   - {pdf_file.name}")
    
    # Load specifications
    print("\n🔄 Loading specifications...")
    loaded_specs = load_all_specs(rag_system, str(SPECS_DIR))
    
    if not loaded_specs:
        print("❌ No specifications could be loaded")
        return
    
    print(f"\n✅ Successfully loaded {len(loaded_specs)} specifications!")
    
    # Interactive mode
    interactive_mode(rag_system)

def interactive_mode(rag_system):
    """Interactive query interface"""
    print("\n" + "=" * 60)
    print("🎯 INTERACTIVE MODE - Ask questions about your specs!")
    print("=" * 60)
    print("Commands:")
    print("  help     - Show help")
    print("  specs    - List loaded specifications") 
    print("  quit     - Exit system")
    print("  Or just ask a question!")
    
    while True:
        try:
            user_input = input("\n📝 Your question: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            elif user_input.lower() == 'help':
                show_help()
                continue
                
            elif user_input.lower() == 'specs':
                show_loaded_specs(rag_system)
                continue
            
            # Process question
            print("\n🤔 Thinking...")
            try:
                response = rag_system.generate_rag_response(user_input)
                print(f"\n🎯 Answer:\n{response}")
            except Exception as e:
                print(f"❌ Error generating response: {e}")
                logging.error(f"Query error: {e}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            logging.error(f"Unexpected error: {e}")

def show_help():
    """Show help information"""
    help_text = """
🔍 Question Examples:
  "How do I declare a SystemVerilog interface?"
  "What are USB endpoint types?"
  "Explain Ethernet frame format"
  "What is PCIe TLP?"

💡 Tips:
  - Be specific in your questions
  - Mention the technology (SystemVerilog, USB, etc.) for better results
  - Ask about syntax, concepts, or implementation details
"""
    print(help_text)

def show_loaded_specs(rag_system):
    """Show information about loaded specifications"""
    print("\n📚 Loaded Specifications:")
    for spec_type in rag_system.collections:
        collection = rag_system.collections[spec_type]
        count = collection.count()
        print(f"  {spec_type.upper()}: {count} chunks")

if __name__ == "__main__":
    main()
