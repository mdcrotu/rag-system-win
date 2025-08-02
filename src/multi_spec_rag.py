#!/usr/bin/env python3
"""
Multi-Specification RAG System - Windows Optimized
"""

import fitz  # PyMuPDF
import re
from typing import List, Dict, Tuple, Optional
import chromadb
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import json
import os
from pathlib import Path
import ollama  # For local LLM integration

@dataclass
class SpecChunk:
    text: str
    section: str
    title: str
    page: int
    chapter: str
    spec_type: str

class MultiSpecRAGSystem:
    def __init__(self, db_path: str = "./data/vector_db"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=str(self.db_path))
        
        # Define spec configurations
        self.spec_configs = {
            'systemverilog': {
                'collection_name': 'ieee_1800_systemverilog',
                'description': 'IEEE 1800 SystemVerilog Specification',
                'section_pattern': r'^(\d+\.\d+(?:\.\d+)?)\s+([^\n]+)',
                'chapter_pattern': r'^(\d+)\s+([A-Z][^.\n]+)'
            },
            'ethernet': {
                'collection_name': 'ieee_802_3_ethernet',
                'description': 'IEEE 802.3 Ethernet Specification',
                'section_pattern': r'^(\d+\.\d+(?:\.\d+)?)\s+([^\n]+)',
                'chapter_pattern': r'^(\d+)\s+([A-Z][^.\n]+)'
            },
            'usb': {
                'collection_name': 'usb_specification',
                'description': 'USB Specification',
                'section_pattern': r'^(\d+\.\d+(?:\.\d+)?)\s+([^\n]+)',
                'chapter_pattern': r'^(\d+)\s+([A-Z][^.\n]+)'
            },
            'pcie': {
                'collection_name': 'pcie_specification',
                'description': 'PCIe Specification',
                'section_pattern': r'^(\d+\.\d+(?:\.\d+)?)\s+([^\n]+)',
                'chapter_pattern': r'^(\d+)\s+([A-Z][^.\n]+)'
            }
        }
        
        # Initialize collections
        self.collections = {}
        for spec_type, config in self.spec_configs.items():
            self.collections[spec_type] = self.chroma_client.get_or_create_collection(
                name=config['collection_name'],
                metadata={"description": config['description'], "spec_type": spec_type}
            )
    
    def extract_text_with_structure(self, pdf_path: str, spec_type: str) -> List[SpecChunk]:
        """Extract text while preserving document structure for any spec type"""
        doc = fitz.open(pdf_path)
        chunks = []
        current_chapter = ""
        config = self.spec_configs[spec_type]
        
        for page_num in range(min(len(doc), 100)):  # Limit for testing
            page = doc[page_num]
            text = page.get_text()
            
            # Look for chapter headers
            chapter_match = re.search(config['chapter_pattern'], text, re.MULTILINE)
            if chapter_match:
                current_chapter = f"{chapter_match.group(1)} {chapter_match.group(2)}"
            
            # Look for section headers
            sections = re.finditer(config['section_pattern'], text, re.MULTILINE)
            
            prev_end = 0
            prev_section_num = ""
            prev_section_title = ""
            
            for section_match in sections:
                section_num = section_match.group(1)
                section_title = section_match.group(2).strip()
                
                # Extract text from previous section
                if prev_end > 0:
                    section_text = text[prev_end:section_match.start()].strip()
                    if len(section_text) > 100:
                        chunks.append(SpecChunk(
                            text=section_text,
                            section=prev_section_num,
                            title=prev_section_title,
                            page=page_num + 1,
                            chapter=current_chapter,
                            spec_type=spec_type
                        ))
                
                prev_section_num = section_num
                prev_section_title = section_title
                prev_end = section_match.end()
            
            # Handle remaining text on page
            if prev_end > 0:
                remaining_text = text[prev_end:].strip()
                if len(remaining_text) > 100:
                    chunks.append(SpecChunk(
                        text=remaining_text,
                        section=prev_section_num,
                        title=prev_section_title,
                        page=page_num + 1,
                        chapter=current_chapter,
                        spec_type=spec_type
                    ))
        
        doc.close()
        return chunks
    
    def intelligent_chunking(self, chunks: List[SpecChunk], max_chunk_size: int = 1000) -> List[SpecChunk]:
        """Break large chunks into smaller ones while preserving context"""
        refined_chunks = []
        
        for chunk in chunks:
            if len(chunk.text) <= max_chunk_size:
                refined_chunks.append(chunk)
                continue
            
            paragraphs = chunk.text.split('\n\n')
            current_text = ""
            
            for para in paragraphs:
                if len(current_text + para) > max_chunk_size and current_text:
                    refined_chunks.append(SpecChunk(
                        text=current_text.strip(),
                        section=chunk.section,
                        title=chunk.title,
                        page=chunk.page,
                        chapter=chunk.chapter,
                        spec_type=chunk.spec_type
                    ))
                    # Add overlap
                    current_text = current_text.split('\n')[-1] + '\n' + para
                else:
                    current_text += '\n\n' + para if current_text else para
            
            if current_text.strip():
                refined_chunks.append(SpecChunk(
                    text=current_text.strip(),
                    section=chunk.section,
                    title=chunk.title,
                    page=chunk.page,
                    chapter=chunk.chapter,
                    spec_type=chunk.spec_type
                ))
        
        return refined_chunks
    
    def load_spec(self, pdf_path: str, spec_type: str):
        """Load a single specification into the appropriate collection"""
        if spec_type not in self.spec_configs:
            raise ValueError(f"Unknown spec type: {spec_type}")
        
        print(f"Processing {spec_type.upper()} specification...")
        
        # Extract and chunk
        raw_chunks = self.extract_text_with_structure(pdf_path, spec_type)
        refined_chunks = self.intelligent_chunking(raw_chunks)
        
        # Prepare data
        documents = [chunk.text for chunk in refined_chunks]
        metadatas = [
            {
                "section": chunk.section,
                "title": chunk.title,
                "page": chunk.page,
                "chapter": chunk.chapter,
                "spec_type": chunk.spec_type,
                "length": len(chunk.text)
            }
            for chunk in refined_chunks
        ]
        ids = [f"{spec_type}_chunk_{i:05d}" for i in range(len(refined_chunks))]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(documents)} chunks...")
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
        
        # Store in appropriate collection
        collection = self.collections[spec_type]
        collection.add(
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings.tolist(),
            ids=ids
        )
        
        print(f"Successfully loaded {len(refined_chunks)} chunks for {spec_type}!")
        return len(refined_chunks)
    
    def query_single_spec(self, question: str, spec_type: str, n_results: int = 3) -> List[Dict]:
        """Query a specific specification"""
        if spec_type not in self.collections:
            raise ValueError(f"Spec type {spec_type} not loaded")
        
        collection = self.collections[spec_type]
        results = collection.query(
            query_texts=[question],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        return self._format_results(results)
    
    def smart_query(self, question: str) -> Dict[str, List[Dict]]:
        """Intelligently determine which specs to query based on question content"""
        question_lower = question.lower()
        
        # Keywords to identify relevant specs
        spec_keywords = {
            'systemverilog': ['systemverilog', 'sv', 'logic', 'module', 'interface', 'class', 'package'],
            'ethernet': ['ethernet', 'mac', 'phy', '802.3', 'csma', 'collision'],
            'usb': ['usb', 'endpoint', 'descriptor', 'enumeration', 'bulk', 'interrupt'],
            'pcie': ['pcie', 'pci express', 'tlp', 'transaction layer', 'link training']
        }
        
        # Determine relevant specs
        relevant_specs = []
        for spec_type, keywords in spec_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                relevant_specs.append(spec_type)
        
        # If no specific match, query all specs
        if not relevant_specs:
            return self.query_all_specs(question, n_results_per_spec=1)
        
        # Query only relevant specs
        results = {}
        for spec_type in relevant_specs:
            if spec_type in self.collections:
                results[spec_type] = self.query_single_spec(question, spec_type, n_results=3)
        
        return results
    
    def query_all_specs(self, question: str, n_results_per_spec: int = 2) -> Dict[str, List[Dict]]:
        """Query all loaded specifications"""
        all_results = {}
        
        for spec_type in self.collections:
            try:
                results = self.query_single_spec(question, spec_type, n_results_per_spec)
                if results:  # Only include if we got results
                    all_results[spec_type] = results
            except Exception as e:
                print(f"Error querying {spec_type}: {e}")
        
        return all_results
    
    def _format_results(self, results) -> List[Dict]:
        """Format ChromaDB results"""
        formatted = []
        for i in range(len(results['documents'][0])):
            formatted.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity_score': 1 - results['distances'][0][i]
            })
        return formatted
    
    def generate_rag_response(self, question: str, model_name: str = "llama3.2:3b") -> str:
        """Generate response using local LLM with RAG context"""
        # Get relevant context
        context_results = self.smart_query(question)
        
        if not context_results:
            return "I couldn't find relevant information in the loaded specifications."
        
        # Build context string
        context_parts = []
        for spec_type, results in context_results.items():
            context_parts.append(f"\n=== {spec_type.upper()} SPECIFICATION ===")
            for i, result in enumerate(results[:2], 1):  # Limit to top 2 per spec
                metadata = result['metadata']
                context_parts.append(
                    f"Section {metadata['section']}: {metadata['title']}\n"
                    f"{result['text'][:800]}...\n"
                )
        
        context = "\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on the following hardware specification excerpts, please answer the question.

CONTEXT FROM SPECIFICATIONS:
{context}

QUESTION: {question}

Please provide a comprehensive answer based on the specification context above. If the context doesn't contain enough information, say so clearly."""
        
        try:
            # Use Ollama for local LLM inference
            response = ollama.chat(model=model_name, messages=[
                {'role': 'user', 'content': prompt}
            ])
            return response['message']['content']
        except Exception as e:
            return f"Error generating response: {e}"

# Usage functions
def load_all_specs(rag_system: MultiSpecRAGSystem, specs_directory: str):
    """Load all specifications from a directory"""
    spec_files = {
        'systemverilog': 'ieee_1800_systemverilog.pdf',
        'ethernet': 'ieee_802_3_ethernet.pdf',
        'usb': 'usb_specification.pdf',
        'pcie': 'pcie_specification.pdf'
    }
    
    loaded_specs = []
    specs_path = Path(specs_directory)
    
    for spec_type, filename in spec_files.items():
        filepath = specs_path / filename
        if filepath.exists():
            try:
                chunks = rag_system.load_spec(str(filepath), spec_type)
                loaded_specs.append((spec_type, chunks))
                print(f"✓ Loaded {spec_type}: {chunks} chunks")
            except Exception as e:
                print(f"✗ Failed to load {spec_type}: {e}")
        else:
            print(f"⚠ File not found: {filepath}")
    
    return loaded_specs
