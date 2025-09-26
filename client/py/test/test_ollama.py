#!/usr/bin/env python3
"""
Test script for Ollama embedder
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from hakesclient.extensions.ollama import OllamaEmbedder

def test_ollama_with_config():
    """Test the Ollama embedder with custom configuration"""
    
    config = {
        "base_url": "http://localhost:11434",
        "model": "nomic-embed-text"
    }
    
    try:
        embedder = OllamaEmbedder.from_config(config)
        texts = ["Another test sentence"]
        embeddings = embedder.embed_text(texts)
        
        print("Successfully generated embeddings with config!")
        print(f"Embedding shape: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"Error testing Ollama embedder with config: {e}")
        return False

if __name__ == "__main__":
    print("Testing Ollama Embedder...")
    print("=" * 40)
    
    
    success2 = test_ollama_with_config()
    print()
    
    if  success2:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        print("\nTroubleshooting:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Pull the embedding model: ollama pull nomic-embed-text")
        print("3. Check if Ollama is accessible at http://localhost:11434")
        print("4. Verify the model name is correct") 