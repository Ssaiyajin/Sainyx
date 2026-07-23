"""
Sainyx Model Factory
Centralized model loading and management
"""

import os
import torch
from typing import Dict, Tuple, Optional, Callable
from huggingface_hub import hf_hub_download
from pathlib import Path

from config import (
    DEVICE, 
    TEXT_MODEL_PATH, 
    DIFFUSION_MODEL_PATH,
    HF_TOKEN, 
    HF_MODEL_REPO,
    HF_STAGING_REPO,
    TEXT_MODEL_HF_FILENAME,
    DIFFUSION_MODEL_HF_FILENAME,
)
from core.models.text import Sainyx


class ModelFactory:
    """Factory for loading and caching models"""
    
    _cache: Dict[str, torch.nn.Module] = {}
    _vocabs: Dict[str, Dict] = {}
    
    @staticmethod
    def download_model_from_hf(
        filename: str,
        repo_id: str = HF_MODEL_REPO,
        force_download: bool = False
    ) -> str:
        """Download model from HuggingFace Hub"""
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type='model',
                token=HF_TOKEN,
                force_download=force_download,
            )
            print(f"✅ Model downloaded to: {path}")
            return path
        except Exception as e:
            print(f"❌ Download failed: {e}")
            raise
    
    @staticmethod
    def ensure_model_exists(model_path: Path, filename: str, repo_id: str) -> Path:
        """Ensure model file exists locally, download if needed"""
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}. Downloading from HuggingFace...")
            downloaded_path = ModelFactory.download_model_from_hf(
                filename=filename,
                repo_id=repo_id,
            )
            # Copy to expected location if different
            if downloaded_path != str(model_path):
                import shutil
                shutil.copy2(downloaded_path, model_path)
        return model_path
    
    @staticmethod
    def load_text_model(force_download: bool = False) -> Tuple[torch.nn.Module, Dict]:
        """Load text generation model with vocabulary"""
        if 'text' in ModelFactory._cache:
            return ModelFactory._cache['text'], ModelFactory._vocabs['text']
        
        # Ensure model exists
        model_path = ModelFactory.ensure_model_exists(
            TEXT_MODEL_PATH,
            TEXT_MODEL_HF_FILENAME,
            HF_MODEL_REPO
        )
        
        print(f"Loading text model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        print("✅ Checkpoint loaded")
        
        # Extract vocabulary
        chars = checkpoint['chars']
        stoi = checkpoint['stoi']
        itos = {int(k) if isinstance(k, str) else k: v for k, v in checkpoint['itos'].items()}
        
        # Create encoding/decoding functions
        encode = lambda s: [stoi.get(c, 0) for c in s]
        decode = lambda l: ''.join([itos.get(i, '?') for i in l])
        
        # Load model
        state_dict = checkpoint['model_state_dict']
        vocab_size = state_dict['token_embedding.weight'].shape[0]
        print(f"Vocab size: {vocab_size}")
        
        model = Sainyx(vocab_size=vocab_size).to(DEVICE)
        print("✅ Model created")
        model.load_state_dict(state_dict)
        print("✅ Weights loaded")
        model.eval()
        
        # Cache
        vocab_dict = {
            'chars': chars,
            'stoi': stoi,
            'itos': itos,
            'encode': encode,
            'decode': decode,
            'vocab_size': vocab_size,
        }
        ModelFactory._cache['text'] = model
        ModelFactory._vocabs['text'] = vocab_dict
        
        print("🔥 Text model ready!")
        return model, vocab_dict
    
    @staticmethod
    def load_diffusion_model(force_download: bool = False):
        """Load diffusion model for image generation"""
        if 'diffusion' in ModelFactory._cache:
            return ModelFactory._cache['diffusion']
        
        # Ensure model exists
        model_path = ModelFactory.ensure_model_exists(
            DIFFUSION_MODEL_PATH,
            DIFFUSION_MODEL_HF_FILENAME,
            HF_STAGING_REPO
        )
        
        print(f"Loading diffusion model from: {model_path}")
        
        try:
            # Import local generate module
            from generation.image.generate import load_model as load_diffusion_model_impl
            diffusion_model, image_size, timesteps = load_diffusion_model_impl(
                str(model_path), 
                device=DEVICE
            )
            
            result = {
                'model': diffusion_model,
                'image_size': image_size,
                'timesteps': timesteps,
            }
            ModelFactory._cache['diffusion'] = result
            print("🔥 Diffusion model ready!")
            return result
            
        except Exception as e:
            print(f"⚠️  Diffusion model loading failed: {e}")
            return None
    
    @staticmethod
    def get_model(model_type: str) -> Optional[torch.nn.Module]:
        """Get cached model or load it"""
        if model_type == 'text':
            model, vocab = ModelFactory.load_text_model()
            return model
        elif model_type == 'diffusion':
            return ModelFactory.load_diffusion_model()
        return None
    
    @staticmethod
    def get_vocab(model_type: str) -> Optional[Dict]:
        """Get cached vocabulary"""
        if model_type not in ModelFactory._vocabs:
            if model_type == 'text':
                ModelFactory.load_text_model()
        return ModelFactory._vocabs.get(model_type)
    
    @staticmethod
    def clear_cache():
        """Clear model cache to free memory"""
        ModelFactory._cache.clear()
        ModelFactory._vocabs.clear()
        print("Model cache cleared")


# Global instance
def get_text_model() -> Tuple[torch.nn.Module, Dict]:
    """Convenience function to load text model"""
    return ModelFactory.load_text_model()


def get_diffusion_model() -> Optional[Dict]:
    """Convenience function to load diffusion model"""
    return ModelFactory.load_diffusion_model()


def get_vocab(model_type: str = 'text') -> Optional[Dict]:
    """Convenience function to get vocabulary"""
    return ModelFactory.get_vocab(model_type)
