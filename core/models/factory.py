"""
Sainyx Model Factory
Centralized loading and in-memory caching for all three generation models
(text, image, video). Each loader ensures its checkpoint exists locally
(downloading from Hugging Face Hub into config.CHECKPOINT_DIR if needed),
then loads and caches the model so repeat requests don't hit disk/network
again.

Video is trained incrementally on Kaggle, so a checkpoint may not exist
yet - load_video_model() returns None instead of raising if that's the
case, same pattern as load_image_model().
"""

import os
import shutil
from typing import Dict, Optional, Tuple

import torch
from huggingface_hub import hf_hub_download

import config
from core.models.text import Sainyx


class ModelFactory:
    """Factory for loading and caching Sainyx's models."""

    _cache: Dict[str, object] = {}
    _vocabs: Dict[str, Dict] = {}

    # ── Shared HF download helper ───────────────────────────────────────
    @staticmethod
    def _ensure_local(repo_id: str, filename: str, local_path: str) -> str:
        """Make sure `filename` from `repo_id` exists at `local_path`,
        downloading it from HF Hub if it doesn't."""
        if os.path.exists(local_path):
            return local_path

        print(f"Downloading {filename} from {repo_id}...")
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            token=config.HF_TOKEN,
        )
        if downloaded_path != local_path:
            shutil.copy2(downloaded_path, local_path)
        print(f"✅ {filename} ready at {local_path}")
        return local_path

    # ── Text model (character-level GPT) ────────────────────────────────
    @staticmethod
    def load_text_model(force_download: bool = False) -> Tuple[torch.nn.Module, Dict]:
        if "text" in ModelFactory._cache and not force_download:
            return ModelFactory._cache["text"], ModelFactory._vocabs["text"]

        if force_download and os.path.exists(config.TEXT_MODEL_LOCAL_PATH):
            os.remove(config.TEXT_MODEL_LOCAL_PATH)

        model_path = ModelFactory._ensure_local(
            config.TEXT_MODEL_REPO_ID,
            config.TEXT_MODEL_FILENAME,
            config.TEXT_MODEL_LOCAL_PATH,
        )

        print(f"Loading text model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=config.DEVICE)

        chars = checkpoint["chars"]
        stoi = checkpoint["stoi"]
        itos = {int(k) if isinstance(k, str) else k: v for k, v in checkpoint["itos"].items()}
        encode = lambda s: [stoi.get(c, 0) for c in s]
        decode = lambda l: "".join(itos.get(i, "?") for i in l)

        state_dict = checkpoint["model_state_dict"]
        vocab_size = state_dict["token_embedding.weight"].shape[0]

        model = Sainyx(vocab_size=vocab_size).to(config.DEVICE)
        compatible_state_dict = {}
        for key, value in state_dict.items():
            normalized_key = key.removeprefix("module.")
            normalized_key = normalized_key.replace(".sa.", ".self_attention.")
            normalized_key = normalized_key.replace(".ff.", ".feed_forward.")
            compatible_state_dict[normalized_key] = value
        model.load_state_dict(compatible_state_dict)
        model.eval()

        vocab_dict = {
            "chars": chars,
            "stoi": stoi,
            "itos": itos,
            "encode": encode,
            "decode": decode,
            "vocab_size": vocab_size,
        }
        ModelFactory._cache["text"] = model
        ModelFactory._vocabs["text"] = vocab_dict
        print("🔥 Text model ready!")
        return model, vocab_dict

    # ── Image diffusion model ───────────────────────────────────────────
    @staticmethod
    def load_image_model(force_download: bool = False) -> Optional[Dict]:
        if "image" in ModelFactory._cache and not force_download:
            return ModelFactory._cache["image"]

        if force_download and os.path.exists(config.IMAGE_MODEL_LOCAL_PATH):
            os.remove(config.IMAGE_MODEL_LOCAL_PATH)

        try:
            model_path = ModelFactory._ensure_local(
                config.IMAGE_MODEL_REPO_ID,
                config.IMAGE_MODEL_FILENAME,
                config.IMAGE_MODEL_LOCAL_PATH,
            )

            from generation.image.generate import load_model as load_image_checkpoint

            model, image_size, timesteps = load_image_checkpoint(model_path, device=config.DEVICE)

            result = {"model": model, "image_size": image_size, "timesteps": timesteps}
            ModelFactory._cache["image"] = result
            print("🔥 Image diffusion model ready!")
            return result

        except Exception as e:
            print(f"⚠️  Image model loading failed: {e}")
            return None

    # ── Video diffusion model (VideoUNet, unconditional clips) ──────────
    @staticmethod
    def _load_video_weights(model_path: str):
        """Return (state_dict, settings) from either the consolidated final file
        or a resumable training checkpoint (which carries EMA weights)."""
        checkpoint = torch.load(model_path, map_location=config.DEVICE)
        state = checkpoint.get("ema_state_dict") or checkpoint["model_state_dict"]
        return state, checkpoint

    @staticmethod
    def load_video_model(force_download: bool = False) -> Optional[Dict]:
        if "video" in ModelFactory._cache and not force_download:
            return ModelFactory._cache["video"]

        # Prefer the final model; fall back to the training checkpoint while a
        # run is still in progress.
        sources = [
            (config.VIDEO_MODEL_PATH_IN_REPO, config.VIDEO_MODEL_LOCAL_PATH),
            (config.VIDEO_CHECKPOINT_PATH_IN_REPO, config.VIDEO_CHECKPOINT_LOCAL_PATH),
        ]
        if force_download:
            for _, local in sources:
                if os.path.exists(local):
                    os.remove(local)

        last_error = None
        for path_in_repo, local_path in sources:
            try:
                model_path = ModelFactory._ensure_local(
                    config.VIDEO_MODEL_REPO_ID, path_in_repo, local_path
                )

                from model.video_unet import VideoUNet

                state, checkpoint = ModelFactory._load_video_weights(model_path)
                base_ch = checkpoint.get("base_ch", config.VIDEO_BASE_CH_DEFAULT)
                model = VideoUNet(base_ch=base_ch).to(config.DEVICE)
                model.load_state_dict(state)
                model.eval()

                result = {
                    "model": model,
                    "image_size": checkpoint.get("image_size", config.VIDEO_IMAGE_SIZE_DEFAULT),
                    "clip_len": checkpoint.get("clip_len", config.VIDEO_CLIP_LEN_DEFAULT),
                    "timesteps": checkpoint.get("timesteps", config.VIDEO_TIMESTEPS_DEFAULT),
                }
                ModelFactory._cache["video"] = result
                print(
                    f"🔥 Video diffusion model ready from {path_in_repo}! "
                    f"({result['clip_len']} frames at {result['image_size']}px, "
                    f"loss {checkpoint.get('final_loss', checkpoint.get('loss', '?'))})"
                )
                return result
            except Exception as e:
                last_error = e

        # Expected before the first video checkpoint exists on the Hub.
        print(f"⚠️  Video model not available yet: {last_error}")
        return None

    # ── Generic accessors ────────────────────────────────────────────────
    @staticmethod
    def get_model(model_type: str):
        if model_type == "text":
            model, _ = ModelFactory.load_text_model()
            return model
        elif model_type == "image":
            result = ModelFactory.load_image_model()
            return result["model"] if result else None
        elif model_type == "video":
            result = ModelFactory.load_video_model()
            return result["model"] if result else None
        return None

    @staticmethod
    def get_vocab(model_type: str = "text") -> Optional[Dict]:
        if model_type not in ModelFactory._vocabs and model_type == "text":
            ModelFactory.load_text_model()
        return ModelFactory._vocabs.get(model_type)

    @staticmethod
    def clear_cache():
        ModelFactory._cache.clear()
        ModelFactory._vocabs.clear()
        print("Model cache cleared")


# ── Convenience module-level functions ──────────────────────────────────
def get_text_model() -> Tuple[torch.nn.Module, Dict]:
    return ModelFactory.load_text_model()


def get_image_model() -> Optional[Dict]:
    return ModelFactory.load_image_model()


def get_video_model() -> Optional[Dict]:
    return ModelFactory.load_video_model()


def get_vocab(model_type: str = "text") -> Optional[Dict]:
    return ModelFactory.get_vocab(model_type)