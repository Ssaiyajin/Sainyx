"""
checkpoint_utils.py
Everything needed so a Kaggle quota cutoff never costs you real training
progress. Save often, push off Kaggle immediately, resume automatically.
"""

import os
import time
import torch


def save_checkpoint(model, optimizer, scheduler_step, epoch, loss, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": scheduler_step,
        "epoch": epoch,
        "loss": loss,
    }, path)


def load_checkpoint(model, optimizer, path, device="cuda"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["step"], checkpoint["loss"]


def push_checkpoint_to_hf(local_path, repo_id, path_in_repo, token):
    """Push a checkpoint straight to your HF model repo so it survives even
    if /kaggle/working gets wiped when the session ends."""
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        token=token,
    )


def download_latest_checkpoint_from_hf(repo_id, path_in_repo, local_dir, token):
    """Pull the most recent checkpoint back down at the start of a fresh
    Kaggle session, so training resumes instead of restarting."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=repo_id,
        filename=path_in_repo,
        local_dir=local_dir,
        token=token,
    )


class SessionTimer:
    """Tracks elapsed wall-clock time so training can save-and-exit
    gracefully BEFORE Kaggle kills the session, instead of getting cut off
    mid-step. Kaggle GPU sessions cap out around 12h - default margin
    leaves a buffer to finish the current step and push the checkpoint."""

    def __init__(self, max_session_seconds=12 * 60 * 60, safety_margin_seconds=20 * 60):
        self.start_time = time.time()
        self.max_session_seconds = max_session_seconds
        self.safety_margin_seconds = safety_margin_seconds

    def should_stop(self):
        elapsed = time.time() - self.start_time
        return elapsed >= (self.max_session_seconds - self.safety_margin_seconds)

    def elapsed_minutes(self):
        return (time.time() - self.start_time) / 60