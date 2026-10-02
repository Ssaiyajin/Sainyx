"""
checkpoint_utils.py
Shared across text, image, and video training pipelines.

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


def prune_old_versions(repo_id, path_in_repo, token, keep_latest=True):
    """Permanently delete superseded versions of ONE file in a HF repo.

    Every upload to the same path keeps the old copy in the repo history, and old
    copies count against storage. Call this right after a push so at most the
    newest version stays. With keep_latest=False every version is removed, for a
    file that was just deleted. Set SAINYX_PRUNE=0 to turn this off.
    Never raises: a failed prune only prints a warning."""
    if os.environ.get("SAINYX_PRUNE", "1") == "0":
        return
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        versions = [f for f in api.list_lfs_files(repo_id, token=token) if f.filename == path_in_repo]
        if not versions:
            return
        keep = {max(versions, key=lambda f: f.pushed_at).file_oid} if keep_latest else set()
        stale = {f.file_oid: f for f in versions if f.file_oid not in keep}
        if not stale:
            return
        freed = sum(f.size for f in stale.values())
        api.permanently_delete_lfs_files(repo_id, list(stale.values()), token=token)
        print(f"   🧹 Pruned {len(stale)} old version(s) of hf://{repo_id}/{path_in_repo} (freed {freed / 1e9:.2f} GB)")
    except Exception as e:
        print(f"   ⚠️  Could not prune old versions of hf://{repo_id}/{path_in_repo}: {e}")


def push_to_both_repos(local_path, targets, token):
    """
    Push a checkpoint to multiple (repo_id, path_in_repo) targets.
    Each push is independent - if one repo fails (permissions, network
    blip, etc.) the others still get attempted, and each result gets
    printed rather than silently swallowed.

    targets: list of (repo_id, path_in_repo) tuples
    Returns True only if every target succeeded.
    """
    all_ok = True
    for repo_id, path_in_repo in targets:
        try:
            push_checkpoint_to_hf(local_path, repo_id, path_in_repo, token)
            print(f"   📤 Pushed to hf://{repo_id}/{path_in_repo}")
            prune_old_versions(repo_id, path_in_repo, token)
        except Exception as e:
            all_ok = False
            print(f"   ⚠️  Push to hf://{repo_id}/{path_in_repo} failed: {e}")
    return all_ok

            

def delete_checkpoint_from_hf(repo_id, path_in_repo, token):
    """Remove the resumable 'latest' checkpoint once the final consolidated
    model has been pushed successfully - it's served its purpose and just
    clutters the repo otherwise."""
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        api.delete_file(
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            token=token,
        )
        print(f"   🗑️  Deleted hf://{repo_id}/{path_in_repo}")
        prune_old_versions(repo_id, path_in_repo, token, keep_latest=False)
    except Exception as e:
        print(f"   ⚠️  Could not delete hf://{repo_id}/{path_in_repo}: {e}")