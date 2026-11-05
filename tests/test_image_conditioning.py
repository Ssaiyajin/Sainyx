import pytest

torch = pytest.importorskip("torch")

from generation.image.diffusion_scheduler import DiffusionScheduler  # noqa: E402
from model.image_unet_cond import CondUNet  # noqa: E402

VOCAB = [
    "solo", "1boy", "dragon_ball", "son_goku", "vegeta", "gogeta", "vegito",
    "blue_hair", "blonde_hair", "super_saiyan", "super_saiyan_blue",
    "trunks_(dragon_ball)", "red_hair",
]

def _tiny_model(num_tags=len(VOCAB)):
    return CondUNet(base_channels=32, ch_mult=(1, 2), num_res_blocks=1,
                    attn_levels=(1,), num_tags=num_tags)


def test_cond_unet_shapes_and_conditioning_matters():
    model = _tiny_model().eval()
    # Zero-initialised output layers make a fresh model output zeros; perturb so
    # conditioning has something to act on.
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.02) if p.dim() > 1 else None
    x = torch.randn(2, 3, 16, 16)
    t = torch.tensor([10, 500])
    a = torch.zeros(2, len(VOCAB)); a[:, 3] = 1
    b = torch.zeros(2, len(VOCAB)); b[:, 5] = 1
    out_a, out_b, out_none = model(x, t, a), model(x, t, b), model(x, t)
    assert out_a.shape == x.shape
    assert not torch.allclose(out_a, out_b)
    assert not torch.allclose(out_a, out_none)


def test_cosine_schedule_is_valid():
    sched = DiffusionScheduler(timesteps=1000, schedule="cosine")
    assert torch.all(sched.betas > 0) and torch.all(sched.betas <= 0.999)
    assert sched.alphas_cumprod[0] > 0.99
    assert sched.alphas_cumprod[-1] < 1e-3
    assert torch.all(sched.alphas_cumprod[1:] <= sched.alphas_cumprod[:-1])


def test_ddim_guided_sampling_runs_and_is_seed_deterministic():
    model = _tiny_model().eval()
    sched = DiffusionScheduler(timesteps=100, schedule="cosine")
    cond = torch.zeros(2, len(VOCAB)); cond[:, 5] = 1

    def run(seed):
        g = torch.Generator().manual_seed(seed)
        return sched.ddim_sample(model, 16, batch_size=2, generator=g, cond=cond,
                                 guidance_scale=5.0, steps=5)

    first, second = run(1), run(1)
    assert first.shape == (2, 3, 16, 16)
    assert torch.isfinite(first).all()
    assert torch.allclose(first, second)


def test_checkpoint_roundtrip_through_load_model(tmp_path):
    from generation.image.generate import load_model, generate_images

    model = _tiny_model()
    path = tmp_path / "m.pt"
    torch.save({
        "model_state_dict": model.state_dict(), "conditional": True,
        "tag_vocab": VOCAB, "character_tags": ["vegeta"], "schedule": "cosine",
        "image_size": 16, "timesteps": 50, "base_channels": 32, "channels": 3,
        "arch": {"ch_mult": [1, 2], "num_res_blocks": 1, "attn_levels": [1]},
    }, path)
    loaded, size, steps = load_model(str(path))
    assert loaded.tag_vocab == VOCAB and size == 16 and steps == 50
    samples = generate_images(loaded, size, steps, num_images=2, seed=3,
                              tags=["vegeta", "blue_hair"], steps=4)
    assert samples.shape == (2, 3, 16, 16)
    assert samples.min() >= 0 and samples.max() <= 1