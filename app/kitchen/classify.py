"""Stage 3: label. Zero-shot CLIP scores every downscaled photo against every menu dish.

Items that share the same visual prompt (the same dish sold under two brands) are scored
once and share the result, so identical dishes never split probability between themselves."""
import numpy as np
from PIL import Image
from tqdm import tqdm

from . import catalog, config

BATCH = 32


def run(con) -> dict:
    import open_clip
    import torch

    items = catalog.active()      # uncooked items are not candidates
    prompts = list(dict.fromkeys(it["clip_prompt"] for it in items))
    texts = [f"a professional food photo of {p}, Indian cuisine" for p in prompts]
    texts += [t for t in config.REJECT_PROMPTS.values()]
    classes = prompts + list(config.REJECT_PROMPTS)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED, device=dev)
    model.eval()
    tok = open_clip.get_tokenizer(config.CLIP_MODEL)

    rows = con.execute(
        "SELECT id,down_path FROM files WHERE status='downscaled' ORDER BY id").fetchall()
    ids = [r["id"] for r in rows]
    with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
        t = model.encode_text(tok(texts).to(dev))
        t = t / t.norm(dim=-1, keepdim=True)
        scale = model.logit_scale.exp().item()
        out, feats = [], []
        for i in tqdm(range(0, len(rows), BATCH), unit="batch"):
            batch = torch.stack([preprocess(Image.open(r["down_path"]).convert("RGB"))
                                 for r in rows[i:i + BATCH]]).to(dev)
            f = model.encode_image(batch)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.float().cpu().numpy())
            out.append((scale * f @ t.T).softmax(dim=-1).float().cpu().numpy())
    probs = np.concatenate(out) if out else np.zeros((0, len(classes)))
    np.savez_compressed(config.EMBEDDINGS, ids=np.array(ids), classes=np.array(classes),
                        prompts=np.array(prompts), probs=probs,
                        feats=np.concatenate(feats) if feats else np.zeros((0, 1)))
    return {"photos": len(ids), "classes": len(classes), "device": dev}
