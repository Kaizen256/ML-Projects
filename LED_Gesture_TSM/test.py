import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import functional as TF

# =========================================================
# Dataset implementation (annotations-only, jpg-only)
# =========================================================

def _sorted_frames(video_dir: Path) -> List[Path]:
    """Return sorted .jpg frames in lexical order (works for ..._000001.jpg)."""
    frames = [p for p in video_dir.iterdir() if p.suffix.lower() == ".jpg"]
    return sorted(frames, key=lambda p: p.name)

@dataclass
class _Clip:
    video_id: str
    indices: List[int]   # absolute frame indices (0-based) for this clip
    label: Optional[int]

class IPNHandFrames(Dataset):
    """
    IPN-Hand dataset loader.
    - frames_root/<video_id>/*.jpg
    - annot_file lines: video_id, class_code, class_id, start, end, duration
    Only videos present locally are used. start/end assumed 1-based inclusive.
    """
    def __init__(
        self,
        frames_root: str | Path,
        annot_file: str | Path,
        clip_len: int = 16,
        resize_to: tuple[int, int] = (224, 224),
        normalize: bool = True,
        sample_within_segment: str = "center",  # "start" | "center" | "end"
        strict_missing: bool = False,
    ) -> None:
        self.root = Path(frames_root)
        self.clip_len = clip_len
        self.resize_to = resize_to
        self.normalize = normalize
        self.sample_within_segment = sample_within_segment
        self.strict_missing = strict_missing

        # Discover local videos
        self.videos: Dict[str, List[Path]] = {}
        for d in sorted(p for p in self.root.iterdir() if p.is_dir()):
            frames = _sorted_frames(d)
            if frames:
                self.videos[d.name] = frames
        if not self.videos:
            raise FileNotFoundError(f"No video frame folders found under {self.root}")

        # Build clips from annotations
        self.clips: List[_Clip] = []
        annot_path = Path(annot_file)
        if not annot_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annot_path}")

        with open(annot_path, "r", encoding="utf-8") as f:
            for line in f:
                row = [x.strip() for x in line.strip().split(",")]
                if not row or row[0].startswith("#"):
                    continue
                vid = row[0]
                if vid not in self.videos:
                    continue
                # class id
                try:
                    class_id = int(row[2]) if len(row) > 2 and row[2] != "" else -1
                except ValueError:
                    class_id = -1
                # start, end
                try:
                    start1 = int(row[3])  # 1-based
                    end1   = int(row[4])
                except Exception:
                    n = len(self.videos[vid])
                    start1, end1 = 1, n

                frames = self.videos[vid]
                n = len(frames)
                seg_start0 = max(0, start1 - 1)
                seg_end0   = min(n, end1)
                seg_len    = max(0, seg_end0 - seg_start0)
                if seg_len <= 0:
                    continue

                if seg_len < self.clip_len:
                    start0 = seg_start0
                    idxs = list(range(start0, min(start0 + self.clip_len, n)))
                else:
                    if self.sample_within_segment == "start":
                        start0 = seg_start0
                    elif self.sample_within_segment == "end":
                        start0 = seg_end0 - self.clip_len
                    else:  # center
                        mid = (seg_start0 + seg_end0 - self.clip_len) // 2
                        start0 = max(seg_start0, min(mid, seg_end0 - self.clip_len))
                    idxs = list(range(start0, start0 + self.clip_len))

                self.clips.append(_Clip(video_id=vid, indices=idxs, label=class_id))

        if len(self.clips) == 0:
            raise RuntimeError("No usable clips built from annotations.")

    def __len__(self) -> int:
        return len(self.clips)

    def _load_frame(self, path: Path) -> torch.Tensor:
        img = read_image(str(path)).float() / 255.0
        if self.resize_to is not None:
            img = TF.resize(img, self.resize_to, antialias=True)
        if self.normalize:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            img = (img - mean) / std
        return img

    def __getitem__(self, i: int):
        clip = self.clips[i]
        frames = self.videos[clip.video_id]
        imgs: List[torch.Tensor] = []
        for k in clip.indices:
            if 0 <= k < len(frames):
                imgs.append(self._load_frame(frames[k]))
            else:
                if self.strict_missing:
                    raise IndexError(f"Frame {k} out of range for {clip.video_id}")
                pad_k = max(0, min(k, len(frames)-1))
                imgs.append(self._load_frame(frames[pad_k]))
        x = torch.stack(imgs, dim=0)  # [T,C,H,W]
        y = torch.tensor(-1 if clip.label is None else int(clip.label), dtype=torch.long)
        return {"video_id": clip.video_id, "frames": x, "label": y}

# =========================================================
# Collate
# =========================================================
def pad_collate(batch: Sequence[dict]):
    vids = [b["video_id"] for b in batch]
    ys   = torch.stack([b["label"] for b in batch])
    xs   = [b["frames"] for b in batch]
    T    = max(t.shape[0] for t in xs)
    C,H,W = xs[0].shape[1:]
    out  = torch.zeros((len(xs), T, C, H, W), dtype=xs[0].dtype)
    for i, t in enumerate(xs):
        out[i, :t.shape[0]] = t
        if t.shape[0] < T:
            out[i, t.shape[0]:] = t[-1:]
    return {"video_id": vids, "frames": out, "label": ys}

# =========================================================
# Main test
# =========================================================
def main():
    FRAMES_ROOT = "frames"
    ANNOT_FILE  = "annotations/Annot_TrainList.txt"

    ds = IPNHandFrames(
        frames_root=FRAMES_ROOT,
        annot_file=ANNOT_FILE,
        clip_len=16,
        resize_to=(224,224),
        normalize=True,
    )

    dl = DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        num_workers=2,    # safe for Jupyter/Windows
        collate_fn=pad_collate,
    )

    for batch in dl:
        x, y = batch["frames"], batch["label"]
        print("Batch frames shape:", x.shape)  # [B,T,C,H,W]
        print("Batch labels shape:", y.shape)  # [B]
        print("Video IDs:", batch["video_id"])
        break

if __name__ == "__main__":
    main()
