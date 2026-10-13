import torch
import torch.nn.functional as F

SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"                         
ACT_DTYPE = torch.float16 if USE_AMP else torch.float32 

MEAN = torch.tensor((0.5071, 0.4867, 0.4408), device=device).view(1, 3, 1, 1)
STD = torch.tensor((0.2675, 0.2565, 0.2761), device=device).view(1, 3, 1, 1)

torch.backends.cudnn.benchmark = True
torch.manual_seed(SEED)

def normalize(x_u8):
    """uint8 (B, 3, H, W) -> normalized float32, channels_last."""
    x = (x_u8.float().div_(255) - MEAN) / STD
    return x.contiguous(memory_format=torch.channels_last)


def augment_batch(x_u8, padding=4):
    """RandomCrop(32, padding=4) + RandomHorizontalFlip, independently per sample, on the GPU.
    Pads with zeros *before* normalization, exactly like the torchvision pipeline."""
    B, C, H, W = x_u8.shape
    dev = x_u8.device
    x = F.pad(x_u8, (padding,) * 4)
    i = torch.randint(0, 2 * padding + 1, (B,), device=dev)
    j = torch.randint(0, 2 * padding + 1, (B,), device=dev)
    rows = (i[:, None] + torch.arange(H, device=dev))[:, None, :, None]    # (B, 1, H, 1)
    cols = (j[:, None] + torch.arange(W, device=dev))[:, None, None, :]    # (B, 1, 1, W)
    b = torch.arange(B, device=dev)[:, None, None, None]
    c = torch.arange(C, device=dev)[None, :, None, None]
    x = x[b, c, rows, cols]                                                # (B, C, H, W)
    flip = torch.rand(B, device=dev) < 0.5
    return torch.where(flip[:, None, None, None], x.flip(3), x)


def make_epoch_tensor(X_u8, perm=None, augment=False, chunk=4096):
    n = X_u8.size(0)
    out = torch.empty(X_u8.shape, dtype=ACT_DTYPE, device=X_u8.device, memory_format=torch.channels_last)
    for s in range(0, n, chunk):
        xb = X_u8[s : s + chunk] if perm is None else X_u8[perm[s : s + chunk]]
        if augment:
            xb = augment_batch(xb)
        out[s : s + chunk] = normalize(xb)
    return out