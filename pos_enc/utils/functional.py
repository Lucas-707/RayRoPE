import time
from collections import namedtuple
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

Camera = namedtuple("Camera", ["K", "camtoworld", "width", "height"])


def random_SO3(batch_size: Tuple[int], device="cpu"):
    # Step 1: Generate a batch of random matrices of shape (batch_size, 3, 3)
    random_matrices = torch.randn((*batch_size, 3, 3), device=device)
    random_matrices = random_matrices.reshape(-1, 3, 3)

    # Step 2: Apply QR decomposition to each matrix in the batch
    # The `torch.linalg.qr` function works for batches of matrices in newer PyTorch versions
    q, r = torch.linalg.qr(random_matrices)
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))[..., None, :]

    # Step 3: Adjust for positive determinant in each matrix
    # Compute the determinants and find indices where the determinant is negative
    det_q = torch.det(q)
    negative_det_indices = det_q < 0

    # Flip the sign of the last column where determinant is negative
    q[negative_det_indices, :, 2] *= -1
    q = q.reshape(*batch_size, 3, 3)

    return q


def random_SE3(batch_size: Tuple[int], device="cpu"):
    random_matrices = torch.eye(4, device=device).repeat(*batch_size, 1, 1)
    random_matrices[..., :3, :3] = random_SO3(batch_size, device)
    random_matrices[..., :3, 3] = torch.randn(*batch_size, 3, device=device)
    return random_matrices


def patchify(x: Tensor, patch_size: int) -> Tensor:
    """Split an image tensor into patches.

    Args:
        x: The input image tensor with shape (..., H * P, W * P, C).
        patch_size: The size of the patch.

    Returns:
        The output tensor with shape (..., H * W, P * P * C).
    """
    assert (
        x.shape[-3] % patch_size == 0
    ), "Expected height to be divisible by patch_size."
    assert (
        x.shape[-2] % patch_size == 0
    ), "Expected width to be divisible by patch_size."

    x = rearrange(
        x, "... (h ph) (w pw) c -> ... (h w) (ph pw c)", ph=patch_size, pw=patch_size
    )
    return x


def unpatchify(x: Tensor, height: int, width: int, patch_size: int) -> Tensor:
    """Combine patches into an image tensor.

    Args:
        x: The input tensor with shape (..., H * W, P * P * C).
        height: The height of the original image.
        width: The width of the original image.
        patch_size: The size of the patch.

    Returns:
        The output tensor with shape (..., H * P, W * P, C).
    """
    assert height % patch_size == 0, "Expected height to be divisible by patch_size."
    assert width % patch_size == 0, "Expected width to be divisible by patch_size."

    x = rearrange(
        x,
        "... (h w) (ph pw c) -> ... (h ph) (w pw) c",
        h=height // patch_size,
        w=width // patch_size,
        ph=patch_size,
        pw=patch_size,
    )
    return x


def camera_to_raymap(
    Ks: Tensor,
    camtoworlds: Tensor,
    height: int,
    width: int,
    downscale: int = 1,
    include_ups: bool = False,
):
    """Construct the raymap from the camera intrinsics and extrinsics.

    Note: This function expects OpenCV camera coordinates.

    Args:
        Ks: The camera intrinsics tensor with shape (..., 3, 3).
        camtoworlds: The camera extrinsics tensor with shape (..., 4, 4).
        height: The height of original image corresponding to intrinsics.
        width: The width of original image corresponding to intrinsics.
        downscale: Downscale factor for the raymap.
        include_ups: Whether to include the up direction in the raymap.

    Returns:
        The raymap tensor with shape (..., H, W, 6).
    """
    assert Ks.shape[-2:] == (3, 3), "Expected Ks to have shape (..., 3, 3)."
    assert camtoworlds.shape[-2:] == (
        4,
        4,
    ), "Expected camtoworlds to have shape (..., 4, 4)."
    assert width % downscale == 0, "Expected width to be divisible by downscale."
    assert height % downscale == 0, "Expected height to be divisible by downscale."

    # Downscale the intrinsics.
    Ks = torch.stack(
        [
            Ks[..., 0, :] / downscale,
            Ks[..., 1, :] / downscale,
            Ks[..., 2, :],
        ],
        dim=-2,
    )  # [..., 3, 3]
    width //= downscale
    height //= downscale

    # Construct pixel coordinates
    x, y = torch.meshgrid(
        torch.arange(width, device=Ks.device),
        torch.arange(height, device=Ks.device),
        indexing="xy",
    )  # [H, W]
    coords = torch.stack([x + 0.5, y + 0.5, torch.ones_like(x)], dim=-1)  # [H, W, 3]

    # To camera coordinates [..., H, W, 3]
    dirs = torch.einsum("...ij,...hwj->...hwi", Ks.inverse(), coords)

    # To world coordinates [..., H, W, 3]
    dirs = torch.einsum("...ij,...hwj->...hwi", camtoworlds[..., :3, :3], dirs)
    dirs = F.normalize(dirs, p=2, dim=-1)

    # Camera origin in world coordinates [..., H, W, 3]
    origins = torch.broadcast_to(camtoworlds[..., None, None, :3, -1], dirs.shape)

    if include_ups:
        # Extract the up direction (second column)
        ups = torch.broadcast_to(camtoworlds[..., None, None, :3, 1], dirs.shape)
        ups = F.normalize(ups, p=2, dim=-1)
        return torch.cat([origins, dirs, ups], dim=-1)
    else:
        return torch.cat([origins, dirs], dim=-1)  # [..., H, W, 6]


def raymap_to_plucker(raymap: Tensor) -> Tensor:
    """Convert raymap to Plücker coordinates.

    Args:
        raymap: The raymap tensor with shape (..., H, W, 6).

    Returns:
        The Plücker coordinates tensor with shape (..., H, W, 3).
    """
    assert raymap.shape[-1] == 6, "Expected raymap to have shape (..., H, W, 6)."
    ray_origins, ray_directions = torch.split(raymap, [3, 3], dim=-1)
    # Normalize ray directions to unit vectors
    ray_directions = F.normalize(ray_directions, p=2, dim=-1)
    plucker_normal = torch.cross(ray_origins, ray_directions, dim=-1)
    return torch.cat([ray_directions, plucker_normal], dim=-1)


def timeit(repeats: int, f: Callable, *args, **kwargs) -> float:
    torch.cuda.reset_peak_memory_stats()
    mem_tic = torch.cuda.max_memory_allocated() / 1024**3

    for _ in range(5):  # warmup
        f(*args, **kwargs)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeats):
        results = f(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.time()

    mem = torch.cuda.max_memory_allocated() / 1024**3 - mem_tic
    return (end - start) / repeats, mem, results
