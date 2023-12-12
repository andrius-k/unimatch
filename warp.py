import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2


def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape testdata into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))
        

def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid


def bilinear_sample(img, sample_coords, mode='bilinear', padding_mode='zeros', return_mask=False):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = torch.nn.functional.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)  # [B, H, W]

        return img, mask

    return img


def flow_warp(feature, flow, mask=False, padding_mode='zeros'):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w).to(flow.device) + flow  # [B, 2, H, W]

    return bilinear_sample(feature, grid, padding_mode=padding_mode,
                           return_mask=mask)


def forward_backward_consistency_check(fwd_flow, bwd_flow,
                                       alpha=0.01,
                                       beta=0.5
                                       ):
    # fwd_flow, bwd_flow: [B, 2, H, W]
    # alpha and beta values are following UnFlow (https://arxiv.org/abs/1711.07837)
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)  # [B, H, W]

    warped_bwd_flow = flow_warp(bwd_flow, fwd_flow)  # [B, 2, H, W]
    warped_fwd_flow = flow_warp(fwd_flow, bwd_flow)  # [B, 2, H, W]

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)  # [B, H, W]
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    threshold = alpha * flow_mag + beta

    fwd_occ = (diff_fwd > threshold).float()  # [B, H, W]
    bwd_occ = (diff_bwd > threshold).float()

    return fwd_occ, bwd_occ


def warp_with_flow(flow2d, fwd_occ, source1_id_str, source2_id_str):
    img_1 = cv2.imread(f"/cluster/work/cvl/haofxu/dynamic_nerf/data/Sequence_7/main/camera_{source1_id_str}.jpeg") # (H, W, 3)
    img_1 = img_1.astype(np.float32) / (255.0 * 1)
    img_2 = cv2.imread(f"/cluster/work/cvl/haofxu/dynamic_nerf/data/Sequence_7/main/camera_{source2_id_str}.jpeg") # (H, W, 3)
    img_2 = img_2.astype(np.float32) / (255.0 * 1)

    # img_1 = cv2.resize(img_1, (320, 576))
    # img_2 = cv2.resize(img_2, (320, 576))
    
    h, w = img_1.shape[0:2]

    # img_1, img_2 = img_2, img_1

    flow2d = torch.from_numpy(flow2d)
    img_1 = torch.from_numpy(img_1) # (H, W, 3)
    img_2 = torch.from_numpy(img_2) # (H, W, 3)

    img_2_input = img_2.permute(2, 0, 1)[None, ...] # (1, 3, H, W)

    y0, x0 = torch.meshgrid(
        torch.arange(h).to(flow2d.device).float(), 
        torch.arange(w).to(flow2d.device).float())
    x1 = x0 + flow2d[...,0]
    y1 = y0 + flow2d[...,1]

    x1_norm = x1 / w
    x1_norm = x1_norm * 2
    x1_norm -= 1
    y1_norm = y1 / h
    y1_norm = y1_norm * 2
    y1_norm -= 1

    grid = torch.stack((x1_norm, y1_norm)).permute(1, 2, 0).unsqueeze(dim=0)

    print(img_2_input.shape)
    print(grid.shape)
    
    img_2_warped = torch.nn.functional.grid_sample(img_2_input, grid, mode="bilinear", padding_mode="border") # (1, 3, H, W)
    img_2_warped = img_2_warped.squeeze() # (3, H, W)
    img_2_warped = img_2_warped.permute(1, 2, 0) # (H, W, 3)

    # img_2_warped[fwd_occ == 1] = img_1[fwd_occ == 1]
    img_2_warped[fwd_occ == 1] = 0
    
    stack = np.hstack((img_2_warped.numpy() * 255, img_1.numpy() * 255, img_2.numpy() * 255))
    cv2.imwrite("warped.jpg", stack)
    cv2.imwrite("occ.jpg", fwd_occ.numpy() * 255)


source1_id = 415
source2_id = 439
source1_id_str = str(source1_id).zfill(4)
source2_id_str = str(source2_id).zfill(4)

flow2d_fwd = readFlow(f"output/flow_{source1_id_str}_{source2_id_str}_fwd.flo")
flow2d_bwd = readFlow(f"output/flow_{source1_id_str}_{source2_id_str}_bwd.flo")

# flow2d_fwd = flow2d_fwd[:, :720, :]
# flow2d_bwd = flow2d_bwd[:, :720, :]

flow_fwd = torch.from_numpy(flow2d_fwd).permute(2, 0, 1).unsqueeze(0) # [1, 2, 1280, 720]
flow_bwd = torch.from_numpy(flow2d_bwd).permute(2, 0, 1).unsqueeze(0) # [1, 2, 1280, 720]
fwd_occ, bwd_occ = forward_backward_consistency_check(flow_fwd, flow_bwd, alpha=0.5, beta=5.0)
fwd_occ, bwd_occ = fwd_occ.squeeze(0), bwd_occ.squeeze(0) # [1280, 720]

warp_with_flow(flow2d_fwd, fwd_occ, source1_id_str, source2_id_str)
