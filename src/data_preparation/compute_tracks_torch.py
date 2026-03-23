import argparse
import glob
import os

import imageio
import mediapy as media
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tapnet_torch import tapir_model, transforms
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

class PointsDataset(torch.utils.data.Dataset):
    """Dataset over all (query frame, point-chunk) pairs."""

    def __init__(self, all_points, chunk_size=128):
        self.base_points = all_points
        self.chunk_size = chunk_size

        self.point_chunks = np.array_split(
            self.base_points,
            indices_or_sections=max(1, int(np.ceil(len(self.base_points) / self.chunk_size))),
            axis=0,
        )
        
    def __len__(self):
        return len(self.point_chunks)

    def __getitem__(self, idx):
        points = self.point_chunks[idx]
        points[:, 0] = float(idx)

        return torch.from_numpy(points.astype(np.float32))
    
class FramesDataset(torch.utils.data.Dataset):
    def __init__(self, frames, window_size=128):
        self.frames = frames
        self.window_size = window_size

        self.frame_chunks = np.array_split(
            self.frames,
            indices_or_sections=max(1, int(np.ceil(len(self.frames) / self.window_size))),
            axis=0,
        )
        
    def __len__(self):
        return len(self.frame_chunks)

    def __getitem__(self, idx):
        frame = self.frame_chunks[idx]
        return frame

def read_video(folder_path):
    frame_paths = sorted(glob.glob(os.path.join(folder_path, "*")))
    video = np.stack([imageio.imread(frame_path) for frame_path in frame_paths])
    print(f"{video.shape=} {video.dtype=} {video.min()=} {video.max()=}")
    video = media._VideoArray(video)
    return video

def read_mask_video(folder_path):
    frame_paths = sorted(glob.glob(os.path.join(folder_path, "*")))
    img_list = [imageio.imread(frame_path) for frame_path in frame_paths]
    img_list = [img[...,0] if img.ndim == 3 else img for img in img_list]
    video = np.stack(img_list)
    print(f"{video.shape=} {video.dtype=} {video.min()=} {video.max()=}")
    video = media._VideoArray(video)
    return video


def preprocess_frames(frames):
    """Preprocess frames to model inputs.

    Args:
      frames: [num_frames, height, width, 3], [0, 255], np.uint8

    Returns:
      frames: [num_frames, height, width, 3], [-1, 1], np.float32
    """
    frames = frames.float()
    frames = frames / 255 * 2 - 1
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="image dir")
    parser.add_argument("--mask_dir", type=str, required=True, help="mask dir")
    parser.add_argument("--out_dir", type=str, required=True, help="out dir")
    parser.add_argument("--grid_size", type=int, default=4, help="grid size")
    parser.add_argument("--resize_height", type=int, default=256, help="resize height")
    parser.add_argument("--resize_width", type=int, default=256, help="resize width")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--chunk_size", type=int, default=128, help="chunk size for points")
    parser.add_argument("--window_size", type=int, default=128, help="window size for frames")
    parser.add_argument("--first_frame", type=int, default=None, help="chunk size for points")
    parser.add_argument("--last_frame", type=int, default=None, help="window size for frames")
    parser.add_argument(
        "--model_type", type=str, choices=["tapir", "bootstapir"], help="model type"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="checkpoints",
        help="checkpoint dir",
    )
    args = parser.parse_args()

    folder_path = args.image_dir
    mask_dir = args.mask_dir
    frame_names = [
        os.path.basename(f) for f in sorted(glob.glob(os.path.join(folder_path, "*")))
    ]
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    done = True
    for t in range(len(frame_names)):
        for j in range(len(frame_names)):
            name_t = os.path.splitext(frame_names[t])[0]
            name_j = os.path.splitext(frame_names[j])[0]
            out_path = f"{out_dir}/{name_t}_{name_j}.npy"
            if not os.path.exists(out_path):
                done = False
                break
    print(f"{done=}")
    if done:
        print("Already done")
        return

    ## Load model
    ckpt_file = (
        "tapir_checkpoint_panning.pt"
        if args.model_type == "tapir"
        else "bootstapir_checkpoint_v2.pt"
    )
    ckpt_path = os.path.join(args.ckpt_dir, ckpt_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tapir_model.TAPIR(pyramid_level=1)
    model.load_state_dict(torch.load(ckpt_path))
    model = model.to(device)

    resize_height = args.resize_height
    resize_width = args.resize_width
    grid_size = args.grid_size

    video = read_video(folder_path)
    num_frames, height, width = video.shape[0:3]
    
    frames = media.resize_video(video, (resize_height, resize_width))
    if args.last_frame is not None:
        frames = frames[:args.last_frame]
    if args.first_frame is not None:
        frames = frames[args.first_frame:]
    frames = torch.from_numpy(frames)
    frames = preprocess_frames(frames)
    
    num_frames = frames.shape[0]

    y, x = np.mgrid[0:height:grid_size, 0:width:grid_size]
    y_resize = y / (height - 1) * (resize_height - 1)
    x_resize = x / (width - 1) * (resize_width - 1)

    all_points = np.stack([np.zeros_like(y, dtype=np.float32), y_resize, x_resize], axis=-1).reshape(-1, 3)
    print("Points shape:", all_points.shape)
    
    points_dataset = PointsDataset(all_points, chunk_size=args.chunk_size)
    frames_dataset = FramesDataset(frames, window_size=args.window_size)

    points_loader = DataLoader(
        points_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 0,
    )

    for t in tqdm(range(num_frames), desc=f"Processing time {t}/{num_frames}"):
        name_t = os.path.splitext(frame_names[t])[0]
        outputs = []
        
        for points in tqdm(points_loader, desc=f"Processing points chunks"):
            t_points = points.to(device) * torch.tensor([t, 1, 1], device=device) 
            batch_output = []
            
            for frames_chunk in tqdm(frames_dataset, desc=f"Processing frames for frame {t}"):
                frames_chunk = frames_chunk.to(device)
                frames_chunk = frames_chunk.unsqueeze(0).repeat(t_points.shape[0], 1, 1, 1, 1)

                print(frames_chunk.shape)
                with torch.inference_mode():
                    preds = model(frames_chunk, t_points)

                tracks = preds["tracks"].detach().cpu().numpy()
                tracks = transforms.convert_grid_coordinates(
                    tracks, (resize_width - 1, resize_height - 1), (width - 1, height - 1)
                )
                occlusions = preds["occlusion"].detach().cpu().numpy()
                expected_dist = preds["expected_dist"].detach().cpu().numpy()
                
                batch_output.append(np.concatenate([tracks, occlusions[..., None], expected_dist[..., None]], axis=-1))
                print("batch shape", np.concatenate([tracks, occlusions[..., None], expected_dist[..., None]], axis=-1).shape)
            
            outputs.append(np.concatenate(batch_output, axis=1).reshape(tracks.shape[0] * tracks.shape[1], num_frames, 4)) 
        outputs = np.concatenate(outputs, axis=0)

        print("Outputs has  shape:", outputs.shape)
        for j in range(num_frames):
            name_j = os.path.splitext(frame_names[j])[0]
            if j == t:
                outputs[:, j, :2] = np.stack([x.reshape(-1), y.reshape(-1)], axis=-1)
            np.save(f"{out_dir}/{name_t}_{name_j}.npy", outputs[:, j])
            
if __name__ == "__main__":
    main()