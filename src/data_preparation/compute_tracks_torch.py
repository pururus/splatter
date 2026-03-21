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


class TrackBatchDataset(Dataset):
    def __init__(self, frames, masks, frame_names, grid_size, resize_height, resize_width,
                 points_per_batch=128, max_batches_per_query=None, temporal_window=None):
        """
        Args:
            frames: Preprocessed frames tensor [num_frames, height, width, 3]
            masks: Mask tensor [num_frames, height, width]
            frame_names: List of frame names
            grid_size: Grid spacing for point sampling
            resize_height, resize_width: Target dimensions
            points_per_batch: Number of points per batch
            max_batches_per_query: Maximum batches per query frame (for memory control)
            temporal_window: Number of frames to keep in each sample track window
        """
        self.frames = frames
        self.masks = masks
        self.frame_names = frame_names
        self.grid_size = grid_size
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.points_per_batch = points_per_batch
        self.max_batches_per_query = max_batches_per_query
        self.temporal_window = temporal_window

        # Pre-compute grid points
        num_frames, height, width = masks.shape
        self.height, self.width = height, width
        y, x = np.mgrid[0:height:grid_size, 0:width:grid_size]
        y_resize, x_resize = y / (height - 1) * (resize_height - 1), x / (width - 1) * (resize_width - 1)

        # Create all query-target pairs
        self.samples = []
        for query_idx in range(num_frames):
            mask = masks[query_idx]
            in_mask = mask[y, x] > 0.5

            if in_mask.sum() == 0:
                continue

            # Get valid points for this query frame
            query_points = np.stack([y_resize[in_mask], x_resize[in_mask]], axis=-1)
            point_indices = np.where(in_mask.ravel())[0]

            # Split points into batches
            num_points = len(query_points)
            if max_batches_per_query:
                num_points = min(num_points, max_batches_per_query * points_per_batch)

            for i in range(0, num_points, points_per_batch):
                end_idx = min(i + points_per_batch, num_points)
                batch_points = query_points[i:end_idx]
                batch_indices = point_indices[i:end_idx]

                # Determine window around query frame (for memory control)
                if self.temporal_window is None or self.temporal_window >= num_frames:
                    window_start = 0
                    window_end = num_frames
                else:
                    half_window = self.temporal_window // 2
                    window_start = max(0, query_idx - half_window)
                    window_end = min(num_frames, window_start + self.temporal_window)
                    window_start = max(0, window_end - self.temporal_window)

                self.samples.append({
                    'query_idx': query_idx,
                    'point_indices': batch_indices,
                    'points': batch_points,
                    'batch_start': i,
                    'batch_end': end_idx,
                    'window_start': window_start,
                    'window_end': window_end
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        query_idx = sample['query_idx']

        # Create track queries for the current query frame only.
        window_start = sample['window_start']
        # (In the model, query time should be local to window_start)
        local_query_t = query_idx - window_start

        all_points = np.zeros((len(sample['points']), 3), dtype=np.float32)
        all_points[:, 0] = local_query_t  # time (in window coordinates)
        all_points[:, 1:] = sample['points']  # y, x coordinates (resized coords)

        return {
            'query_idx': query_idx,
            'point_indices': sample['point_indices'],
            'points': torch.from_numpy(all_points),
            'batch_start': sample['batch_start'],
            'batch_end': sample['batch_end'],
            'window_start': window_start,
            'window_end': sample['window_end']
        }


def collate_track_batch(batch):
    """Collate function for track batches."""
    query_indices = [item['query_idx'] for item in batch]
    point_indices = [item['point_indices'] for item in batch]
    points = torch.stack([item['points'] for item in batch])
    batch_starts = [item['batch_start'] for item in batch]
    batch_ends = [item['batch_end'] for item in batch]
    window_starts = [item['window_start'] for item in batch]
    window_ends = [item['window_end'] for item in batch]

    return {
        'query_indices': query_indices,
        'point_indices': point_indices,
        'points': points,
        'batch_starts': batch_starts,
        'batch_ends': batch_ends,
        'window_starts': window_starts,
        'window_ends': window_ends
    }


def process_batch_tracks(model, frames, batch_data, resize_width, resize_height, width, height, grid_size):
    """Process a batch of track queries efficiently (single model call, vectorized post-processing)."""
    points = batch_data['points'].to(frames.device)  # [B, num_points, 3]
    batch_size = points.shape[0]
    
    # Determine union of windows for frame slicing
    window_start = min(batch_data['window_starts'])
    window_end = max(batch_data['window_ends'])
    frames_window = frames[:, window_start:window_end]  # [1, window_len, H, W, 3]
    
    # Expand frames batch dimension to match points batch size [B, window_len, H, W, 3]
    frames_batch = frames_window.expand(batch_size, -1, -1, -1, -1)
    
    # Single model call with full batch
    with torch.inference_mode():
        preds = model(frames_batch, points)
    
    tracks = preds["tracks"].detach().cpu().numpy()  # [B, num_points, window_len, 2]
    occlusions = preds["occlusion"].detach().cpu().numpy()  # [B, num_points, window_len]
    expected_dist = preds["expected_dist"].detach().cpu().numpy()  # [B, num_points, window_len]
    
    # Vectorized coordinate conversion: flatten, convert, reshape
    batch_size_actual, num_points_actual, window_len, coord_dim = tracks.shape
    tracks_flat = tracks.reshape(-1, coord_dim)
    tracks_converted = transforms.convert_grid_coordinates(
        tracks_flat[None],
        (resize_width - 1, resize_height - 1),
        (width - 1, height - 1)
    )[0]
    tracks = tracks_converted.reshape(batch_size_actual, num_points_actual, window_len, coord_dim)
    
    # Pre-compute grid once (not per-item)
    y_grid, x_grid = np.mgrid[0:height:grid_size, 0:width:grid_size]
    y_grid = y_grid.astype(np.float32)
    x_grid = x_grid.astype(np.float32)
    y_grid_ravel = y_grid.ravel()
    x_grid_ravel = x_grid.ravel()
    
    # Combine outputs
    batch_outputs = []
    for i in range(batch_size):
        query_idx = batch_data['query_indices'][i]
        batch_tracks = tracks[i]
        batch_occlusions = occlusions[i]
        batch_expected_dist = expected_dist[i]
        
        # Get valid points using pre-computed grid
        valid_y = y_grid_ravel[batch_data['point_indices'][i]]
        valid_x = x_grid_ravel[batch_data['point_indices'][i]]
        
        # Set query frame coordinates to original grid positions
        local_query_idx = query_idx - window_start
        batch_tracks[:, local_query_idx, 0] = valid_y
        batch_tracks[:, local_query_idx, 1] = valid_x
        
        combined = np.concatenate([
            batch_tracks,
            batch_occlusions[..., None],
            batch_expected_dist[..., None]
        ], axis=-1)
        
        batch_outputs.append(combined)
    
    return batch_outputs, batch_data['query_indices'], batch_data['point_indices']


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
    parser.add_argument(
        "--model_type", type=str, choices=["tapir", "bootstapir"], help="model type"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="checkpoints",
        help="checkpoint dir",
    )
    # New batch processing arguments
    parser.add_argument("--batch_size", type=int, default=4, help="batch size for data loader")
    parser.add_argument("--num_workers", type=int, default=2, help="number of workers for data loader")
    parser.add_argument("--points_per_batch", type=int, default=128, help="points per batch")
    parser.add_argument("--max_batches_per_query", type=int, default=None, help="max batches per query frame")
    parser.add_argument("--max_video_frames", type=int, default=None, help="truncate video to this many frames before computing tracks")
    parser.add_argument("--temporal_window", type=int, default=None, help="number of frames to process in each track window; if None uses full sequence")
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
    masks = read_mask_video(mask_dir)
    masks = (masks.reshape((num_frames, height, width, -1)) > 0).any(axis=-1)

    if args.max_video_frames is not None and args.max_video_frames < num_frames:
        print(f"Truncating video to first {args.max_video_frames} frames (from {num_frames})")
        video = video[: args.max_video_frames]
        masks = masks[: args.max_video_frames]
        frame_names = frame_names[: args.max_video_frames]
        num_frames = args.max_video_frames

    print(f"{video.shape=} {masks.shape=} {masks.max()=} {masks.sum()=}")

    frames = media.resize_video(video, (resize_height, resize_width))
    print(f"{frames.shape=}")
    frames = torch.from_numpy(frames).to(device)
    frames = preprocess_frames(frames)[None]
    print(f"preprocessed {frames.shape=}")

    # Create batch dataset and data loader
    track_dataset = TrackBatchDataset(
        frames=frames,
        masks=masks,
        frame_names=frame_names,
        grid_size=grid_size,
        resize_height=resize_height,
        resize_width=resize_width,
        points_per_batch=args.points_per_batch,
        max_batches_per_query=args.max_batches_per_query,
        temporal_window=args.temporal_window
    )

    data_loader = DataLoader(
        track_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Keep order for deterministic results
        num_workers=args.num_workers,
        collate_fn=collate_track_batch,
        pin_memory=True
    )

    print(f"Created dataset with {len(track_dataset)} batches")

    # Process batches
    results_cache = {}  # Cache results by (query_idx, point_start, point_end)

    for batch_data in tqdm(data_loader, desc="Processing batches"):
        batch_outputs, query_indices, point_indices_list = process_batch_tracks(
            model, frames, batch_data, resize_width, resize_height, width, height, grid_size
        )

        # Store results
        for i, query_idx in enumerate(query_indices):
            batch_outputs_i = batch_outputs[i]
            point_indices = point_indices_list[i]
            window_start = batch_data['window_starts'][i]
            window_end = batch_data['window_ends'][i]
            window_len = window_end - window_start

            # Save results for each target frame in this window only
            for local_target_idx in range(window_len):
                target_idx = window_start + local_target_idx
                name_query = os.path.splitext(frame_names[query_idx])[0]
                name_target = os.path.splitext(frame_names[target_idx])[0]
                out_path = f"{out_dir}/{name_query}_{name_target}.npy"

                # Load existing results or create new array (global frame dimension)
                if os.path.exists(out_path):
                    existing_results = np.load(out_path)
                    if len(existing_results) < len(point_indices):
                        extended_results = np.zeros((len(point_indices), num_frames, 4), dtype=np.float32)
                        extended_results[:len(existing_results)] = existing_results
                        existing_results = extended_results
                else:
                    existing_results = np.zeros((len(point_indices), num_frames, 4), dtype=np.float32)

                # Update with new batch window results
                existing_results[:len(batch_outputs_i), target_idx] = batch_outputs_i[:, local_target_idx]
                np.save(out_path, existing_results)

    print("Batch processing completed!")


if __name__ == "__main__":
    main()