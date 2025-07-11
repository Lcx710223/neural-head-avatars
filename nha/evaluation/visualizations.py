### LCX修改20250622 V4
### 错误关键行来自 visualizations.py 的 generate_novel_view_folder：Image.open(img).save(f_outpath)  FileNotFoundError: [Errno 2] No such file or directory: 'rgba_pred'
### 这说明 img 被赋值了字符串 'rgba_pred'，而不是 tensor 或文件路径。
### LCX20250623 58-64行。原文是从YAML调取5个参数，但是与保存的CKPTS里的数据不能对应。现在修改为直接从CKPTS里直接读取5个超参数。
### LCX20250710 修改。

import torch
import os
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.utils as vutils


def plot_grid(tensor_list, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
    """
    Make a grid of images.
    Based on torchvision.utils.make_grid but handles a list of tensors.
    """
    if not tensor_list:
        return torch.empty(0) # Return empty if the list is empty

    # Ensure all tensors are on the same device and have the same shape (except batch dimension)
    first_tensor = tensor_list[0]
    for tensor in tensor_list:
        if tensor.device != first_tensor.device or tensor.shape[1:] != first_tensor.shape[1:]:
            # Attempt to move to first device and resize/crop if necessary (basic handling)
            # More robust handling might be needed based on expected input
            print("[93m[WARNING]: Tensors in list have different devices or shapes. Attempting basic handling.[0m")
            # Example basic resize (assuming all are images C x H x W)
            try:
                tensor = tensor.to(first_tensor.device)
                if tensor.shape[1:] != first_tensor.shape[1:]:
                    from torchvision.transforms.functional import resize
                    tensor = resize(tensor, first_tensor.shape[1:])
            except Exception as e:
                 print(f"[91m[ERROR]: Could not process tensor in plot_grid: {e}[0m")
                 continue # Skip this tensor if processing fails


    # Concatenate all tensors into a single batch
    try:
        grid_tensor = torch.cat(tensor_list, dim=0)
    except RuntimeError as e:
         print(f"[91m[ERROR]: Could not concatenate tensors in plot_grid: {e}[0m")
         return torch.empty(0) # Return empty if concatenation fails


    # Use torchvision's make_grid
    grid = vutils.make_grid(grid_tensor, nrow=nrow, padding=padding, normalize=normalize, range=range, scale_each=scale_each, pad_value=pad_value)

    return grid


def save_image_grid(tensor_list, filename, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
    """
    Saves a grid of images from a list of tensors.
    """
    grid = plot_grid(tensor_list, nrow=nrow, padding=padding, normalize=normalize, range=range, scale_each=scale_each, pad_value=pad_value)

    if grid.numel() == 0: # Check if the grid is empty
        print(f"[93m[WARNING]: Cannot save empty image grid to {filename}.[0m")
        return

    # Ensure the directory exists
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Convert to numpy and save using OpenCV
    # Assuming the input tensors are in C x H x W format and range [0, 1]
    grid_np = grid.permute(1, 2, 0).mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8).numpy()

    # Convert RGB to BGR for OpenCV
    grid_np = cv2.cvtColor(grid_np, cv2.COLOR_RGB2BGR)

    cv2.imwrite(filename, grid_np)


def reconstruct_sequence(model, data_module, split="train", angles=[[0, 0]], output_dir="./reconstruction"):
    """
    Reconstructs a sequence of images from a dataset and saves them to a folder.
    """
    # Ensure the model is in evaluation mode
    model.eval()
    device = next(model.parameters()).device # Get model device

    # Get the appropriate dataset based on the split
    # Note: Assuming RealDataModule now stores _train_set and _val_set
    if split == "train":
        dataset = data_module._train_set
    elif split == "val":
        dataset = data_module._val_set
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")

    # Create a DataLoader for the dataset
    # Use a batch size of 1 for sequence reconstruction
    # Ensure dataset is not None and has length > 0
    if dataset is None or len(dataset) == 0:
        print(f"[93m[WARNING]: Dataset for split '{split}' is empty. Skipping reconstruction.[0m")
        return torch.empty(0)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=data_module.data_worker)


    pred_rgb = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc=f"Reconstructing {split} sequence")):
            if batch is None: # Skip if the batch is None (e.g., due to missing files)
                print(f"[93m[WARNING]: Skipping batch {i} as it is None.[0m")
                continue

            # Move batch to the correct device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # Perform reconstruction for each angle
            for angle in angles:
                # Create a modified batch with the new camera rotation
                modified_batch = batch.copy()
                # Assuming camera_R is a rotation matrix or can be converted
                # This part might need adjustment based on how camera_R is represented
                # Example: Assuming camera_R is a 3x3 rotation matrix
                # If it's axis-angle or quaternion, conversion is needed
                if 'camera_R' in modified_batch and modified_batch['camera_R'].shape[-2:] == (3, 3):
                     # Simple rotation around Y axis for example
                     # This angle modification logic might need to match the project's camera model
                     # Convert angle degrees to tensor and move to device
                     angle_deg_tensor = torch.tensor(angle, dtype=torch.float32).to(device)
                     angle_rad_tensor = torch.deg2rad(angle_deg_tensor)

                     # Construct rotation matrices for Y and X (pitch and yaw)
                     # This assumes angle[0] is yaw (rotation around Y) and angle[1] is pitch (rotation around X)
                     # The order of multiplication matters and depends on the convention used in the project
                     rot_y = torch.eye(3, device=device)
                     rot_y[0, 0] = torch.cos(angle_rad_tensor[0])
                     rot_y[0, 2] = torch.sin(angle_rad_tensor[0])
                     rot_y[2, 0] = -torch.sin(angle_rad_tensor[0])
                     rot_y[2, 2] = torch.cos(angle_rad_tensor[0])

                     rot_x = torch.eye(3, device=device)
                     rot_x[1, 1] = torch.cos(angle_rad_tensor[1])
                     rot_x[1, 2] = -torch.sin(angle_rad_tensor[1])
                     rot_x[2, 1] = torch.sin(angle_rad_tensor[1])
                     rot_x[2, 2] = torch.cos(angle_rad_tensor[1])

                     # Apply the new rotation - combine original rotation with novel view rotation
                     # This assumes original camera_R is the identity or initial orientation
                     # If original camera_R is the orientation for the input image,
                     # you might need to multiply the novel view rotation with it.
                     # Example: New_R = Novel_View_Rotation @ Original_R or Original_R @ Novel_View_Rotation
                     # Let's assume Novel_View_Rotation is applied relative to the original orientation
                     # The order (rot_y @ rot_x) or (rot_x @ rot_y) depends on Euler angle convention
                     # Let's use Y then X for example
                     novel_view_rotation = torch.matmul(rot_y, rot_x)

                     # Apply the novel view rotation to the original camera rotation
                     # Assuming original camera_R is batch_size x 3 x 3
                     original_camera_R = modified_batch['camera_R'] # Shape: B x 3 x 3
                     # Ensure novel_view_rotation is broadcastable or match batch size
                     # Assuming novel_view_rotation is 3x3, broadcast it to match batch size
                     novel_view_rotation = novel_view_rotation.unsqueeze(0).expand(original_camera_R.size(0), -1, -1)

                     # Apply the new rotation (example: multiply)
                     # This multiplication order might need to be adjusted based on how camera_R is applied in the model
                     # Example: new_camera_R = novel_view_rotation @ original_camera_R
                     # Or: new_camera_R = original_camera_R @ novel_view_rotation
                     # Let's assume novel view rotation is applied before original camera_R's effect
                     modified_batch['camera_R'] = torch.matmul(novel_view_rotation, original_camera_R) # Example, adjust as needed


                else:
                    # Handle cases where camera_R is not a 3x3 matrix or not present
                    print(f"[93m[WARNING]: camera_R not in batch or not a 3x3 matrix for frame {batch.get('frame_id', 'N/A')}. Skipping angle modification for angle {angle}.[0m")
                    # If angle modification is essential, you might need to skip this angle or sample
                    # For now, we continue with the original camera_R if it exists, or without camera_R if not loaded
                    if 'camera_R' not in modified_batch:
                        print(f"[93m[WARNING]: camera_R not found in batch for frame {batch.get('frame_id', 'N/A')}. Novel view generation might not work as expected.[0m")
                        # You might need to create a default camera_R or skip if camera is essential

                # Get the predicted RGB image
                # Assuming the model's forward pass returns a dictionary including 'pred_rgb'
                try:
                    model_output = model(modified_batch)
                    if 'pred_rgb' in model_output:
                        # Assuming model_output['pred_rgb'] is B x C x H x W
                        pred_rgb.append(model_output['pred_rgb'].detach().cpu().squeeze(0)) # Append B=1 tensor, squeeze batch dim
                    else:
                         print(f"[93m[WARNING]: Model output does not contain 'pred_rgb' for frame {batch.get('frame_id', 'N/A')} at angle {angle}.[0m")
                except Exception as e:
                    print(f"[91m[ERROR]: Error during model prediction for frame {batch.get('frame_id', 'N/A')} at angle {angle}: {e}[0m")
                    # Optionally, you might want to append a placeholder or skip


    # Check if pred_rgb is empty before concatenating
    if not pred_rgb:
        print(f"[91m[ERROR]: No predicted RGB tensors were generated for the {split} sequence. The pred_rgb list is empty.[0m")
        # Return an empty tensor or handle appropriately
        return torch.empty(0, 3, data_module.img_res[1], data_module.img_res[0]) # Return an empty tensor with correct shape dims if possible

    # Concatenate the predicted RGB images
    # Ensure tensors have the same shape before concatenating (except batch dim)
    # This might be necessary if some predictions failed or had different sizes
    if len(pred_rgb) > 1:
        first_pred_shape = pred_rgb[0].shape
        for i in range(1, len(pred_rgb)):
            if pred_rgb[i].shape != first_pred_shape:
                 print(f"[93m[WARNING]: Predicted RGB tensors have different shapes. Attempting resize for concatenation.[0m")
                 # Attempt to resize to the shape of the first tensor
                 try:
                     from torchvision.transforms.functional import resize
                     pred_rgb[i] = resize(pred_rgb[i], first_pred_shape[1:])
                 except Exception as e:
                     print(f"[91m[ERROR]: Could not resize predicted tensor for concatenation: {e}[0m")
                     # Optionally, remove the problematic tensor
                     # pred_rgb.pop(i) # This would require adjusting the loop or using a new list
                     pass # Continue for now, but concatenation might still fail

    try:
        pred_rgb_concatenated = torch.cat(pred_rgb, dim=0) # This line will no longer error if pred_rgb is empty
    except RuntimeError as e:
         print(f"[91m[ERROR]: Final concatenation of predicted RGB tensors failed: {e}[0m")
         return torch.empty(0, 3, data_module.img_res[1], data_module.img_res[0])


    # Save the reconstructed sequence (example - you'll need to implement saving)
    # This part is project-specific and depends on how you want to save the output
    # Example: Convert to numpy and save as images or a video
    # os.makedirs(output_dir, exist_ok=True)
    # for i, img_tensor in enumerate(pred_rgb_concatenated):
    #     img_np = img_tensor.permute(1, 2, 0).numpy() * 255.0
    #     img_np = img_np.astype(np.uint8)
    #     cv2.imwrite(os.path.join(output_dir, f"reconstructed_{i:04d}.png"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    return pred_rgb_concatenated # Return the concatenated tensor


def generate_novel_view_folder(model, data_module, angles=[[0, 0], [-30, 0], [-60, 0]], output_dir="./reconstruction", split="train"):
    """
    Generates and saves novel views for each frame in a dataset split.
    """
    print(f"Making Plots for {split} set")

    # Get the appropriate dataset based on the split
    # Note: Assuming RealDataModule now stores _train_set and _val_set
    if split == "train":
        dataset = data_module.train_dataloader().dataset  ###LCX20250711
    elif split == "val":
        dataset = data_module._val_set
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'.")

    # Ensure dataset is not None and has length > 0
    if dataset is None or len(dataset) == 0:
        print(f"[93m[WARNING]: Dataset for split '{split}' is empty. Skipping novel view generation.[0m")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Ensure the model is in evaluation mode
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for i, batch in enumerate(tqdm(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=data_module.data_worker), desc=f"Generating novel views for {split}")):
             if batch is None: # Skip if the batch is None
                print(f"[93m[WARNING]: Skipping batch {i} in novel view generation as it is None.[0m")
                continue

             # Move batch to the correct device
             for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

             frame_id = batch.get('frame_id', i) # Get frame_id or use index as fallback

             # List to hold novel views for the current frame
             frame_novel_views = []

             for angle in angles:
                 # Create a modified batch with the new camera rotation
                modified_batch = batch.copy()
                # Assuming camera_R is a rotation matrix or can be converted
                # This part might need adjustment based on how camera_R is represented
                # Example: Assuming camera_R is a 3x3 rotation matrix
                # If it's axis-angle or quaternion, conversion is needed
                if 'camera_R' in modified_batch and modified_batch['camera_R'].shape[-2:] == (3, 3):
                     # Simple rotation around Y axis for example
                     # This angle modification logic might need to match the project's camera model
                     # Convert angle degrees to tensor and move to device
                     angle_deg_tensor = torch.tensor(angle, dtype=torch.float32).to(device)
                     angle_rad_tensor = torch.deg2rad(angle_deg_tensor)

                     # Construct rotation matrices for Y and X (pitch and yaw)
                     # This assumes angle[0] is yaw (rotation around Y) and angle[1] is pitch (rotation around X)
                     # The order of multiplication matters and depends on Euler angle convention
                     # Let's use Y then X for example
                     rot_y = torch.eye(3, device=device)
                     rot_y[0, 0] = torch.cos(angle_rad_tensor[0])
                     rot_y[0, 2] = torch.sin(angle_rad_tensor[0])
                     rot_y[2, 0] = -torch.sin(angle_rad_tensor[0])
                     rot_y[2, 2] = torch.cos(angle_rad_tensor[0])

                     rot_x = torch.eye(3, device=device)
                     rot_x[1, 1] = torch.cos(angle_rad_tensor[1])
                     rot_x[1, 2] = -torch.sin(angle_rad_tensor[1])
                     rot_x[2, 1] = torch.sin(angle_rad_tensor[1])
                     rot_x[2, 2] = torch.cos(angle_rad_tensor[1])

                     # Apply the new rotation - combine original rotation with novel view rotation
                     # This assumes original camera_R is the identity or initial orientation
                     # If original camera_R is the orientation for the input image,
                     # you might need to multiply the novel view rotation with it.
                     # Example: New_R = Novel_View_Rotation @ Original_R or Original_R @ Novel_View_Rotation
                     # Let's assume Novel_View_Rotation is applied relative to the original orientation
                     # The order (rot_y @ rot_x) or (rot_x @ rot_y) depends on Euler angle convention
                     # Let's use Y then X for example
                     novel_view_rotation = torch.matmul(rot_y, rot_x)

                     # Apply the novel view rotation to the original camera rotation
                     # Assuming original camera_R is batch_size x 3 x 3
                     original_camera_R = modified_batch['camera_R'] # Shape: B x 3 x 3
                     # Ensure novel_view_rotation is broadcastable or match batch size
                     # Assuming novel_view_rotation is 3x3, broadcast it to match batch size
                     novel_view_rotation = novel_view_rotation.unsqueeze(0).expand(original_camera_R.size(0), -1, -1)

                     # Apply the new rotation (example: multiply)
                     # This multiplication order might need to be adjusted based on how camera_R is applied in the model
                     # Example: new_camera_R = novel_view_rotation @ original_camera_R
                     # Or: new_camera_R = original_camera_R @ novel_view_rotation
                     # Let's assume novel view rotation is applied before original camera_R's effect
                     modified_batch['camera_R'] = torch.matmul(novel_view_rotation, original_camera_R) # Example, adjust as needed


                else:
                    # Handle cases where camera_R is not a 3x3 matrix or not present
                    print(f"[93m[WARNING]: camera_R not in batch or not a 3x3 matrix for frame {batch.get('frame_id', 'N/A')}. Skipping angle modification for angle {angle}.[0m")
                    # If angle modification is essential, you might need to skip this angle or sample
                    if 'camera_R' not in modified_batch:
                        print(f"[93m[WARNING]: camera_R not found in batch for frame {batch.get('frame_id', 'N/A')}. Novel view generation might not work as expected.[0m")
                        # You might need to create a default camera_R or skip if camera is essential


                 # Get the predicted RGB image for this angle
                 try:
                     model_output = model(modified_batch)
                     if 'pred_rgb' in model_output:
                         # Assuming model_output['pred_rgb'] is B x C x H x W, squeeze batch dim
                         frame_novel_views.append(model_output['pred_rgb'].detach().cpu().squeeze(0))
                     else:
                         print(f"[93m[WARNING]: Model output does not contain 'pred_rgb' for frame {frame_id} at angle {angle}.[0m")
                 except Exception as e:
                     print(f"[91m[ERROR]: Error during model prediction for frame {frame_id} at angle {angle}: {e}[0m")
                     # Optionally, append a placeholder or handle skipped

             # Check if any novel views were generated for this frame
             if not frame_novel_views:
                 print(f"[91m[ERROR]: No novel views generated for frame {frame_id}. Skipping saving for this frame.[0m")
                 continue # Skip saving for this frame


             # Save the novel views for the current frame as a grid
             # Add the original image to the grid for comparison (optional)
             # Assuming original image is in batch['image'] (B x C x H x W)
             images_to_grid = []
             if 'image' in batch:
                 images_to_grid.append(batch['image'].detach().cpu().squeeze(0)) # Add original image
             images_to_grid.extend(frame_novel_views) # Add novel views

             # Save the grid
             output_filename = os.path.join(output_dir, f"frame_{frame_id:04d}_novel_views.png")
             save_image_grid(images_to_grid, output_filename, nrow=len(images_to_grid)) # Adjust nrow as needed


# You may have other functions in visualizations.py, include them here if necessary
# For example:
# def other_visualization_function(...):
#     pass
