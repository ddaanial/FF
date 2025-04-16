from pathlib import Path
import sys
import os
import json
from ultralytics.models.sam import SAM2VideoPredictor
import cv2
import numpy as np
from torchvision import transforms
from tqdm import tqdm

sys.path.append(str(Path().resolve()))
sys.path.append(str(Path().resolve().parent))
sys.path.append(str(os.path.join(Path().resolve(), 'src')))
from fuseformer_poetry.model.fuseformer import FuseFormer
import torch

if __name__ == '__main__':
    # Create the predictor
    overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="sam2_b.pt")
    predictor = SAM2VideoPredictor(overrides=overrides)

    # Run inference
    #results = predictor(source="test2.mp4", points=[1350, 650], labels=[1])
    selected_point = []
    scale_factor = 0.5  # Resize just for display, keep original for prediction

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Scale coordinates back to original
            orig_x = int(x / scale_factor)
            orig_y = int(y / scale_factor)
            selected_point.append((orig_x, orig_y))
            print(f"🖱️ Selected point: ({orig_x}, {orig_y})")
            cv2.destroyAllWindows()

    # Extract first frame from video
    cap = cv2.VideoCapture("test1.mp4")
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to read video.")

    # Resize for display
    display_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
    cv2.imshow("Click on the object", display_frame)
    cv2.setMouseCallback("Click on the object", mouse_callback)
    cv2.waitKey(0)

    if not selected_point:
        raise RuntimeError("No point selected.")

    point = selected_point[0]
    results = predictor(source="test1.mp4", points=[point[0], point[1]], labels=[1])

    # Create output directories
    os.makedirs("masks", exist_ok=True)
    os.makedirs("frames", exist_ok=True)

    # Iterate through each frame's results
    for i, result in enumerate(results):
        # Save original frame
        frame = result.orig_img  # shape: (H, W, 3), dtype=uint8
        cv2.imwrite(f"frames/frame{i:04d}.png", frame)

        # Skip if no masks
        if result.masks is None:
            continue

        # Get binary masks (numpy arrays of shape [n, H, W])
        masks_np = result.masks.data.cpu().numpy()  # shape: (num_masks, H, W)

        # Save each mask
        for j, mask in enumerate(masks_np):
            mask_img = (mask * 255).astype(np.uint8)
            cv2.imwrite(f"masks/frame{i:04d}_mask{j}.png", mask_img)

    frames_folder = "frames"
    masks_folder = "masks"
    output_folder = "inpainted_frames"
    video_output_path = "inpainted_video.mp4"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_folder, exist_ok=True)


    ckpt_path = os.path.join(os.getcwd(), 'checkpoints', 'epoch=96-step=7954-train_gen_loss=0.15-train_dis_loss=0.92-val_gen_loss=0.14-val_dis_loss=0.92-avg_psnr=35.66-avg_ssim=0.97.ckpt')
    model = FuseFormer.load_from_checkpoint(ckpt_path)
    model.eval()

        # --- PREPROCESSING ---
    transform = transforms.Compose([
        transforms.ToTensor(),                 # [0, 255] -> [0, 1]
        transforms.Normalize([0.5]*3, [0.5]*3) # [0, 1] -> [-1, 1]
    ])

    # --- GET FILES ---
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith(('.png', '.jpg'))])
    mask_files = sorted([f for f in os.listdir(masks_folder) if f.endswith(('.png', '.jpg'))])

    assert len(frame_files) == len(mask_files), "Mismatch between number of frames and masks!"


    # --- INPAINTING LOOP ---
    for fname_frame, fname_mask in tqdm(zip(frame_files, mask_files), total=len(frame_files)):
        # Load image and mask
        frame = cv2.imread(os.path.join(frames_folder, fname_frame))
        mask = cv2.imread(os.path.join(masks_folder, fname_mask), cv2.IMREAD_GRAYSCALE)

        w, h = 432, 240
        frame = cv2.resize(frame, (w, h))
        mask = cv2.resize(mask, (w, h))
        mask = (mask > 127).astype(np.float32)  # Binarize

        # Convert frame from BGR (OpenCV default) to RGB (for model input)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

        # Transform
        frame_tensor = transform(frame).unsqueeze(0).to(device)  # [1, 3, H, W]
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]

        # Apply mask to frame
        masked_frame = frame_tensor * (1. - mask_tensor)

        # Inpaint
        with torch.no_grad():
            masked_frame = masked_frame.unsqueeze(1)
            pred_img = model(masked_frame)  # [1, 3, H, W]

        # Postprocess predicted image
        pred_img = pred_img[0].cpu()
        pred_img = (pred_img + 1) / 2  # [-1, 1] -> [0, 1]
        pred_img_np = (pred_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Convert predicted image from RGB to BGR for saving with OpenCV
        pred_img_np = cv2.cvtColor(pred_img_np, cv2.COLOR_RGB2BGR)

        # Save frame
        out_path = os.path.join(output_folder, fname_frame)
        cv2.imwrite(out_path, pred_img_np)


    # --- COMPILE VIDEO ---
    frame_example = cv2.imread(os.path.join(output_folder, frame_files[0]))
    height, width, _ = frame_example.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(video_output_path, fourcc, 24, (width, height))

    for fname in frame_files:
        frame = cv2.imread(os.path.join(output_folder, fname))
        out_video.write(frame)

    out_video.release()
    print(f"✅ Final inpainted video saved to: {video_output_path}")


