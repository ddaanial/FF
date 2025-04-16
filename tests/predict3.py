from pathlib import Path
import sys
import os
import json
from ultralytics.models.sam import SAM2VideoPredictor
import cv2
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import torch
from PIL import Image
sys.path.append(str(Path().resolve()))
sys.path.append(str(Path().resolve().parent))
sys.path.append(str(os.path.join(Path().resolve(), 'src')))
from fuseformer_poetry.model.fuseformer import FuseFormer


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        mode = img_group[0].mode
        if mode == '1':
            img_group = [img.convert('L') for img in img_group]
            mode = 'L'
        if mode == 'L':
            return np.stack([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif mode == 'RGB':
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.stack(img_group, axis=2)
        else:
            raise NotImplementedError(f"Image mode {mode}")


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # numpy img: [L, C, H, W]
            img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255) if self.div else img.float()
        return img
    
w, h = 432, 240
ref_length = 150  # ref_step
num_ref = -1
neighbor_stride = 3
default_fps = 24

_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()])

# read frame-wise masks
def read_mask(mpath):
    masks = []
    mnames = os.listdir(mpath)
    mnames.sort()
    for m in mnames:
        if not m.startswith('.'):
            m = Image.open(os.path.join(mpath, m))
            m = m.resize((w, h), Image.NEAREST)
            m = np.array(m.convert('L'))
            m = np.array(m > 0).astype(np.uint8)
            m = cv2.dilate(m, cv2.getStructuringElement(
                cv2.MORPH_CROSS, (3, 3)), iterations=4)
            masks.append(Image.fromarray(m*255))
    return masks


#  read frames from video
def read_frame_from_videos(vname):
    frames = []
    lst = os.listdir(vname)
    lst.sort()
    fr_lst = [vname+'/'+name for name in lst]
    for fr in fr_lst:
        if not os.path.basename(fr).startswith('.'):
            image = cv2.imread(fr)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image.resize((w, h)))
    return frames

# sample reference frames from the whole video
def get_ref_index(f, neighbor_ids, length):
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if not i in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref//2))
        end_idx = min(length, f + ref_length * (num_ref//2))
        for i in range(start_idx, end_idx+1, ref_length):
            if not i in neighbor_ids:
                if len(ref_index) > num_ref:
                    # if len(ref_index) >= 5-len(neighbor_ids):
                    break
                ref_index.append(i)
    return ref_index

if __name__ == '__main__':
    # Create the predictor
    overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="sam2_b.pt")
    predictor = SAM2VideoPredictor(overrides=overrides)

    # Run inference for segmentation
    selected_point = []
    scale_factor = 0.5  # Resize just for display, keep original for prediction

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            orig_x = int(x / scale_factor)
            orig_y = int(y / scale_factor)
            selected_point.append((orig_x, orig_y))
            print(f"🖱️ Selected point: ({orig_x}, {orig_y})")
            cv2.destroyAllWindows()

    # Extract first frame from video
    cap = cv2.VideoCapture("test_14.mp4")
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
    results = predictor(source="test_14.mp4", points=[point[0], point[1]], labels=[1])

    # Create output directories
    os.makedirs("masks", exist_ok=True)
    os.makedirs("frames", exist_ok=True)

    # Save frames and masks
    for i, result in enumerate(results):
        frame = result.orig_img
        cv2.imwrite(f"frames/frame{i:04d}.png", frame)
        if result.masks is None:
            continue
        masks_np = result.masks.data.cpu().numpy()
        for j, mask in enumerate(masks_np):
            mask_img = (mask * 255).astype(np.uint8)
            cv2.imwrite(f"masks/frame{i:04d}_mask{j}.png", mask_img)

    frames_folder = "frames"
    masks_folder = "masks"
    output_folder = "inpainted_frames"
    video_output_path = "inpainted_video.mp4"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #os.makedirs(output_folder, exist_ok=True)

    # Load model
    config_path = os.path.join(Path().resolve(), 'src', 'fuseformer_poetry','config.json')
    config = json.load(open(config_path))
    model = FuseFormer(config)
    #ckpt_path = os.path.join(os.getcwd(), 'checkpoints', 'epoch=96-step=7954-train_gen_loss=0.15-train_dis_loss=0.92-val_gen_loss=0.14-val_dis_loss=0.92-avg_psnr=35.66-avg_ssim=0.97.ckpt')
    #model = FuseFormer.load_from_checkpoint(ckpt_path)
    model.eval()
    model.to(device)

    frames = read_frame_from_videos(frames_folder)
    video_length = len(frames)
    imgs = _to_tensors(frames).unsqueeze(0)*2-1
    frames = [np.array(f).astype(np.uint8) for f in frames]

    masks = read_mask(masks_folder)
    binary_masks = [np.expand_dims(
        (np.array(m) != 0).astype(np.uint8), 2) for m in masks]
    
    masks = _to_tensors(masks).unsqueeze(0)
    imgs, masks = imgs.to(device), masks.to(device)
    comp_frames = [None]*video_length
    print('frames and masks loaded.')

    # completing holes by spatial-temporal transformers
    for f in range(0, video_length, neighbor_stride):
        neighbor_ids = [i for i in range(
            max(0, f-neighbor_stride), min(video_length, f+neighbor_stride+1))]
        ref_ids = get_ref_index(f, neighbor_ids, video_length)
        print(f, len(neighbor_ids), len(ref_ids))
        len_temp = len(neighbor_ids) + len(ref_ids)
        selected_imgs = imgs[:1, neighbor_ids+ref_ids, :, :, :]
        selected_masks = masks[:1, neighbor_ids+ref_ids, :, :, :]
        with torch.no_grad():
            masked_imgs = selected_imgs*(1-selected_masks)
            pred_img = model(masked_imgs)
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy()*255
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(
                    np.uint8)*binary_masks[idx] + frames[idx] * (1-binary_masks[idx])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(
                        np.float32)*0.5 + img.astype(np.float32)*0.5
    name = frames_folder
    writer = cv2.VideoWriter(f"{name}_result.mp4", cv2.VideoWriter_fourcc(
        *"mp4v"), default_fps, (w, h))
    for f in range(video_length):
        comp = np.array(comp_frames[f]).astype(
            np.uint8)*binary_masks[f] + frames[f] * (1-binary_masks[f])
        if w != w:
            comp = cv2.resize(comp, (w, h),
                              interpolation=cv2.INTER_LINEAR)
        writer.write(cv2.cvtColor(
            np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB))
    writer.release()
    print('Finish in {}'.format(f"{name}_result.mp4"))




