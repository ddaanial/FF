import re
import lightning as L
import torch
import torch.nn as nn
from pathlib import Path
import sys
from torchvision.utils import make_grid 
import mlflow
import numpy as np
import tempfile
import matplotlib.pyplot as plt 
import os
from skimage import metrics
import cv2

parent_path = Path(__file__).parent
sys.path.append(str(parent_path))

from generator import InpaintGenerator
from discriminator import Discriminator
from adversarial_loss import AdversarialLoss

class FuseFormer(L.LightningModule):
    
    def __init__(self, args):
        super().__init__()

        # save hyper parameters
        self.save_hyperparameters()

        # manual optimization
        self.automatic_optimization = False

        # networks
        self.netG = InpaintGenerator()
        weights_path = os.path.join(str(parent_path.parent), "weights/fuseformer.pth")
        state_dict = torch.load(weights_path, map_location='cpu')
        self.netG.load_state_dict(state_dict, strict=False)
        self.netD = Discriminator(
                in_channels=3, use_sigmoid=self.hparams.args['losses']['GAN_LOSS'] != 'hinge')
        
        # losses
        self.l1_loss = nn.L1Loss()
        self.adversarial_loss = AdversarialLoss(type=self.hparams.args['losses']['GAN_LOSS'])

        # optimizers
        self.optimizer_g = None
        self.optimizer_d = None

        # validation metrics
        self.total_ssim = 0.0
        self.total_psnr = 0.0
        self.total_frames = 0

    def forward(self, x):
        return self.netG(x)
    
    def _get_frames_from_batch(self, batch, batch_idx):
        """
        Processes a batch of video frames and masks to generate inpainted images.

        Args:
            batch: Tuple containing (frames, masks).
            batch_idx: Index of the current batch.

        Returns:
            concatenated_img: on_top of each other visualization of original, masked, and inpainted images.
            frames: Flattened frames tensor.
            masks: Flattened masks tensor.
            comp_img: Final composite image combining predictions with unmasked regions.
            pred_img: The inpainted image prediction.
        """
        frames, masks = batch
        b, t, c, h, w = frames.shape  # Batch size, time steps, channels, height, width

        # Extract first frame and mask, normalize to [0,1] --> for plotting
        ori_img_plot = ((frames[0, 0].permute(1, 2, 0).detach().cpu().numpy()) + 1) / 2
        ori_mask = masks[0, 0].permute(1, 2, 0).detach().cpu().numpy()
        masked_img_plot = ori_img_plot * (1 - ori_mask)


        masked_frame = frames * (1. - masks)  # Apply masks
        

        # Generate predicted inpainted image
        pred_img = self(masked_frame) 

        # Extract first inpainted frame, normalize to [0,1] --> for plotting
        pred_img_plot = pred_img[0].clone()
        pred_img_plot = (pred_img_plot + 1) / 2  # Normalize to [0,1]
        pred_img_plot = pred_img_plot.permute(1, 2, 0).detach().cpu().numpy()

        # Concatenate original, masked, and predicted images for visualization (stacked vertically)
        concatenated_img = np.concatenate((ori_img_plot, masked_img_plot, pred_img_plot), axis=0)

        # Reshape frames and masks for processing
        frames = frames.view(b * t, c, h, w)
        masks = masks.view(b * t, 1, h, w)

        # Composite image: use predicted image where mask is present
        comp_img = pred_img * masks + frames * (1. - masks)

        return concatenated_img, frames, masks, comp_img, pred_img
    
    def opt_step(self, frames, masks, comp_img, pred_img, t_v = True):

        gen_loss = 0.0
        dis_loss = 0.0
        real_vid_feat = self.netD(frames)
        fake_vid_feat = self.netD(comp_img.detach())
        dis_real_loss = self.adversarial_loss(real_vid_feat, True, True)
        dis_fake_loss = self.adversarial_loss(fake_vid_feat, False, True)
        dis_loss = (dis_real_loss + dis_fake_loss) / 2

        if t_v:
            self.optimizer_d.zero_grad()
            self.manual_backward(dis_loss)
            self.optimizer_d.step()

        gen_vid_feat = self.netD(comp_img)
        gan_loss = self.adversarial_loss(gen_vid_feat, True, False)
        gan_loss = gan_loss * self.hparams.args['losses']['adversarial_weight']
        gen_loss += gan_loss

        hole_loss = self.l1_loss(pred_img*masks, frames*masks)
        hole_loss = hole_loss / torch.mean(masks) * self.hparams.args['losses']['hole_weight']
        gen_loss += hole_loss 


        valid_loss = self.l1_loss(pred_img*(1-masks), frames*(1-masks))
        valid_loss = valid_loss / torch.mean(1-masks) * self.hparams.args['losses']['valid_weight']
        gen_loss += valid_loss 

        if t_v:
            self.optimizer_g.zero_grad()
            self.manual_backward(gen_loss)
            self.optimizer_g.step()
        
        return gen_loss, dis_loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        #print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        #print(f"Reserved Memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        self.logger.experiment.log_image(
            image=outputs['log_img'], 
            run_id=self.logger.run_id,
            artifact_file=f"train_images/epoch_{self.current_epoch}_batch_{batch_idx}.png"
        )

    def on_validation_epoch_end(self):
        if self.total_frames > 0:
            avg_psnr = self.total_psnr / self.total_frames
            avg_ssim = self.total_ssim / self.total_frames
        else:
            avg_psnr = 0.0
            avg_ssim = 0.0

        self.log("avg_psnr", avg_psnr, on_epoch=True, prog_bar=True, logger=True)
        self.log("avg_ssim", avg_ssim, on_epoch=True, prog_bar=True, logger=True)
        
        self.total_ssim = 0.0
        self.total_psnr = 0.0
        self.total_frames = 0      
        
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        # batch psnr and ssim
        batch_psnr = 0.0
        batch_ssim = 0.0

        imgs, masks = batch
        b, t, c, h, w = imgs.size()
        imgs_np = (((imgs + 1) / 2)* 255).cpu().numpy().astype(np.uint8)
        imgs_np = np.transpose(imgs_np, (0, 1, 3, 4, 2))
        masks_np = masks.permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.uint8) 


        inputs = imgs * (1 - masks)
        output = self(inputs)
        output = (output + 1) / 2
        output  = output.reshape(b, t, c, h, w)
        output = output.permute(0, 1, 3, 4, 2).cpu().numpy() * 255
        output = output.astype(np.uint8)

        comp_imgs = output * masks_np \
                        + imgs_np * (1 - masks_np)


        for batch in range(b):
            for frame in range(t):
                batch_psnr += metrics.peak_signal_noise_ratio(
                        imgs_np[batch][frame], 
                        comp_imgs[batch][frame],
                        data_range=255
                    )
                    
                batch_ssim += metrics.structural_similarity(
                        imgs_np[batch][frame], 
                        comp_imgs[batch][frame], 
                        multichannel=True,
                        data_range=255, 
                        win_size=65,
                        channel_axis = 2
                    )

        self.total_ssim += batch_ssim
        self.total_psnr += batch_psnr
        self.total_frames += (b * t)


        self.logger.experiment.log_image(
            image=comp_imgs[0][0] / 255., 
            run_id=self.logger.run_id,
            artifact_file=f"val_combined_images/epoch_{self.current_epoch}_batch_{batch_idx}.png" 
        )

        self.logger.experiment.log_image(
            image=outputs['log_img'], 
            run_id=self.logger.run_id,
            artifact_file=f"val_images/epoch_{self.current_epoch}_batch_{batch_idx}.png"
        )
    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx = 0):

        frames, masks = batch
        b, t, c, h, w = frames.shape 
        ori_img_plot = ((frames[0].permute(0, 2, 3, 1).detach().cpu().numpy()) + 1) / 2
        ori_mask = masks[0].permute(0, 2, 3, 1).detach().cpu().numpy()
        masked_img_plot = ori_img_plot * (1 - ori_mask)


        masked_frame = frames * (1. - masks) 
        

        
        pred_img = self(masked_frame) 

        
        pred_img_plot = pred_img.clone()
        pred_img_plot = (pred_img_plot + 1) / 2  

        pred_img_plot = pred_img_plot.permute(0, 2, 3, 1).detach().cpu().numpy()
        writer = cv2.VideoWriter(f'video_{self.current_epoch}_{batch_idx}.mp4', cv2.VideoWriter_fourcc(*'mp4v'),self.hparams.args['trainer']["fps_save_test_videos"], (w, h))

        for frame in pred_img_plot:
            frame = (frame * 255).astype(np.uint8) 
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
            writer.write(frame)


        writer.release()
        print(f"Video saved")
    def training_step(self, batch, batch_idx):

        self.optimizer_g, self.optimizer_d = self.optimizers()
        log_img, frames, masks, comp_img, pred_img = self._get_frames_from_batch(batch, batch_idx)

        gen_loss, dis_loss = self.opt_step(frames, masks, comp_img, pred_img, True)

        self.log("train_gen_loss", gen_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_dis_loss", dis_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {"train_gen_loss": gen_loss, "train_dis_loss": dis_loss, "log_img": log_img}
    
    def validation_step(self, batch, batch_idx):

        log_img, frames, masks, comp_img, pred_img = self._get_frames_from_batch(batch, batch_idx)
        gen_loss, dis_loss = self.opt_step(frames, masks, comp_img, pred_img, False)

        self.log("val_gen_loss", gen_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_dis_loss", dis_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"val_gen_loss": gen_loss, "val_dis_loss": dis_loss, "log_img": log_img}
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


    def configure_optimizers(self):

        opt_g = torch.optim.Adam(
            self.netG.parameters(), 
            lr=self.hparams.args['trainer']['lr'],
            betas=(self.hparams.args['trainer']['beta1'], self.hparams.args['trainer']['beta2']))
        opt_d = torch.optim.Adam(
                self.netD.parameters(), 
                lr=self.hparams.args['trainer']['lr'],
                betas=(self.hparams.args['trainer']['beta1'], self.hparams.args['trainer']['beta2']))

        return [opt_g, opt_d], []
