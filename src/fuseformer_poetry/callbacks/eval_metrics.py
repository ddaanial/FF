from lightning.pytorch.callbacks import Callback
import sys
from pathlib import Path
from skimage import metrics
import numpy as np

parent_path = Path(__file__).parent
sys.path.append(str(parent_path))

class EvalMetrics(Callback):

    def __init__(self):
        super().__init__()
        self.total_ssim = 0.0
        self.total_psnr = 0.0
        self.total_frames = 0


    def on_validation_epoch_end(self, trainer, pl_module):
        if self.total_frames > 0:
            avg_psnr = self.total_psnr / self.total_frames
            avg_ssim = self.total_ssim / self.total_frames
        else:
            avg_psnr = 0.0
            avg_ssim = 0.0

        trainer.logger.experiment.log_metric(
            run_id=trainer.logger.run_id,
            key="epoch_val_psnr",
            value=avg_psnr,
            step=trainer.current_epoch
        )

        trainer.logger.experiment.log_metric(
            run_id=trainer.logger.run_id,
            key="epoch_val_ssim",
            value=avg_ssim,
            step=trainer.current_epoch
        )

        
        self.total_ssim = 0.0
        self.total_psnr = 0.0
        self.total_frames = 0

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):

        # batch psnr and ssim
        batch_psnr = 0.0
        batch_ssim = 0.0

        imgs, masks = batch
        b, t, c, h, w = imgs.size()
        imgs_np = (((imgs + 1) / 2)* 255).cpu().numpy().astype(np.uint8)
        imgs_np = np.transpose(imgs_np, (0, 1, 3, 4, 2))
        masks_np = masks.permute(0, 1, 3, 4, 2).cpu().numpy().astype(np.uint8) 


        inputs = imgs * (1 - masks)
        output = pl_module(inputs)
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

        # trainer.logger.experiment.log_metric(
        #     run_id=trainer.logger.run_id, 
        #     key="batch_val_psnr", 
        #     value=batch_psnr / (b * t), 
        #     step=trainer.current_epoch 
        # )

        # trainer.logger.experiment.log_metric(
        #     run_id=trainer.logger.run_id, 
        #     key="batch_val_ssim", 
        #     value=batch_ssim / (b * t), 
        #     step=trainer.current_epoch 
        # )
        self.total_ssim += batch_ssim
        self.total_psnr += batch_psnr
        self.total_frames += (b * t)


        trainer.logger.experiment.log_image(
            image=comp_imgs[0][0] / 255., 
            run_id=trainer.logger.run_id,
            artifact_file=f"val_combined_images/epoch_{trainer.current_epoch}_batch_{batch_idx}.png" 
        )


        # compute vfid
        # to be done ...

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        pass

