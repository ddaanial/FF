from pathlib import Path
import sys
import os
import json
sys.path.append(str(Path().resolve()))
sys.path.append(str(Path().resolve().parent))
sys.path.append(str(os.path.join(Path().resolve(), 'src')))
from fuseformer_poetry.model.fuseformer import FuseFormer
from fuseformer_poetry.callbacks.eval_metrics import EvalMetrics
from fuseformer_poetry.data.data_module import CusDataModule
from lightning.pytorch.loggers import MLFlowLogger
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.tuner import Tuner
import torch

if __name__ == '__main__':
    config_path = os.path.join(Path().resolve(), 'src', 'fuseformer_poetry','config.json')
    config = json.load(open(config_path))

    dmodule = CusDataModule(config)
    #model = FuseFormer(config)
    ckpt_path = os.path.join(os.getcwd(), 'checkpoints', 'epoch=96-step=7954-train_gen_loss=0.15-train_dis_loss=0.92-val_gen_loss=0.14-val_dis_loss=0.92-avg_psnr=35.66-avg_ssim=0.97.ckpt')
    model = FuseFormer.load_from_checkpoint(ckpt_path)
    dmodule.setup(stage='fit')
    dmodule.setup(stage='test')

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="{epoch}-{step}-{train_gen_loss:.2f}-{train_dis_loss:.2f}-{val_gen_loss:.2f}-{val_dis_loss:.2f}-{avg_psnr:.2f}-{avg_ssim:.2f}",
        monitor="avg_psnr",
        mode="max",
        save_last=True
    )
    es_callback = EarlyStopping(monitor="avg_psnr", mode="max")
    eval_callback = EvalMetrics()

    mlflow_logger = MLFlowLogger(experiment_name= "fuseformer",run_name=f"test_2_ne_{config['trainer']['num_epoch']}+100_small_masks", tracking_uri="http://127.0.0.1:5050")

    trainer = L.Trainer(callbacks=[checkpoint_callback], 
                        accelerator='gpu', 
                        devices=1,
                        logger=mlflow_logger,
                        precision= config['trainer']['precision'], 
                        max_epochs=config['trainer']['num_epoch'], 
                        log_every_n_steps= config['trainer']['log_every_n_steps'])


    trainer.fit(model, dmodule)
    trainer.test(model, dmodule)
