# Protocol: Fine-tuning BraSyn on BraTS-Africa

Oct 6, 2026 · @Paul

## Overview

The pretrained BraSyn baseline model was fine-tuned on 64 BraTS-Africa (SSA) cases to synthesize one missing MRI modality from the other three. Training used the repo's own `train.py`, started from the released weights, with a 5× lower learning rate and fewer epochs than the original from-scratch training.

| Item | Value |
| --- | --- |
| Code | [WinstonHuTiger/BraSyn\_tutorial](https://github.com/WinstonHuTiger/BraSyn_tutorial), branch `main` |
| Model | 3D pix2pix GAN; generator `sit` (2.118 M parameters), PatchGAN discriminator `n_layers`, `n_layers_D 1` (0.545 M parameters) |
| Task | Input: 3 available modalities; output: the missing one (t1c, t1n, t2f or t2w, chosen at random per sample) |
| Starting weights | `mlcube/workspace/additional_files/weights/your_weight_name/latest_net_G.pth` and `latest_net_D.pth`, copied to `project/checkpoints/brasyn_pretrained/` |
| Fine-tuning data | BraTS-Africa: 64 train, 7 validation, 74 test cases |
| Hardware | Dell workstation `promaxgb10-f2c8`, NVIDIA GB10 (aarch64, unified memory), driver 580.173.02 |
| Run name | `brats_africa_ft` |

## Data preparation

Of 146 cases in the split file, 145 were used; one training case was excluded because its T2w file is empty. The data was restructured with `prepare_finetune_data.py` (written for this project, not part of the repo) into `~/software/brasyn_finetune`.

**Source data.** `/home/dell/software/BraTs Africa/BraTS-Africa Dataset/BraTS-Africa/`, classes `51_OtherNeoplasms` and `95_Glioma`. Each case holds `t1c`, `t1n`, `t2f`, `t2w` and `seg` as uncompressed `.nii`, size 240 × 240 × 155.

**Split file.** `brats_africa_split.csv` (columns `split`, `class`, `case_id`, `path`): 72 train, 74 test.

**Steps performed by the script:**

1. Read the split file (tab or comma separated).
2. Hold out 10 % of the train rows as validation, stratified by class, random seed 42.
3. Create the folder names the data loader hard-codes: `ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/` (train), `ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/` (validation) and `test/`. Despite "GLI" in the names, they hold only SSA cases.
4. Write gzip-compressed copies (`.nii.gz`, level 6) of every `.nii` file, because the repo's loader and evaluation scripts require `.nii.gz`. Originals were not modified.
5. Skip any case with a missing or empty modality file.

**Excluded case.** `BraTS-SSA-00230-000` (95\_Glioma, train split): `BraTS-SSA-00230-000-t2w.nii.gz` is 0 bytes in the source data. A first training attempt crashed on it; the case was then excluded.

| Split | Cases | Folder under `~/software/brasyn_finetune/` |
| --- | --- | --- |
| Train | 64 | `ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/` |
| Validation | 7 | `ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/` |
| Test | 74 | `test/` |

No BraTS GLI cases were mixed into training (`--mix_gli_n 0`). Open question: record the result of the full-load integrity check over all `.nii.gz` files here.

## Software environment

A Python virtual environment at `~/software/brasyn-env` was used instead of the README's conda setup. The README's PyTorch 2.1.2 + CUDA 11.8 build does not support this aarch64 machine and its Blackwell GPU.

| Package | Version | Note |
| --- | --- | --- |
| Python | 3.12 | system `python3 -m venv` |
| torch | 2.11.0+cu128 | GPU verified with a Conv3d test on the GB10 |
| torchvision | 0.26.0+cu128 |  |
| CUDA (driver) | 13.0, driver 580.173.02 |  |
| nibabel | 5.4.2 |  |
| numpy | 2.5.2 |  |
| pandas | 3.0.6 | for the prep script |
| scikit-image | 0.26.0 |  |
| scikit-learn | 1.9.1 |  |
| ray | 2.59.0 |  |
| tensorboard | 2.21.0 |  |
| typer | 0.9.0 | pinned by the repo |
| blitz-bayesian-pytorch | 0.2.8 |  |
| visdom | 0.3.0 | installed without its `openTSNE` dependency, which fails to compile on aarch64; visdom is not used (`--display_id 0`) |

Requirements were installed from `project/requirements.txt` with the `visdom` line removed. Expected harmless warnings during training: `FutureWarning` about `torch.cuda.amp.autocast` and `torch.cuda.amp.GradScaler`.

## Code modification

One line of `project/train.py` (around line 124) was changed; no other repo code was modified. With visdom off (`--display_id 0`), `Visualizer.plot_data` is never created, so the validation plot call crashed after the first validation pass. The fix applies the same guard the script already uses for the training-loss plot. Validation scores are still written to the log just before this line.

Before:

```python
visualizer.plot_current_validation_losses(epoch, val_loss)
```

After:

```python
if opt.display_id is None or opt.display_id > 0:
    visualizer.plot_current_validation_losses(epoch, val_loss)
```

`project/fine_tune.py` was not used. It implements unsupervised domain adaptation for a different data loader (`brain_3D_transfer`), which does not apply here because all BraTS-Africa cases include ground truth for every modality.

## Training

Training is set to 41 epochs (0–40). The first epoch took about 3.5 minutes (3:13 training + 0:15 validation), so the full run should take roughly 2.4 hours. It is run from `~/software/BraSyn_tutorial/project` inside a `tmux` session with the environment activated:

```bash
python train.py \
  --dataroot ~/software/brasyn_finetune \
  --name brats_africa_ft \
  --pretrained_name brasyn_pretrained --continue_train --epoch latest --epoch_count 0 \
  --model pix2pix --direction AtoB --dataset_mode brain_3D_random_mod \
  --B_modality random --input_nc 3 --output_nc 1 \
  --paired --netG sit --netD n_layers --n_layers_D 1 \
  --lr 0.00002 --n_epochs 20 --n_epochs_decay 20 \
  --batch_size 1 --gpu_ids 0 \
  --save_epoch_freq 5 --display_id 0
```

| Parameter | Fine-tuning (this run) | Original from scratch (README) | Note |
| --- | --- | --- | --- |
| Initial weights | `brasyn_pretrained` (G and D) | random | via `--continue_train --epoch latest --pretrained_name` |
| `lr` (= `glr` = `dlr`) | 0.00002 | 0.0001 | 5× lower for fine-tuning |
| `n_epochs` | 20 | 50 | epochs at constant learning rate |
| `n_epochs_decay` | 20 | 70 | epochs of linear decay to 0 |
| `epoch_count` | 0 | 1 (default) | 0 avoids loading loss logs the pretrained folder lacks |
| `lr_policy` | linear | linear | default |
| Optimizer | Adam, `beta1` 0.5 | Adam, `beta1` 0.5 | optimizer state not restored, starts fresh |
| `gan_mode` | lsgan | lsgan | default |
| `lambda_L1` | 100 | 100 | default |
| `lambda_perceptual` | 1.5 | 1.5 | default |
| `batch_size` | 1 | 1 |  |
| `model` / `netG` / `netD` | pix2pix / sit / n\_layers (`n_layers_D 1`) | same | must match the pretrained weights |
| `input_nc` / `output_nc` | 3 / 1 | 3 / 1 |  |
| `dataset_mode` / `B_modality` | brain\_3D\_random\_mod / random | same | missing modality chosen at random per sample |
| `preprocess` / `pool_size` | resize\_and\_crop / 0 | same | defaults |
| `save_epoch_freq` | 5 | not set (default 100) | snapshot every 5 epochs |
| `display_id` | 0 | visdom on | visdom disabled |
| `print_freq` / `save_latest_freq` | 10 / 5000 | defaults |  |
| `gpu_ids` | 0 | 0 |  |

On start-up the log confirmed: `The number of training images = 64` and `loading the model from ./checkpoints/brasyn_pretrained/latest_net_G.pth` (and `latest_net_D.pth`).

## Outputs

All results are written to `~/software/BraSyn_tutorial/project/checkpoints/brats_africa_ft/`. The pretrained weights in `checkpoints/brasyn_pretrained/` stay unchanged for comparison.

| File | Content |
| --- | --- |
| `latest_net_G.pth` | fine-tuned generator after the most recent epoch (overwritten each epoch) |
| `latest_net_D.pth` | matching discriminator, only needed to continue training |
| `0_net_G.pth`, `5_net_G.pth`, …, `40_net_G.pth` | generator snapshots every 5 epochs (matching `_net_D.pth` files too) |
| `loss_log.txt` | training losses, every 10 iterations |
| `val_loss_log.txt` | validation L1 and 1−SSIM per epoch |
| `train_opt.txt` | every option used in this run |
| `architecture.txt` | printed network structure |

## Caveats and evaluation plan

The learning rate and epoch counts are reasonable starting values, not tuned ones. Whether fine-tuning helped is decided only by the test-set comparison below.

- **Noisy validation.** Only 7 cases, and the loader picks a random missing modality per case on every pass. Judge the trend in `val_loss_log.txt`, not single epochs.
- **Best epoch.** `train.py` does not keep the best model. Pick the snapshot with the best validation trend, which may not be `latest`.
- **Possible forgetting.** No GLI cases were mixed in, so performance on GLI data may drop.
- **Assumption.** The released checkpoint is assumed to be trained on BraTS GLI data with the settings in `mlcube/workspace/parameters.yaml`.

**Evaluation plan:**

- [ ] Drop one modality per test case with `drop_modality.py` on `~/software/brasyn_finetune/test`, and keep the dropped modality per case fixed
- [ ] Generate the missing modality with `generate_missing_modality.py`, once with `brasyn_pretrained` and once with `brats_africa_ft`
- [ ] Compare SSIM/PSNR against the real images, and Dice from the nnU-Net segmentation step in the README
- [ ] If the gain is small, compare learning rates 1e-5, 2e-5 and 5e-5 on validation
