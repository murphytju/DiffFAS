import warnings
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image
from tensorfn import load_config as DiffConfig
import numpy as np
from config.diffconfig import DiffusionConfig, get_model_conf
import torch.distributed as dist
import os
import cv2
import time
from diffusion import create_gaussian_diffusion, ddim_steps
import torchvision.transforms as transforms
import argparse
from FAS_dataset import *
import torchvision


def seed_torch(seed=1029):
    """Set seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args):
    """Main function for sampling."""
    seed_torch(args.random_seed)

    conf = DiffConfig(DiffusionConfig, args.DiffConfigPath, show=False)
    betas = conf.diffusion.beta_schedule.make()
    diffusion = create_gaussian_diffusion(betas, predict_xstart=False)
    if args.use_pair == True:
        conf = get_model_conf()
        conf.in_channels = 3+3
        model = conf.make_model()
    else:    
        model = get_model_conf().make_model()

    model.load_state_dict(torch.load(args.model_path)["model"],strict=False)
    model = model.cuda().eval()
    encoder = model.encoder(args.pretrain_classifier)
    encoder.eval()

    # torch.save(
    #                 {
    #                     "model": model.state_dict(),
    #                     "conf": conf,
    #                 },
    #                 "Model_WMCA_V1.pt"
    #             )

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    val_img = Image.open(args.live_face_path)
    val_spoof = Image.open(args.spoof_face_path)
    val_img = transform(val_img).cuda().unsqueeze(0)
    val_spoof = transform(val_spoof).cuda().unsqueeze(0)

    with torch.no_grad():
        if args.sample_algorithm == 'ddpm':
            print('Sampling algorithm used: DDPM')
            samples = diffusion.p_sample_loop(
                model,
                encoder,
                x_cond=[val_spoof, val_img],
                progress=True,
                cond_scale=args.cond_scale,
                sample_initial_noise=args.sample_initial_noise,
                means_size=args.means_size,
                var_size=args.var_size,
                use_pair=args.use_pair,
                noise=None
            )
        elif args.sample_algorithm == 'ddim':
            print('Sampling algorithm used: DDIM')
            ddim_skip = args.DDIM_skip
            seq = range(0, args.sample_initial_noise, ddim_skip)
            samples, _ = ddim_steps(
                encoder,
                seq,
                model,
                betas.cuda(),
                [val_spoof, val_img],
                diffusion=diffusion,
                cond_scale=args.cond_scale,
                sample_initial_noise=args.sample_initial_noise,
                means_size=args.means_size,
                var_size=args.var_size,
                use_pair=args.use_pair
            )
            samples = samples.cuda()

    # Save images
    grid = torch.cat([val_img, val_spoof, samples], -1)
    grid = grid * 0.5 + 0.5
    save_image(grid, os.path.join(args.output_dir, "output.png"), nrow=grid.shape[0] // 2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPM/DDIM Sampling")
    parser.add_argument('--DiffConfigPath', type=str, default='./config/diffusion.conf', help='Path to diffusion config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to pretrained model')
    parser.add_argument('--pretrain_classifier', type=str, required=True, help='Path to pretrained Classifier Model')
    parser.add_argument('--live_face_path', type=str, required=True, help='Path to input validation live face')
    parser.add_argument('--spoof_face_path', type=str, required=True, help='Path to input validation spoof face')
    parser.add_argument('--sample_algorithm', type=str, default='ddim', choices=['ddpm', 'ddim'], help='Sampling algorithm: ddpm or ddim')
    parser.add_argument('--cond_scale', type=float, default=2.0, help='Condition scaling factor for sampling')
    parser.add_argument('--random_seed', type=int, default=1029, help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save output images')
    parser.add_argument('--DDIM_skip', type=int, default=10)
    parser.add_argument('--sample_initial_noise', type=int, default=250, help='Initial noise level for sampling')
    parser.add_argument('--means_size', type=int, default=5, help='Means size for conditioning')
    parser.add_argument('--var_size', type=int, default=3, help='Variance size for conditioning')
    parser.add_argument('--use_pair', type=bool, default=False, help='Whether to use paired conditioning')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)

