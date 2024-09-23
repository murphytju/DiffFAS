import os
import warnings

warnings.filterwarnings("ignore")
import random
import time, cv2, torch
import torch.distributed as dist
from config.diffconfig import DiffusionConfig, get_model_conf
from tensorfn import load_config as DiffConfig
from diffusion import create_gaussian_diffusion, make_beta_schedule, ddim_steps
from tensorfn.optim import lr_scheduler
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from FAS_dataset import *
import torchvision
from custom_rn import resnet18

def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)        
        
def train(conf, loader, val_loader, model, ema, diffusion, betas, optimizer, scheduler, guidance_prob, cond_scale, device, means_size, var_size):

    iters = 0
    loss_list = []
    loss_mean_list = []
    loss_vb_list = []
    encoder = model.encoder(args.pretrain_classifier)
    encoder.eval()

    for epoch in range(args.max_epochs):

        print ('#Epoch - '+str(epoch+1))
        progress_bar = tqdm(loader, desc="Training")
        for batch in progress_bar:
            iters = iters + 1
            img = batch['content']
            target_img = batch['GT']
            target_pose = batch['style_spoof']
            img = img.to(device)
            target_img = target_img.to(device)
            target_pose = target_pose.to(device)
            
            time_t = torch.randint(
                0,
                conf.diffusion.beta_schedule["n_timestep"],
                (img.shape[0],),
                device=device,
            )
            
            loss_dict = diffusion.training_losses(
                model,
                encoder,
                x_start=target_img,
                t=time_t,
                betas=betas.cuda(),
                cond_input=[img, target_pose],
                prob=1 - args.guidance_prob,
                means_size = args.means_size,
                var_size = args.var_size,
                use_pair = args.use_pair
            )
            
            loss = loss_dict['loss'].mean()
            loss_mse = loss_dict['mse'].mean()
            loss_vb = loss_dict['vb'].mean()

            optimizer.zero_grad()
            loss.backward()
            scheduler.step()
            optimizer.step()

            loss_list.append(loss.detach().item())
            loss_mean_list.append(loss_mse.detach().item())
            loss_vb_list.append(loss_vb.detach().item())

            accumulate(ema, model, 0 if iters < conf.training.scheduler.warmup else 0.9999)

            
            if iters % args.print_loss_every_iters == 0:
                avg_loss = sum(loss_list) / len(loss_list)
                avg_loss_vb = sum(loss_vb_list) / len(loss_vb_list)
                avg_loss_mean = sum(loss_mean_list) / len(loss_mean_list)
                
                
                progress_bar.set_description(
                    f"Loss: {avg_loss:.4f}, Loss_vb: {avg_loss_vb:.4f}, Loss_mean: {avg_loss_mean:.4f}, Epoch: {epoch+1}, Steps: {iters}"
                )
                
            loss_list = []
            loss_mean_list = []
            loss_vb_list = []



            if iters%args.save_checkpoints_every_iters == 0:
                model_module = model

                torch.save(
                    {
                        "model": model_module.state_dict(),
                        "ema": ema.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "conf": conf,
                    },
                    conf.training.ckpt_path + f"/model_{str(iters).zfill(6)}.pt"
                )


            if (iters)%args.save_images_every_iters==0:

                print ('Generating samples at iters number ' + str(iters))                
                val_loader_iter = iter(val_loader)
                val_batch = next(val_loader_iter)
                val_img = val_batch['content'].cuda()
                val_pose = val_batch['style_spoof'].cuda()
                num_samples = val_pose.size(0)
                shuffle_idx = torch.randperm(num_samples)
                val_pose_shuffled = val_pose[shuffle_idx]
                val_GT = val_batch['GT'].cuda()
                with torch.no_grad():
                    if args.sample_algorithm == 'ddpm':
                        print('Sampling algorithm used: DDPM')
                        samples = diffusion.p_sample_loop(
                            model, 
                            encoder, 
                            x_cond=[val_pose_shuffled, val_img], 
                            progress=True, 
                            cond_scale=args.cond_scale, 
                            sample_initial_noise=args.sample_initial_noise, 
                            means_size=args.means_size, 
                            var_size=args.var_size,
                            use_pair=args.use_pair
                        )
                    elif args.sample_algorithm == 'ddim':
                        print('Sampling algorithm used: DDIM')
                        ddim_skip = args.DDIM_skip
                        
                        seq = range(0, args.sample_initial_noise, ddim_skip)
                        xs, x0_preds = ddim_steps(
                            encoder,
                            seq, 
                            model, 
                            betas.cuda(), 
                            [val_pose_shuffled, val_img],
                            diffusion = diffusion,
                            cond_scale=args.cond_scale, 
                            sample_initial_noise=args.sample_initial_noise, 
                            means_size=args.means_size, 
                            var_size=args.var_size,
                            use_pair=args.use_pair)
                        samples = xs.cuda()
                grid = torch.cat([val_img, val_pose_shuffled[:, :3], samples], -1)
                grid = grid * 0.5 + 0.5
                img_name = f"{args.save_img_path}{iters}_output.png"
                torchvision.utils.save_image(grid, img_name, nrow=grid.shape[0] // 2)

              

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
 
 
def main(settings, EXP_NAME):
    [args, DiffConf] = settings
    seed_torch(seed=1)

    root_dir = args.data_dir
    json_file = args.data_path
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(), 
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])
    if args.protocol == "PADISI":
        dataset = PADISIDataset(json_file, root_dir, transform=transform)
    elif args.protocol == "OCIM":
        dataset = OCIMDataset(json_file, root_dir, transform=transform)
    elif args.protocol == "WMCA":
        dataset = WMCADataset(json_file, root_dir, transform=transform)
    elif args.protocol == "More_datasets":
        '''
        Need to be modified for flexible combination.
        '''
        dataset_padisi = PADISIDataset(json_file = "./datasets/PADISI/PADISI.json", root_dir = "./datasets/PADISI", transform=transform)
        dataset_ocim = OCIMDataset(json_file = "./datasets/OCIM/OCIM.json", root_dir = "./datasets/OCIM", transform=transform)
        dataset = data.ConcatDataset([dataset_padisi,dataset_ocim])
    else:
        raise ValueError(f"Unsupported protocol '{args.protocol}'. Please choose from 'PADISI', 'OCIM', 'WMCA', or 'More_datasets'.")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    if args.use_pair == True:
        conf = get_model_conf()
        conf.in_channels = 3+3
        model = conf.make_model()
        model = model.to(args.device)
        conf.in_channels = 3+3
        ema = conf.make_model()
        ema = ema.to(args.device)
    else:
        model = get_model_conf().make_model()
        model = model.to(args.device)
        ema = get_model_conf().make_model()
        ema = ema.to(args.device)
    

    optimizer = DiffConf.training.optimizer.make(model.parameters())
    scheduler = DiffConf.training.scheduler.make(optimizer)

    
    if args.pretrain_path is not None:
        ckpt = torch.load(args.pretrain_path)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        scheduler.load_state_dict(ckpt["scheduler"])
        optimizer.load_state_dict(ckpt["optimizer"])
        print ('model loaded successfully')

    betas = DiffConf.diffusion.beta_schedule.make()
    diffusion = create_gaussian_diffusion(betas, predict_xstart = False)
    train(
        DiffConf, dataloader, dataloader, model, ema, diffusion, betas, optimizer, scheduler, args.guidance_prob, args.cond_scale, args.device, args.means_size, args.var_size
    )        
        
        
       
if __name__ == "__main__":


    import argparse
    parser = argparse.ArgumentParser(description='help')
    parser.add_argument('--exp_name', type=str, default='FAS_Styletransfer')
    parser.add_argument('--DiffConfigPath', type=str, default='./config/diffusion.conf')
    parser.add_argument('--data_dir', type=str, default="./datasets/PADISI/")
    parser.add_argument('--data_path', type=str, default="./datasets/PADISI/PADISI.json")
    parser.add_argument('--save_path', type=str, default='checkpoints')
    parser.add_argument('--save_img_path', type=str, default="./checkpoints/FAS_Styletransfer/sample/")
    parser.add_argument('--protocol', type=str, default='PADISI')
    parser.add_argument('--pretrain_classifier', type=str, default="./PADISI.pkl")
    parser.add_argument('--cond_scale', type=int, default=2)
    parser.add_argument('--guidance_prob', type=int, default=0.2)
    parser.add_argument('--DDIM_skip', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=400)
    parser.add_argument('--means_size', type=str, default=5)
    parser.add_argument('--var_size', type=str, default=3)
    parser.add_argument('--random_seed', type=str, default=2024)
    parser.add_argument('--pretrain_path', type=str, default=None)
    parser.add_argument('--use_pair', type=bool, default=True)
    parser.add_argument('--sample_algorithm', type=str, default='ddpm')
    parser.add_argument('--sample_initial_noise', type=str, default=250)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--save_checkpoints_every_iters', type=int, default=10000)
    parser.add_argument('--print_loss_every_iters', type=int, default=1)
    parser.add_argument('--save_images_every_iters', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--n_machine', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    print ('Experiment: '+ args.exp_name)
    DiffConf = DiffConfig(DiffusionConfig,  args.DiffConfigPath, args.opts, False)
    DiffConf.training.ckpt_path = os.path.join(args.save_path, args.exp_name)
    
    if not os.path.isdir(args.save_path): os.mkdir(args.save_path)
    if not os.path.isdir(DiffConf.training.ckpt_path): os.mkdir(DiffConf.training.ckpt_path)
    if not os.path.isdir(args.save_img_path): os.mkdir(args.save_img_path)
    main(settings = [args, DiffConf], EXP_NAME = args.exp_name)