import os
import wandb
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.offline_metrics import evaluate_pairwise
from util import util
import time
import torch
import copy

def train():
    wandb.init()
    
    # wandb config에서 하이퍼파라미터 가져오기
    config = wandb.config
    
    # 기본 옵션 설정
    opt = TrainOptions().parse()
    
    # wandb config의 하이퍼파라미터 적용
    opt.netG = config.netG
    opt.n_layers_D = config.n_layers_D
    opt.norm = config.norm
    opt.batch_size = config.batch_size
    opt.gan_mode = config.gan_mode
    opt.lr = config.lr
    opt.beta1 = config.beta1
    opt.lambda_A = config.lambda_A
    opt.lambda_B = config.lambda_B
    opt.lambda_identity = config.lambda_identity
    
    # sweep을 위한 dataset 설정
    opt.dataset_mode = 'sweep'
    opt.phase = 'train'
    
    # 체크포인트 폴더 설정
    opt.name = "sweep_run_V"
    os.makedirs(os.path.join(opt.checkpoints_dir, opt.name), exist_ok=True)
    
    # wandb에 하이퍼파라미터 로깅
    wandb.config.update({
        "netG": opt.netG,
        "n_layers_D": opt.n_layers_D,
        "norm": opt.norm,
        "batch_size": opt.batch_size,
        "gan_mode": opt.gan_mode,
        "lr": opt.lr,
        "beta1": opt.beta1,
        "lambda_A": opt.lambda_A,
        "lambda_B": opt.lambda_B,
        "lambda_identity": opt.lambda_identity
    })
    
    # dataset 생성
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)
    
    # 모델 생성
    model = create_model(opt)
    model.setup(opt)
    
    # visualizer 생성
    visualizer = Visualizer(opt)
    
    # training 시작
    total_iters = 0
    best_val_psnr = 0.0
    best_epoch = 0
    
    # 하이퍼파라미터 정보를 포함한 체크포인트 이름 생성
    checkpoint_name = f"netG_{opt.netG}_nlayers_{opt.n_layers_D}_norm_{opt.norm}_bs_{opt.batch_size}_gan_{opt.gan_mode}_lr_{opt.lr}_beta1_{opt.beta1}_lambdaA_{opt.lambda_A}_lambdaB_{opt.lambda_B}_lambdaId_{opt.lambda_identity}"
    
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()
        model.update_learning_rate()
        
        # training
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()
            
            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
            
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
            
            iter_data_time = time.time()
        
        # validation
        val_opt = copy.deepcopy(opt)
        val_opt.phase = 'val'
        val_dataset = create_dataset(val_opt)
        real_paths = []
        fake_paths = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for vdata in val_dataset:
            model.set_input(vdata)
            model.test()
            fake = model.get_current_visuals()['fake_B']
            b_path = vdata['B_paths']
            if isinstance(b_path, list):
                b_path = b_path[0]  # Take the first path if it's a list
            img_name = os.path.basename(b_path)
            save_path = os.path.join(opt.checkpoints_dir, opt.name, f'val_fake_{img_name}')
            util.save_image(util.tensor2im(fake), save_path)
            fake_paths.append(save_path)
            real_paths.append(b_path)
        
        metrics = evaluate_pairwise(real_paths, fake_paths, device)
        visualizer.print_current_metrics(epoch, metrics)
        if opt.display_id > 0:
            visualizer.plot_current_metrics(epoch, metrics)
        
        # wandb에 metrics 로깅
        wandb.log({
            "epoch": epoch,
            **metrics,
            **model.get_current_losses()
        })
        
        # best model 저장 및 wandb sweep metric 업데이트
        if metrics['PSNR'] > best_val_psnr:
            best_val_psnr = metrics['PSNR']
            best_epoch = epoch
            # best 모델 저장 시 하이퍼파라미터 정보 포함
            model.save_networks(f'best_{checkpoint_name}')
            # wandb sweep metric 업데이트
            wandb.run.summary["best_psnr"] = best_val_psnr
            wandb.run.summary["best_epoch"] = best_epoch
            wandb.run.summary["best_checkpoint"] = f'best_{checkpoint_name}'
        
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        print('Best PSNR: %.2f at epoch %d' % (best_val_psnr, best_epoch))

if __name__ == '__main__':
    train()