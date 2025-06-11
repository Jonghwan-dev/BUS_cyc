import copy
import time
import wandb
import numpy as np

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util import util
from util import offline_metrics


def compute_val_psnr(model, opt):
    val_dataset = create_dataset(opt)
    psnr_sum = 0.0
    count = 0
    for data in val_dataset:
        model.set_input(data)
        model.test()
        fake = model.get_current_visuals()["fake_B"]
        real = data["B"]
        fake_np = util.tensor2im(fake).astype(np.float32) / 255.0
        real_np = util.tensor2im(real).astype(np.float32) / 255.0
        psnr_sum += offline_metrics.compute_psnr(real_np, fake_np)
        count += 1
    return psnr_sum / max(count, 1)


def train():
    opt = TrainOptions().parse()
    wandb.init(project=opt.wandb_project_name, config=vars(opt))
    config = wandb.config
    for key in [
        "netG",
        "n_layers_D",
        "norm",
        "batch_size",
        "gan_mode",
        "lr",
    ]:
        if hasattr(config, key):
            setattr(opt, key, getattr(config, key))
    opt.use_wandb = True

    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print("The number of training images = %d" % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0

    val_opt = copy.deepcopy(opt)
    val_opt.phase = "val"

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()
        model.update_learning_rate()
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
                visualizer.display_current_results(
                    model.get_current_visuals(), epoch, save_result
                )

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(
                    epoch, epoch_iter, losses, t_comp, t_data
                )
                if opt.display_id > 0:
                    visualizer.plot_current_losses(
                        epoch, float(epoch_iter) / dataset_size, losses
                    )

            if total_iters % opt.save_latest_freq == 0:
                print(
                    "saving the latest model (epoch %d, total_iters %d)"
                    % (epoch, total_iters)
                )
                save_suffix = (
                    "iter_%d" % total_iters if opt.save_by_iter else "latest"
                )
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        val_psnr = compute_val_psnr(model, val_opt)
        wandb.log({"val_psnr": val_psnr, "epoch": epoch})

        if epoch % opt.save_epoch_freq == 0:
            print(
                "saving the model at the end of epoch %d, iters %d"
                % (epoch, total_iters)
            )
            model.save_networks("latest")
            model.save_networks(epoch)

        print(
            "End of epoch %d / %d \t Time Taken: %d sec"
            % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time)
        )

    wandb.finish()


if __name__ == "__main__":
    train()
