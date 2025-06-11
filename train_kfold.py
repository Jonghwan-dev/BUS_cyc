import copy
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.offline_metrics import evaluate_pairwise
from util import util
import torch
import os


def train_one_fold(opt, fold):
    opt.current_fold = fold
    opt.phase = 'train'
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f'Fold {fold}: training images = {dataset_size}')

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0

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
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:
                print(f'saving the latest model (epoch {epoch}, total_iters {total_iters})')
                save_suffix = f'iter_{total_iters}' if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        # run validation
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
            save_path = os.path.join(opt.checkpoints_dir, opt.name, f'fold{fold}_val_fake_{img_name}')
            util.save_image(util.tensor2im(fake), save_path)
            fake_paths.append(save_path)
            real_paths.append(b_path)
        metrics = evaluate_pairwise(real_paths, fake_paths, device)
        visualizer.print_current_metrics(epoch, metrics)
        if opt.display_id > 0:
            visualizer.plot_current_metrics(epoch, metrics)

        if epoch % opt.save_epoch_freq == 0:
            print(f'saving the model at the end of epoch {epoch}, iters {total_iters}')
            model.save_networks('latest')
            model.save_networks(epoch)
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

    # evaluation on test set
    test_opt = copy.deepcopy(opt)
    test_opt.phase = 'test'
    test_dataset = create_dataset(test_opt)
    real_paths = []
    fake_paths = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for data in test_dataset:
        model.set_input(data)
        model.test()
        fake = model.get_current_visuals()['fake_B']
        b_path = data['B_paths']
        if isinstance(b_path, list):
            b_path = b_path[0]  # Take the first path if it's a list
        img_name = os.path.basename(b_path)
        save_path = os.path.join(opt.checkpoints_dir, opt.name, f'fold{fold}_fake_{img_name}')
        util.save_image(util.tensor2im(fake), save_path)
        fake_paths.append(save_path)
        real_paths.append(b_path)
    metrics = evaluate_pairwise(real_paths, fake_paths, device)
    return metrics


def main():
    opt = TrainOptions().parse()
    opt.dataset_mode = 'kfold'
    all_metrics = []
    for fold in range(opt.k_folds):
        m = train_one_fold(opt, fold)
        print(f'Fold {fold} metrics: {m}')
        all_metrics.append(m)
    # average metrics
    keys = all_metrics[0].keys()
    avg = {k: sum(d[k] for d in all_metrics) / len(all_metrics) for k in keys}
    print('Average metrics:', avg)


if __name__ == '__main__':
    main()
