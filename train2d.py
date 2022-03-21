
#python3 train2d.py --config ./config/config_astro_acorn_64.ini

# Enable import from parent package
import os
import re
import cv2
import sys
import utils
import torch
import dataio
import skimage
import modules
import training
import warnings
import numpy as np
import configargparse
import loss_functions
import pruning_functions
import matplotlib.pyplot as plt

from tqdm import tqdm
from time import time
from functools import partial
from torch.utils.data import DataLoader

#python3 train_trail.py --config ./config/config_pluto_acorn_1k.in

dim = '2d_5'
data_dir = '../../data'
orig_img_dir = os.path.join(data_dir, 'pdr3_output', dim, 'orig_imgs')
output_dir = os.path.join(data_dir, 'pdr3_output/'+dim+'/ACORN')

p = configargparse.ArgumentParser()
p.add('-c', '--config', required=False, is_config_file=True, help='Path to config file.')

# General training options
p.add_argument('--lr', type=float, default=1e-3, help='learning rate. default=1e-3')
p.add_argument('--num_iters', type=int, default=1000,
               help='Number of iterations to train for.')
p.add_argument('--num_workers', type=int, default=1,
               help='number of dataloader workers.')
p.add_argument('--skip_logging', action='store_true', default=True,
               help="don't use summary function, only save loss and models")
p.add_argument('--eval', action='store_true', default=False,
               help='run evaluation')
p.add_argument('--resume', action='store_true', default=False,
               help='resume training or not.')
p.add_argument('--gpu', type=int, default=0,
               help='GPU ID to use')

# logging options
p.add_argument('--experiment_name', type=str, default='trail', required=False,
               help='path to directory where checkpoints & tensorboard events will be saved.')
p.add_argument('--epochs_til_ckpt', type=int, default=2,
               help='Epochs until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=500,
               help='Number of iterations until tensorboard summary is saved.')
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--logging_root', type=str, default=output_dir, help='root for logging')

# dataset options
p.add_argument('--res', nargs='+', type=int, default=[512],
               help='image resolution.')
p.add_argument('--dataset', type=str, default='camera', choices=['camera', 'pluto', 'tokyo', 'mars'],
               help='which dataset to use')
p.add_argument('--nfilters', type=int, default=5,
               help='number of filters')

# model options
p.add_argument('--patch_size', nargs='+', type=int, default=[32],
               help='patch size.')
p.add_argument('--hidden_features', type=int, default=512,
               help='hidden features in network')
p.add_argument('--hidden_layers', type=int, default=4,
               help='hidden layers in network')
p.add_argument('--w0', type=int, default=5,
               help='w0 for the siren model.')
p.add_argument('--steps_til_tiling', type=int, default=500,
               help='How often to recompute the tiling, also defines number of steps per epoch.')
p.add_argument('--max_patches', type=int, default=1024,
               help='maximum number of patches in the optimization')
p.add_argument('--model_type', type=str, default='multiscale', required=False,
               choices=['multiscale', 'siren', 'pe'],
               help='Type of model to evaluate, default is multiscale.')
p.add_argument('--scale_init', type=int, default=3,
               help='which scale to initialize active patches in the quadtree')

opt = p.parse_args()

opt.logging_root += '/' + str(opt.res[0])

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

for k, v in opt.__dict__.items():
    print(k, v)


def main():
    # load data
    coord_dataset = load_data()
    header = utils.get_header(data_dir, opt.res[0])

    image_resolution = (opt.res, opt.res)
    opt.num_epochs = opt.num_iters // coord_dataset.__len__()
    dataloader = DataLoader(coord_dataset, shuffle=False, batch_size=1,
                            pin_memory=True, num_workers=opt.num_workers)
    # define loss
    loss_fn = partial(loss_functions.image_mse,
                      tiling_every=opt.steps_til_tiling,
                      dataset=coord_dataset,
                      model_type=opt.model_type)

    # define pruning function
    pruning_fn = partial(pruning_functions.no_pruning,
                         pruning_every=1)

    # init model and optimizer
    out_features = opt.nfilters
    model = init_model(out_features)
    optim = torch.optim.Adam(lr=opt.lr, params=model.parameters())

    model_file_id, start_epoch, total_steps = 0, 0, 0
    if opt.resume:
        model_file_id, start_epoch, total_steps, \
            model, optimizer, coord_dataset = load_net(model, optim, coord_dataset)

    # train or recon
    if opt.eval:
        run_eval(model, coord_dataset, header)
    else:
        objs_to_save = {'quadtree': coord_dataset.quadtree}

        root_path = os.path.join(opt.logging_root, opt.experiment_name)
        p.write_config_file(opt, [os.path.join(root_path, 'config.ini')])
        checkpoint_dir = os.path.join(opt.logging_root, opt.experiment_name, 'models')

        training.train(model, optim, dataloader, start_epoch, opt.num_epochs, total_steps,
                       model_file_id, opt.lr, opt.epochs_til_ckpt, checkpoint_dir, loss_fn,
                       pruning_fn, objs_to_save=objs_to_save)

def run_eval(model, coord_dataset, header):
    recon_dir = os.path.join(opt.logging_root, opt.experiment_name, 'recons')
    metrics_dir = os.path.join(opt.logging_root, opt.experiment_name, 'metrics')
    checkpoint_dir = os.path.join(opt.logging_root, opt.experiment_name, 'models')

    ckpt_fns = sorted([f for f in os.listdir(checkpoint_dir)])
    ckpt_fns = [os.path.join(checkpoint_dir, f) for f in ckpt_fns]

    psnrs, ssims, mses = [], [], []

    for id, ckpt_fn in enumerate(ckpt_fns):
        ckpt = torch.load(ckpt_fn)

        model.load_state_dict(ckpt['model_state_dict'])
        coord_dataset.quadtree.__load__(ckpt['quadtree'])
        coord_dataset.synchronize()

        # save image and calculate psnr
        coord_dataset.toggle_eval()
        model_input, gt = coord_dataset[0]
        coord_dataset.toggle_eval()

        # convert to cuda and add batch dimension
        tmp = {}
        for key, value in model_input.items():
            if isinstance(value, torch.Tensor):
                tmp.update({key: value[None, ...].cpu()})
            else:
                tmp.update({key: value})
        model_input = tmp

        tmp = {}
        for key, value in gt.items():
            if isinstance(value, torch.Tensor):
                tmp.update({key: value[None, ...].cpu()})
            else:
                tmp.update({key: value})
        gt = tmp

        mse, psnr, ssim = utils.reconstruct(id, coord_dataset, gt, model_input,
                                            model, recon_dir, header)
        mses.append(mse);psnrs.append(psnr);ssims.append(ssim)

    np.save(os.path.join(metrics_dir, 'mse.npy'), np.array(mses))
    np.save(os.path.join(metrics_dir, 'psnr.npy'), np.array(psnrs))
    np.save(os.path.join(metrics_dir, 'ssim.npy'), np.array(ssims))


def load_data():
    '''
    if opt.dataset == 'camera':
        img_dataset = dataio.Camera()
    elif opt.dataset == 'pluto':
        pluto_url = "https://upload.wikimedia.org/wikipedia/commons/e/ef/Pluto_in_True_Color_-_High-Res.jpg"
        img_dataset = dataio.ImageFile(os.path.join(orig_img_dir, 'pluto.jpg'),
                                       url=pluto_url, grayscale=False) #opt.grayscale)
    elif opt.dataset == 'tokyo':
        img_dataset = dataio.ImageFile('../data/tokyo.tif', grayscale=opt.grayscale)
    elif opt.dataset == 'mars':
        img_dataset = dataio.ImageFile('../data/mars.tif', grayscale=opt.grayscale)
    '''

    img_sz = opt.res[0]
    img_dataset = dataio.AstroImageFile(os.path.join(orig_img_dir, '0_'+str(img_sz)+'.npy'))

    if len(opt.patch_size) == 1:
        opt.patch_size = opt.nfilters * opt.patch_size

    # set up dataset
    coord_dataset = dataio.Patch2DWrapperMultiscaleAdaptive\
        (img_dataset, sidelength=opt.res, patch_size=opt.patch_size[1:],
         jitter=True, num_workers=opt.num_workers, length=opt.steps_til_tiling,
         scale_init=opt.scale_init, max_patches=opt.max_patches)

    return coord_dataset


def init_model(out_features):
    if opt.model_type == 'multiscale':
        model = modules.ImplicitAdaptivePatchNet\
            (in_features=3, out_features=out_features,
             num_hidden_layers=opt.hidden_layers,
             hidden_features=opt.hidden_features,
             feature_grid_size=(opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]),
             sidelength=opt.res, num_encoding_functions=10,
             patch_size=opt.patch_size[1:])

    elif opt.model_type == 'siren':
        model = modules.ImplicitNet\
            (opt.res, in_features=2, out_features=out_features,
             num_hidden_layers=4, hidden_features=1536,
             mode='siren', w0=opt.w0)

    elif opt.model_type == 'pe':
        model = modules.ImplicitNet\
            (opt.res, in_features=2, out_features=out_features,
             num_hidden_layers=4, hidden_features=1536, mode='pe')
    else:
        raise NotImplementedError('Only model types multiscale, siren, and pe are implemented')

    if torch.cuda.is_available():
        model.cuda()
    return model


def load_net(model, optim, coord_dataset):
    print('Loading checkpoints')
    try:
        model_dir = os.path.join(opt.logging_root, opt.experiment_name, 'models')
        assert(os.path.isdir(model_dir))
        assert opt.config is not None, 'Specify config file'

        nmodels = len(os.listdir(model_dir))
        if nmodels < 1:
            raise ValueError("No saved models to load")

        model_fn = os.path.join(model_dir, f'model{nmodels-1:06d}.pth')
        checkpoint = torch.load(model_fn)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.train()

        if cuda:
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in optim.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
        else:
            optim.load_state_dict(checkpoint['optimizer_state_dict'])

        coord_dataset.quadtree.__load__(checkpoint['quadtree'])
        coord_dataset.synchronize()

        model_file_id = nmodels
        start_epoch = checkpoint['epoch']
        total_steps = checkpoint['total_steps']

        return model_file_id, start_epoch, total_steps, \
            model, optim, coord_dataset

    except FileNotFoundError:
        print('Loading failed, start training from begining')
        return 0, 0, 0, model, optim, coord_dataset

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()
