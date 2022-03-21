import os
import torch
import numpy as np
import skimage.measure
import matplotlib.pyplot as plt

from tqdm import tqdm
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from torchvision.utils import make_grid


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_psnr(pred_img, gt_img, writer, iter, prefix):
    batch_size = pred_img.shape[0]

    pred_img = pred_img.detach().cpu().numpy()
    gt_img = gt_img.detach().cpu().numpy()

    psnrs = list()
    for i in range(batch_size):
        p = pred_img[i].transpose(1, 2, 0)
        trgt = gt_img[i].transpose(1, 2, 0)

        p = (p / 2.) + 0.5
        p = np.clip(p, a_min=0., a_max=1.)
        trgt = (trgt / 2.) + 0.5

        psnr = skimage.metrics.peak_signal_noise_ratio(p, trgt, data_range=1)
        psnrs.append(psnr)

    writer.add_scalar(prefix + "psnr", np.mean(psnrs), iter)


def write_image_patch_multiscale_summary(image_resolution, patch_size, dataset, model, model_input, gt,
                                         model_output, writer, total_steps, prefix='train_',
                                         model_type='multiscale', skip=False):
    if skip:
        return

    # uniformly sample the image
    dataset.toggle_eval()
    model_input, gt = dataset[0]
    dataset.toggle_eval()

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

    # run the model on uniform samples
    n_channels = gt['img'].shape[-1]
    pred_img = process_batch_in_chunks(model_input, model)['model_out']['output']

    # get pixel idx for each coordinate
    coords = model_input['fine_abs_coords'].detach().cpu().numpy()
    pixel_idx = np.zeros_like(coords).astype(np.int32)
    pixel_idx[..., 0] = np.round((coords[..., 0] + 1.)/2. * (dataset.sidelength[0]-1)).astype(np.int32)
    pixel_idx[..., 1] = np.round((coords[..., 1] + 1.)/2. * (dataset.sidelength[1]-1)).astype(np.int32)
    pixel_idx = pixel_idx.reshape(-1, 2)

    # get pixel idx for each coordinate in frozen patches
    frozen_coords, frozen_values = dataset.get_frozen_patches()
    if frozen_coords is not None:
        frozen_coords = frozen_coords.detach().cpu().numpy()
        frozen_pixel_idx = np.zeros_like(frozen_coords).astype(np.int32)
        frozen_pixel_idx[..., 0] = np.round((frozen_coords[..., 0] + 1.) / 2. * (dataset.sidelength[0] - 1)).astype(np.int32)
        frozen_pixel_idx[..., 1] = np.round((frozen_coords[..., 1] + 1.) / 2. * (dataset.sidelength[1] - 1)).astype(np.int32)
        frozen_pixel_idx = frozen_pixel_idx.reshape(-1, 2)

    # init a new reconstructed image
    display_pred = np.zeros((*dataset.sidelength, n_channels))

    # assign predicted image values into a new array
    # need to use numpy since it supports index assignment
    pred_img = pred_img.reshape(-1, n_channels).detach().cpu().numpy()
    display_pred[[pixel_idx[:, 0]], [pixel_idx[:, 1]]] = pred_img

    # assign frozen image values into the array too
    if frozen_coords is not None:
        frozen_values = frozen_values.reshape(-1, n_channels).detach().cpu().numpy()
        display_pred[[frozen_pixel_idx[:, 0]], [frozen_pixel_idx[:, 1]]] = frozen_values

    # show reconstructed img
    display_pred = torch.tensor(display_pred)[None, ...]
    display_pred = display_pred.permute(0, 3, 1, 2)

    gt_img = gt['img'].reshape(-1, n_channels).detach().cpu().numpy()
    display_gt = np.zeros((*dataset.sidelength, n_channels))
    display_gt[[pixel_idx[:, 0]], [pixel_idx[:, 1]]] = gt_img
    display_gt = torch.tensor(display_gt)[None, ...]
    display_gt = display_gt.permute(0, 3, 1, 2)

    fig = dataset.quadtree.draw()
    writer.add_figure(prefix + 'tiling', fig, global_step=total_steps)

    if 'img' in gt:
        output_vs_gt = torch.cat((display_gt, display_pred), dim=0)
        writer.add_image(prefix + 'gt_vs_pred', make_grid(output_vs_gt, scale_each=False, normalize=True),
                         global_step=total_steps)
        write_psnr(display_pred, display_gt, writer, total_steps, prefix+'img_')


def dict2cuda(a_dict):
    tmp = {}
    for key, value in a_dict.items():
        if isinstance(value, torch.Tensor):
            tmp.update({key: value.cuda()})
        else:
            tmp.update({key: value})
    return tmp


def dict2cpu(a_dict):
    tmp = {}
    for key, value in a_dict.items():
        if isinstance(value, torch.Tensor):
            tmp.update({key: value.cpu()})
        elif isinstance(value, dict):
            tmp.update({key: dict2cpu(value)})
        else:
            tmp.update({key: value})
    return tmp


def process_batch_in_chunks(in_dict, model, max_chunk_size=1024, progress=None):

    in_chunked = []
    for key in in_dict:
        chunks = torch.split(in_dict[key], max_chunk_size, dim=1)
        in_chunked.append(chunks)

    list_chunked_batched_in = \
        [{k: v for k, v in zip(in_dict.keys(), curr_chunks)} for curr_chunks in zip(*in_chunked)]
    del in_chunked

    list_chunked_batched_out_out = {}
    list_chunked_batched_out_in = {}
    for chunk_batched_in in tqdm(list_chunked_batched_in):
        if torch.cuda.is_available():
            chunk_batched_in = {k: v.cuda() for k, v in chunk_batched_in.items()}
        else:
            chunk_batched_in = {k: v for k, v in chunk_batched_in.items()}

        tmp = model(chunk_batched_in)
        tmp = dict2cpu(tmp)

        for key in tmp['model_out']:
            if tmp['model_out'][key] is None:
                continue

            out_ = tmp['model_out'][key].detach().clone().requires_grad_(False)
            list_chunked_batched_out_out.setdefault(key, []).append(out_)

        for key in tmp['model_in']:
            if tmp['model_in'][key] is None:
                continue

            in_ = tmp['model_in'][key].detach().clone().requires_grad_(False)
            list_chunked_batched_out_in.setdefault(key, []).append(in_)

        del tmp, chunk_batched_in

    # Reassemble the output chunks in a batch
    batched_out = {}
    for key in list_chunked_batched_out_out:
        batched_out_lin = torch.cat(list_chunked_batched_out_out[key], dim=1)
        batched_out[key] = batched_out_lin

    batched_in = {}
    for key in list_chunked_batched_out_in:
        batched_in_lin = torch.cat(list_chunked_batched_out_in[key], dim=1)
        batched_in[key] = batched_in_lin

    return {'model_in': batched_in, 'model_out': batched_out}


def subsample_dict(in_dict, num_views, multiscale=False):
    if multiscale:
        out = {}
        for k, v in in_dict.items():
            if v.shape[0] == in_dict['octant_coords'].shape[0]:
                # this is arranged by blocks
                out.update({k: v[0:num_views[0]]})
            else:
                # arranged by rays
                out.update({k: v[0:num_views[1]]})
    else:
        out = {key: value[0:num_views, ...] for key, value in in_dict.items()}

    return out

def get_header(dir, sz):
    hdu = fits.open(os.path.join(dir, 'pdr3_dud/calexp-HSC-G-9813-0%2C0.fits'))[1]
    header = hdu.header
    cutout = Cutout2D(hdu.data, position=(sz//2, sz//2),
                      size=sz, wcs=WCS(header))
    return cutout.wcs.to_header()

def reconstruct(id, coord_dataset, gt, model_input, model, recon_dir, header):

    n_channels = gt['img'].shape[-1]

    with torch.no_grad():
        recon = process_batch_in_chunks\
            (model_input, model, max_chunk_size=512)['model_out']['output']

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # get pixel idx for each coordinate
    coords = model_input['fine_abs_coords'].detach().cpu().numpy()
    pixel_idx = np.zeros_like(coords).astype(np.int32)
    pixel_idx[..., 0] = np.round((coords[..., 0] + 1.)/2.*
                                 (coord_dataset.sidelength[0] - 1)).astype(np.int32)

    pixel_idx[..., 1] = np.round((coords[..., 1] + 1.)/2. *
                                 (coord_dataset.sidelength[1] - 1)).astype(np.int32)
    pixel_idx = pixel_idx.reshape(-1, 2)

    recon = recon.detach().cpu().numpy()[0].transpose((2,0,1))
    gt = gt['img'].detach().cpu().numpy()[0].transpose((2,0,1))

    # record and save metrics
    psnr, ssim, mse = get_metrics(recon, gt)
    print(f'PSNR: {psnr:.04f}, SSIM: {ssim:.04f}, MSE:{mse:.06f}')

    # save images
    recon_fn = os.path.join(recon_dir, '{}'.format(id))
    np.save(recon_fn + '.npy', recon)
    hdu = fits.PrimaryHDU(data=recon, header=header)
    hdu.writeto(recon_fn + '.fits', overwrite=True)

    # save tiling
    tiling_fname = os.path.join(recon_dir, 'tiling_{}.pdf'.format(id))
    coord_dataset.quadtree.draw()
    plt.savefig(tiling_fname)

    return mse, psnr, ssim


def get_metrics(pred_img, gt_img):
    #pred_img = pred_img.detach().cpu().numpy().squeeze()
    #gt_img = gt_img.detach().cpu().numpy().squeeze()

    p = pred_img.transpose(1, 2, 0)
    trgt = gt_img.transpose(1, 2, 0)

    p = (p / 2.) + 0.5
    p = np.clip(p, a_min=0., a_max=1.)

    trgt = (trgt / 2.) + 0.5

    mse = np.mean((p-trgt)**2)
    psnr = skimage.metrics.peak_signal_noise_ratio(p, trgt, data_range=1)
    ssim = skimage.metrics.structural_similarity(p, trgt, multichannel=True, data_range=1)

    return psnr, ssim, mse

'''
# gt/recon, [c,h,w]
def reconstruct(gt, recon, recon_path, loss_dir, header=None):
    sz = gt.shape[1]
    np.save(recon_path + '.npy', recon)

    if header is not None:
        print('GT max', np.round(np.max(gt, axis=(1,2)), 3) )
        print('Recon pixl max ', np.round(np.max(recon, axis=(1,2)), 3) )
        print('Recon stat ', round(np.min(recon), 3), round(np.median(recon), 3),
              round(np.mean(recon), 3), round(np.max(recon), 3))

        hdu = fits.PrimaryHDU(data=recon, header=header)
        hdu.writeto(recon_path + '.fits', overwrite=True)

        losses = get_losses(gt, recon, None, [1,2,4])

        for nm, loss in zip(['_mse','_psnr','_ssim'], losses):
            fn = '0_'+str(sz)+nm+'_0.npy'
            loss = np.expand_dims(loss, axis=0)
            print(loss)
            np.save(os.path.join(loss_dir, fn), loss)
'''
