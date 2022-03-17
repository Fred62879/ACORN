
import os
import time
import utils
import torch
import shutil
import numpy as np

from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter


def train(model, optim, train_dataloader, start_epoch, epochs, total_steps,
          model_file_id, lr, epochs_til_checkpoint, checkpoints_dir, loss_fn,
          pruning_fn, double_precision=False, clip_grad=False,
          loss_schedules=None, objs_to_save={}, epochs_til_pruning=4):

    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        pbar.update(total_steps)

        for epoch in range(start_epoch, epochs):
            start_time = time.time()

            # prune
            if not epoch % epochs_til_pruning and epoch:
                pruning_fn(model, train_dataloader.dataset)
            retile = False if not (epoch + 1) % epochs_til_pruning else True

            for step, (model_input, gt) in enumerate(train_dataloader):
                tmp = {}
                for key, value in model_input.items():
                    if isinstance(value, torch.Tensor):
                        if torch.cuda.is_available(): value = value.cuda()
                        tmp.update({key: value})
                    else:
                        tmp.update({key: value})
                model_input = tmp

                tmp = {}
                for key, value in gt.items():
                    if isinstance(value, torch.Tensor):
                        if torch.cuda.is_available(): value = value.cuda()
                        tmp.update({key: value})
                    else:
                        tmp.update({key: value})
                gt = tmp

                if double_precision:
                    model_input = {key: value.double() for key, value in model_input.items()}
                    gt = {key: value.double() for key, value in gt.items()}

                model_output = model(model_input)
                losses = loss_fn(model_output, gt, total_steps, retile=retile)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        single_loss *= loss_schedules[loss_name](total_steps)
                    train_loss += single_loss

                optim.zero_grad()
                train_loss.backward()

                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                optim.step()
                pbar.update(1)
                total_steps += 1

            # after epoch
            tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" %
                       (epoch, train_loss, time.time() - start_time))

            # save model
            if not epoch % epochs_til_checkpoint or epoch == 0:
                save_dict = {
                    'epoch': epoch,
                    'total_steps': total_steps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict()
                }
                save_dict.update(objs_to_save)

                model_fn = os.path.join(checkpoints_dir, 'model{}.pth'.format(model_file_id))
                torch.save(save_dict, model_fn)
                model_file_id += 1
