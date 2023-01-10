import os
from argparse import ArgumentParser
import torch.optim as optim
import torch
import random
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from generator import Generator
from discriminator import Discriminator
import torchvision.utils as vutils
from config import HP
from dataset_face import face_loader, invTrans

logger = SummaryWriter('./log')

# seed init: Ensure Reproducible Result
torch.random.manual_seed(HP.seed)
torch.cuda.manual_seed(HP.seed)
random.seed(HP.seed)
np.random.seed(HP.seed)


def save_checkpoint(model_, epoch_, optm, checkpoint_path):
    save_dict = {
        'epoch': epoch_,
        'model_state_dict': model_.state_dict(),
        'optimizer_state_dict': optm.state_dict()
    }
    torch.save(save_dict, checkpoint_path)


def train():
    parser = ArgumentParser(description='Model Training')
    parser.add_argument(
        '--c', # G and D checkpoint path: model_g_xxx.pth~model_d_xxx.pth
        default=None,
        type=str,
        help='training from scratch or resume training'
    )
    args = parser.parse_args()

    # model init
    G = Generator() # new a generator model instance
    G.apply(G.weights_init) # apply weight init for G
    D = Discriminator()  # new a discriminator model instance
    D.apply(D.weights_init)  # apply weight init for G
    G.to(HP.device)
    D.to(HP.device)

    # loss criterion
    criterion = nn.BCELoss() # binary classification loss

    # optimizer
    optimizer_g = optim.Adam(G.parameters(), lr=HP.init_lr, betas=(HP.beta, 0.999))
    optimizer_d = optim.Adam(D.parameters(), lr=HP.init_lr, betas=(HP.beta, 0.999))

    start_epoch, step = 0, 0 # start position

    if args.c: # model_g_xxx.pth~model_d_xxx.pth
        model_g_path = args.c.split('~')[0]
        checkpoint_g = torch.load(model_g_path)
        G.load_state_dict(checkpoint_g['model_state_dict'])
        optimizer_g.load_state_dict(checkpoint_g['optimizer_state_dict'])
        start_epoch_gc = checkpoint_g['epoch']

        model_d_path = args.c.split('~')[1]
        checkpoint_d = torch.load(model_d_path)
        D.load_state_dict(checkpoint_d['model_state_dict'])
        optimizer_d.load_state_dict(checkpoint_d['optimizer_state_dict'])
        start_epoch_dc = checkpoint_d['epoch']

        start_epoch = start_epoch_gc if start_epoch_dc > start_epoch_gc else start_epoch_dc
        print('Resume Training From Epoch: %d' % start_epoch)
    else:
        print('Training From Scratch!')

    G.train()   # set training flag
    D.train()   # set training flag

    # fixed latent z for G logger
    fixed_latent_z = torch.randn(size=(64, 100), device=HP.device)

    # main loop
    for epoch in range(start_epoch, HP.epochs):
        print('Start Epoch: %d, Steps: %d' % (epoch, len(face_loader)))
        for batch, _ in face_loader: # batch shape [N, 3, 64, 64]
            # log(D(x)) + log(1-D(G(z)))
            b_size = batch.size(0) # 64
            optimizer_d.zero_grad() # gradient clean
            # gt: ground truth: real data
            # label smoothing: 0.85, 0.1 /  softmax: logist output -> [0, 1] Temperature Softmax
            # multi label: 1.jpg : cat and dog
            labels_gt = torch.full(size=(b_size, ), fill_value=0.9, dtype=torch.float, device=HP.device)
            predict_labels_gt = D(batch.to(HP.device)).squeeze() # [64, 1] -> [64,]
            loss_d_of_gt = criterion(predict_labels_gt, labels_gt)

            labels_fake = torch.full(size=(b_size, ), fill_value=0.1, dtype=torch.float, device=HP.device)
            latent_z = torch.randn(size=(b_size, HP.z_dim), device=HP.device)
            predict_labels_fake = D(G(latent_z)).squeeze() # [64, 1] - > [64,]
            loss_d_of_fake = criterion(predict_labels_fake, labels_fake)

            loss_D = loss_d_of_gt + loss_d_of_fake  # add the two parts
            loss_D.backward()
            optimizer_d.step()
            logger.add_scalar('Loss/Discriminator', loss_D.mean().item(), step)
            
            # log(1-D(G(z)))
            optimizer_g.zero_grad() # G gradient clean
            latent_z = torch.randn(size=(b_size, HP.z_dim), device=HP.device)
            labels_for_g = torch.full(size=(b_size, ), fill_value=0.9, dtype=torch.float, device=HP.device)
            predict_labels_from_g = D(G(latent_z)).squeeze() # [N, ]

            loss_G = criterion(predict_labels_from_g, labels_for_g)
            loss_G.backward()
            optimizer_g.step()
            logger.add_scalar('Loss/Generator', loss_G.mean().item(), step)

            if not step % HP.verbose_step:
                with torch.no_grad():
                    fake_image_dev = G(fixed_latent_z)
                    logger.add_image('Generator Faces', invTrans(vutils.make_grid(fake_image_dev.detach().cpu(), nrow=8)), step)

            if not step % HP.save_step: # save G and D
                model_path = 'model_g_%d_%d.pth' % (epoch, step)
                save_checkpoint(G, epoch,optimizer_g, os.path.join('model_save', model_path))
                model_path = 'model_d_%d_%d.pth' % (epoch, step)
                save_checkpoint(D, epoch, optimizer_d, os.path.join('model_save', model_path))

            step += 1
            logger.flush()
            print('Epoch: [%d/%d], step: %d G loss: %.3f, D loss %.3f' %
                  (epoch, HP.epochs, step, loss_G.mean().item(), loss_D.mean().item()))

    logger.close()


if __name__ == '__main__':
    train()


