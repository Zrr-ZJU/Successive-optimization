import os
import torch

from data import train_dataloader
from utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
from optics_model import OpticalNet
from skimage.metrics import peak_signal_noise_ratio

import logging


def _train(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.L1Loss()
    optics = OpticalNet(args.lens_file)
    pad_size = optics.args['kernel_size'] // 2
    dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker, pad=pad_size * 2)
    max_iter = len(dataloader)
    optimizer_list = []
    scheduler_list = []

    ##############################################
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.gamma)
    optimizer_list.append(optimizer)
    scheduler_list.append(scheduler)

    ###############################################
    if args.e2e:
        optic_optm = optics.get_optimizer_Adam()
        optic_scheduler = torch.optim.lr_scheduler.MultiStepLR(optic_optm, args.lr_steps, args.gamma)
        optimizer_list.append(optic_optm)
        scheduler_list.append(optic_scheduler)
    optic_keys = optics.args['loss_functions'].keys()
    epoch = 1
    if args.resume:
        state = torch.load(args.resume, map_location=lambda storage, loc: storage)
        epoch = state['epoch']
        for o, st in zip(optimizer_list, state['optimizer_list']):
            o.load_state_dict(st)
        for s, st in zip(scheduler_list, state['scheduler_list']):
            s.load_state_dict(st)
        model.load_state_dict(state['model'])
        print('Resume from %d'%epoch)
        epoch += 1

    writer = SummaryWriter(args.model_save_dir)
    logging.info(args)
    logging.info(optics.args)
    epoch_dict = {
        'pixel': Adder(),
        'fft': Adder(),
        'psnr': Adder(),
        'bpsnr': Adder(),
    }
    iter_dict = {
        'pixel': Adder(),
        'fft': Adder(),
        'psnr': Adder(),
        'bpsnr': Adder(),
    }

    epoch_dict.update({k: Adder() for k in optic_keys})
    iter_dict.update({k: Adder() for k in optic_keys})

    epoch_timer = Timer('m')
    iter_timer = Timer('m')


    for epoch_idx in range(epoch, args.num_epoch + 1):

        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(dataloader):

            input_img, label_img = batch_data
            input_img = input_img.to(device).to(torch.float32)
            # ----------------------------------
            optics.update()
            fov = torch.rand(1) * optics.args['max_view']
            azimuth = torch.rand(1) * 360.
            blur_img, field_img, stack_list = optics.rendering_patch(fov[0], azimuth[0], input_img)
            blur_img += torch.randn_like(blur_img) * args.add_noise
            blur_img = torch.clamp(blur_img, 0, 1)
            blur_img = torch.cat([blur_img, field_img.detach()], dim=1)
            loss_optic_dict, optic_eval = optics.get_optical_loss_dict(stack_list)
            loss_optic = sum(loss_optic_dict.values())
            label_img = input_img[:, :, pad_size:-pad_size, pad_size:-pad_size]
            # --------------------------------
            for o in optimizer_list:
                o.zero_grad()
            if args.abla_net:
                blur_img.detach()
            pred_img = model(blur_img)
            label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
            label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')
            l1 = criterion(pred_img[0], label_img4)
            l2 = criterion(pred_img[1], label_img2)
            l3 = criterion(pred_img[2], label_img)
            loss_content = l1+l2+l3

            label_fft1_temp = torch.fft.fft2(label_img4, dim=(-2, -1))
            label_fft1 = torch.stack((label_fft1_temp.real, label_fft1_temp.imag), -1)

            pred_fft1_temp = torch.fft.fft2(pred_img[0], dim=(-2, -1))
            pred_fft1 = torch.stack((pred_fft1_temp.real, pred_fft1_temp.imag), -1)

            label_fft2_temp = torch.fft.fft2(label_img2, dim=(-2, -1))
            label_fft2 = torch.stack((label_fft2_temp.real, label_fft2_temp.imag), -1)

            pred_fft2_temp = torch.fft.fft2(pred_img[1], dim=(-2, -1))
            pred_fft2 = torch.stack((pred_fft2_temp.real, pred_fft2_temp.imag), -1)

            label_fft3_temp = torch.fft.fft2(label_img, dim=(-2, -1))
            label_fft3 = torch.stack((label_fft3_temp.real, label_fft3_temp.imag), -1)

            pred_fft3_temp = torch.fft.fft2(pred_img[2], dim=(-2, -1))
            pred_fft3 = torch.stack((pred_fft3_temp.real, pred_fft3_temp.imag), -1)

            f1 = criterion(pred_fft1, label_fft1)
            f2 = criterion(pred_fft2, label_fft2)
            f3 = criterion(pred_fft3, label_fft3)
            loss_fft = f1+f2+f3

            loss = loss_content + 0.1 * loss_fft + loss_optic*args.optic_weight
            loss.backward()

            for o in optimizer_list:
                o.step()
            for k in optic_keys:
                iter_dict[k](optic_eval[k])
                epoch_dict[k](optic_eval[k])

            iter_dict['pixel'](loss_content.item())
            iter_dict['fft'](loss_fft.item())

            epoch_dict['pixel'](loss_content.item())
            epoch_dict['fft'](loss_fft.item())

            # -------------------------------
            pred_clip = torch.clamp(pred_img[2], 0, 1)
            p_numpy = pred_clip.squeeze(0).detach().cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()
            blur_numpy = blur_img[:, :3, :, :].squeeze(0).detach().cpu().numpy()

            psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)
            blur_psnr = peak_signal_noise_ratio(blur_numpy, label_numpy, data_range=1)

            iter_dict['psnr'](psnr)
            epoch_dict['psnr'](psnr)
            iter_dict['bpsnr'](blur_psnr)
            epoch_dict['bpsnr'](blur_psnr)
            # ----------------------------------

            if (iter_idx + 1) % args.print_freq == 0:
                lr = check_lr(optimizer)
                sentence = "Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f PSNR: %7.4f BPSNR: %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, lr, iter_dict['pixel'].average(), iter_dict['fft'].average(),
                    iter_dict['psnr'].average(), iter_dict['bpsnr'].average())
                logging.info(sentence)

                writer.add_scalar('Pixel Loss', iter_dict['pixel'].average(), iter_idx + (epoch_idx-1) * max_iter)
                writer.add_scalar('FFT Loss', iter_dict['fft'].average(), iter_idx + (epoch_idx - 1) * max_iter)
                writer.add_scalar('PSNR', iter_dict['psnr'].average(), iter_idx + (epoch_idx - 1) * max_iter)
                writer.add_scalar('blur PSNR', iter_dict['bpsnr'].average(), iter_idx + (epoch_idx - 1) * max_iter)
                for k in optic_keys:
                    writer.add_scalar(k, iter_dict[k].average(), iter_idx + (epoch_idx - 1) * max_iter)

                iter_timer.tic()

                word_list = [f"{k}: {iter_dict[k].average():7.4f}" for k in optic_keys]
                sentence = 'Iter: ' + ' '.join(word_list)
                logging.info(sentence)

                for v in iter_dict.values():
                    v.reset()

        overwrite_name = os.path.join(args.model_save_dir, args.save_name+'.pkl')
        torch.save({'model': model.state_dict(),
                    'optimizer_list': [o.state_dict() for o in optimizer_list],
                    'scheduler_list': [s.state_dict() for s in scheduler_list],
                    'epoch': epoch_idx}, overwrite_name)
        optics.lens.save_to_json(os.path.join(args.model_save_dir, args.save_name+'.json'))
        if epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, args.save_name + '_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict(),
                        'optimizer_list': [o.state_dict() for o in optimizer_list],
                        'scheduler_list': [s.state_dict() for s in scheduler_list],
                        'epoch': epoch_idx}, save_name)
            optics.lens.save_to_json(os.path.join(args.model_save_dir, args.save_name + '_%d.json' % epoch_idx))
        sentence = "EPOCH: %02d\n Elapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f Epoch PSNR: %7.4f BPSNR: %7.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_dict['pixel'].average(), epoch_dict['fft'].average(),
            epoch_dict['psnr'].average(), epoch_dict['bpsnr'].average())
        logging.info(sentence)

        word_list = [f"{k}: {epoch_dict[k].average():7.4f}" for k in optic_keys]
        sentence = 'EPOCH: ' + ' '.join(word_list)
        logging.info(sentence)

        for v in epoch_dict.values():
            v.reset()

        for s in scheduler_list:
            s.step()


    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    optics.lens.save_to_json(os.path.join(args.model_save_dir, args.save_name + 'Final.json'))
    torch.save({'model': model.state_dict()}, save_name)
