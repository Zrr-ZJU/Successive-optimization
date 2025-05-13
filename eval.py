import os

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import functional as F
import numpy as np
from utils import Adder
from data import valid_dataloader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import time
from optics_model import OpticalNet
import lpips


def calculate_psnr_ssim(img1, img2):
    img1_np = img1.permute(1, 2, 0).cpu().numpy()  # (B, C, H, W) -> (H, W, C)
    img2_np = img2.permute(1, 2, 0).cpu().numpy()
    psnr_value = peak_signal_noise_ratio(img1_np, img2_np, data_range=1.0)
    ssim_value = structural_similarity(img1_np, img2_np, channel_axis=-1, data_range=1.0)
    return psnr_value, ssim_value


def _eval(model, args):
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])
    lpips_model = lpips.LPIPS(net='alex', spatial=False).cuda().eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 7
    dataloader = valid_dataloader(args.valid_data_dir, batch_size=batch_size, num_workers=0)
    optics = OpticalNet(args.lens_file)
    adder = Adder()
    model.eval()

    H=optics.args['H']
    W=optics.args['W']
    patch_itv = optics.args['patch_itv']

    if args.save_kernel:
        optics.analysis.save_zmx(args.result_dir + f"{args.save_name}.zmx")
        kernel_set, field_set = optics.rendering_psf_map(H, W, patch_itv, file_name=args.result_dir + f"{args.save_name}_psf_{patch_itv}.pdf", save=True)
        kernel_set = kernel_set.to(torch.float32)
        np.save(args.result_dir + f"{args.save_name}_psf_{patch_itv}.npy", kernel_set.cpu().numpy())
    else:
        kernel_set = torch.from_numpy(np.load(args.result_dir + f"{args.save_name}_psf_{patch_itv}.npy")).to(device)
    with torch.no_grad():
        psnr_adder = Adder()
        ssim_adder = Adder()
        bpsnr_adder = Adder()
        bssim_adder = Adder()
        lpips_adder = Adder()
        blpips_adder = Adder()

        # Main Evaluation
        for iter_idx, data in enumerate(dataloader):
            torch.cuda.empty_cache()
            input_img = data.to(torch.float32)
            input_img = F.center_crop(input_img, [H + 130, W + 130]).to(device)
            label_img = input_img[:, :, 65:-65, 65:-65]


            tm = time.time()
            patchs, fields = optics.crop_patch(input_img, patch_size=patch_itv, kernel_size=31)
            output_patchs = []
            blur_patchs = []
            label_patchs = []
            H_num = int(H / patch_itv)
            W_num = int(W / patch_itv)
            for h_index in range(H_num):
                for w_index in range(W_num):
                    blur_patch = optics.blur_patch(patchs[h_index][w_index], kernel_set[h_index, w_index])
                    blur_patch += torch.randn_like(blur_patch) * args.add_noise
                    blur_patch = torch.clamp(blur_patch, 0, 1)
                    field = fields[h_index][w_index]
                    blur_patchs.append(blur_patch[:, :, 50:-50, 50:-50])
                    model_in = torch.cat([blur_patch, field.permute(2, 0, 1).repeat(blur_patch.shape[0], 1, 1, 1).to(device)], dim=1)
                    pred_patch = model(model_in)[2]
                    output_patchs.append(pred_patch[:, :, 50:-50, 50:-50])

            pred = optics.sew_up_img(output_patchs, patch_size=patch_itv, img_size=[H, W]).to(torch.float32).to(device)
            blur = optics.sew_up_img(blur_patchs, patch_size=patch_itv, img_size=[H, W]).to(torch.float32).to(device)
            pred = torch.clamp(pred, 0, 1)

            elapsed = time.time() - tm
            adder(elapsed)

            pred_clip = torch.clamp(pred, 0, 1)

            for p,b,l,i in zip(pred_clip, blur, label_img, range(batch_size)):
                psnr, ssim = calculate_psnr_ssim(p, l)
                bpsnr, bssim = calculate_psnr_ssim(b, l)
                lpips_score = lpips_model(p, l).item()
                blpips_score = lpips_model(b, l).item()
            # save img
                if args.save_image:
                    F.to_pil_image(l.cpu(), 'RGB').save(args.result_image_dir + f"vlabel_{iter_idx * batch_size + i + 1}_{args.save_name}.png")
                    F.to_pil_image(p.cpu(), 'RGB').save(args.result_image_dir + f"vpred_{iter_idx * batch_size + i + 1}_{args.save_name}.png")
                    F.to_pil_image(b.cpu(), 'RGB').save(args.result_image_dir + f"vblur_{iter_idx * batch_size + i + 1}_{args.save_name}.png")

                psnr_adder(psnr)
                ssim_adder(ssim)
                bpsnr_adder(bpsnr)
                bssim_adder(bssim)
                lpips_adder(lpips_score)
                blpips_adder(blpips_score)


                # lpips_adder(lpips_score)
                print('%d iter PSNR: %.4f time: %f' % (iter_idx * batch_size + i + 1, psnr, elapsed))
                print('%d iter SSIM: %.4f time: %f' % (iter_idx * batch_size + i + 1, ssim, elapsed))
                print('%d iter LPIPS: %.4f time: %f' % (iter_idx * batch_size + i + 1, lpips_score, elapsed))
                print('%d iter BPSNR: %.4f time: %f' % (iter_idx * batch_size + i + 1, bpsnr, elapsed))
                print('%d iter BSSIM: %.4f time: %f' % (iter_idx * batch_size + i + 1, bssim, elapsed))
                print('%d iter BLPIPS: %.4f time: %f' % (iter_idx * batch_size + i + 1, blpips_score, elapsed))

        print('==========================================================')
        print('The average PSNR is %.4f dB' % (psnr_adder.average()))
        print('The average SSIM is %.4f' % (ssim_adder.average()))
        print('The average LPIPS is %.4f' % (lpips_adder.average()))
        print('The average BPSNR is %.4f dB' % (bpsnr_adder.average()))
        print('The average BSSIM is %.4f' % (bssim_adder.average()))
        print('The average BLPIPS is %.4f' % (blpips_adder.average()))
        print("Average time: %f" % adder.average())
