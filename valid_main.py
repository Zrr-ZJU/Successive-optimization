import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import argparse
from torch.backends import cudnn
from models.MIMOUNet import build_net
from train import _train
from eval import _eval


def main(args):
    # CUDNN
    cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if not os.path.exists(args.result_image_dir):
        os.makedirs(args.result_image_dir)

    model = build_net(args.model_name)
    # print(model)
    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train':
        _train(model, args)

    elif args.mode == 'test':
        _eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    exam_name = 'phone/phone'
    model_name = f"./results_paper/MIMO-UNetFov/{exam_name}.pkl"

    # Directories
    parser.add_argument('--model_name', default='MIMO-UNetFov', choices=['MIMO-UNet', 'MIMO-UNetPlus', 'MIMO-UNetFov'], type=str)
    parser.add_argument('--valid_data_dir', type=str, default=r'/data1/zhoujw/DIV8K/label/6000_8000/train/')

    parser.add_argument('--mode', default='test', choices=['train', 'test'], type=str)

    parser.add_argument('--lens_file', type=str, default='./configs/phone_design_eval.yml')
    parser.add_argument('--add_noise', type=float, default=0.03)

    parser.add_argument('--test_model', type=str, default=model_name)
    parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])
    parser.add_argument('--save_kernel', type=bool, default=True, choices=[True, False])
    parser.add_argument('--save_name', type=str, default='phone')
    args = parser.parse_args()

    args.model_save_dir = os.path.join('results/', args.model_name, 'phone/')
    args.result_dir = os.path.join('results/', args.model_name, 'result_image/')
    args.result_image_dir = os.path.join('results/', args.model_name, 'result_image/', args.save_name + '/')

    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.multiprocessing.set_start_method('spawn')
    # torch.set_default_dtype(torch.float32)
    main(args)
