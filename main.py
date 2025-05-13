import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import argparse
from torch.backends import cudnn
from models.MIMOUNet import build_net
from train import _train
from eval import _eval
import logging


def set_logger(dir='./'):
    logger = logging.getLogger()
    logger.setLevel('DEBUG')
    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel('INFO')

    fhlr = logging.FileHandler(f"{dir}/output.log")
    fhlr.setFormatter(formatter)
    fhlr.setLevel('INFO')


    logger.addHandler(chlr)
    logger.addHandler(fhlr)



def main(args):
    # CUDNN
    cudnn.benchmark = True

    os.makedirs(args.model_save_dir, exist_ok=True)
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    set_logger(args.model_save_dir)
    model = build_net(args.model_name)
    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train':
        _train(model, args)

    elif args.mode == 'test':
        _eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    num_epoch = 1000
    step = 200
    batch = 16
    lr_scale = 1
    e2e = True
    optic_weight = 0.1
    abla_net = False
    exam_name = f'phone'

    # Directories
    parser.add_argument('--model_name', default='MIMO-UNetFov', choices=['MIMO-UNet', 'MIMO-UNetPlus', 'MIMO-UNetFov'], type=str)
    parser.add_argument('--data_dir', type=str, default=r'/data1/DIV2K/DIV2K_train_HR/')
    parser.add_argument('--valid_data_dir', type=str, default=r'/data1/zhoujw/DIV8K/label/6000_8000/train/')
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)

    # Optics
    parser.add_argument('--lens_file', type=str, default='./configs/phone_design.yml')
    parser.add_argument('--save_name', type=str, default=exam_name)
    parser.add_argument('--e2e', type=bool, default=e2e, choices=[True, False])
    parser.add_argument('--abla_net', type=bool, default=abla_net, choices=[True, False])
    parser.add_argument('--optic_weight', type=float, default=optic_weight)
    parser.add_argument('--add_noise', type=float, default=0.03)

    # Train
    parser.add_argument('--batch_size', type=int, default=batch)
    parser.add_argument('--learning_rate', type=float, default=(batch/8)*lr_scale*1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=num_epoch)
    parser.add_argument('--print_freq', type=int, default=5)
    parser.add_argument('--num_worker', type=int, default=batch//2)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--valid_freq', type=int, default=100)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lr_steps', type=list, default=[(x+1) * step for x in range(num_epoch//step)])

    # Test
    parser.add_argument('--test_model', type=str, default='')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])

    args = parser.parse_args()

    args.model_save_dir = os.path.join('results_paper/', args.model_name, f'{exam_name}/')
    args.result_dir = os.path.join('results/', args.model_name, 'result_image/')

    torch.manual_seed(1)
    torch.multiprocessing.set_start_method('spawn')
    torch.set_default_dtype(torch.float32)
    main(args)
