import argparse

def load_cifar_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('--z', default=256, type=int, help='latent space width')
    parser.add_argument('--ze', default=512, type=int, help='encoder dimension')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=200000, type=int)
    parser.add_argument('--target', default='mednet', type=str)
    parser.add_argument('--dataset', default='cifar', type=str)
    parser.add_argument('--beta', default=10, type=float)
    parser.add_argument('--exp', default='0', type=str)
    parser.add_argument('--model', default='full', type=str)
    parser.add_argument('--pretrain_e', default=False, type=bool)
    parser.add_argument('--scratch', default=False, type=bool)

    args = parser.parse_args()
    return args


def load_mnist_args():

    parser = argparse.ArgumentParser(description='param-wgan')
    parser.add_argument('--z', default=128, type=int, help='latent space width')
    parser.add_argument('--ze', default=256, type=int, help='encoder dimension')
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--epochs', default=200000, type=int)
    parser.add_argument('--target', default='small2', type=str)
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--beta', default=10, type=int)
    parser.add_argument('--use_x', default=False, type=bool)
    parser.add_argument('--pretrain_e', default=False, type=bool)
    parser.add_argument('--scratch', default=False, type=bool)
    parser.add_argument('--exp', default='0', type=str)
    parser.add_argument('--model', default='small', type=str)
    parser.add_argument('--task', default='plot', type=str)
    parser.add_argument('--cdf', default=False, type=bool)

    args = parser.parse_args()
    return args


