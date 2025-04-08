import argparse

def get_args():
    parser = argparse.ArgumentParser(description='sensor')
    parser.add_argument('--xulie', default=1, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=9e-5,type=float)
    parser.add_argument('--n_epoch', default=441, type=int)
    parser.add_argument('--hidden', default=400, type=int)
    parser.add_argument('--uz', default=50, type=int)
    parser.add_argument('--lambda_pre', default=1, type=float)
    parser.add_argument('--lambda_rel', default=1,type=float)
    parser.add_argument('--lambda_inv', default=1, type=float)
    parser.add_argument('--input', default=420, type=int)
    parser.add_argument('--lambda_infor', default=1, type=float)
    args = parser.parse_args()

    return args