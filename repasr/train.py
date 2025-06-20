# flake8: noqa
import os.path as osp

import repasr.archs
import repasr.models
import repasr.data
import repasr.losses
from basicsr.train import train_pipeline


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
