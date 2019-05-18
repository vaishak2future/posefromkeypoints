import os
import json
import argparse
from collections import namedtuple

class TrainOptions(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser()

        req = self.parser.add_argument_group('Required')
        req.add_argument('--name', required=True, help='Name of the experiment')

        gen = self.parser.add_argument_group('General')
        gen.add_argument('--num_workers', type=int, default=8, help='Number of processes used for data loading')

        io = self.parser.add_argument_group('io')
        io.add_argument('--log_dir', default='logs', help='Directory to store logs')

        data_proc = self.parser.add_argument_group('Data Preprocessing')
        data_proc.add_argument('--degrees', type=float, default=45, help='Random rotation angle in the range [-degrees, degrees]')
        data_proc.add_argument('--crop_size', type=int, default=256, help='Size of cropped image to feed to the network')
        data_proc.add_argument('--heatmap_size', type=int, default=64, help='Size of output heatmaps')
        data_proc.add_argument('--detection_thresh', type=float, default=1e-1, help='Size of output heatmaps')
        data_proc.add_argument('--dist_thresh', type=float, default=10, help='Size of output heatmaps')

        arch = self.parser.add_argument_group('Architecture')
        arch.add_argument('--num_stacks', type=int, default=1, help='Number of channels in the hourglass') 
        arch.add_argument('--num_blocks', type=int, default=1, help='Number of hourglasses') 
        arch.add_argument('--num_classes', type=int, default=10, help='Number of stacked residual blocks') 

        train = self.parser.add_argument_group('Training Options')
        train.add_argument('--num_epochs', type=int, default=800, help='Total number of training epochs')
        train.add_argument('--batch_size', type=int, default=8, help='Batch size')
        train.add_argument('--test_batch_size', type=int, default=32, help='Batch size')
        train.add_argument('--summary_steps', type=int, default=100, help='Summary saving frequency')
        train.add_argument('--test_steps', type=int, default=100, help='Testing frequency')

        optim = self.parser.add_argument_group('Optimization')
        optim.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
        return 

    def parse_args(self, args=None):
        self.args = self.parser.parse_args(args)
        self.args.log_dir = os.path.join(os.path.abspath(self.args.log_dir), self.args.name)
        self.args.summary_dir = os.path.join(self.args.log_dir, 'tensorboard')
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        self.args.checkpoint_dir = os.path.join(self.args.log_dir, 'checkpoints')
        if not os.path.exists(self.args.checkpoint_dir):
            os.makedirs(self.args.checkpoint_dir)
        self.save_dump()
        return self.args

    def save_dump(self):
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        with open(os.path.join(self.args.log_dir, "config.json"), "w") as f:
            json.dump(vars(self.args), f, indent=4)
        return
