#!/usr/bin/python
# -*- coding: utf-8 -*-

#  Created by Brian B. Moser.
#  Contact: Brian.Moser@DFKI.de

import argparse
import os
import sys
import inspect

path_file = os.path.abspath(
    os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
)     # noqa

sys.path.insert(0, path_file + '/..')   # noqa

from lucky_trainer.utils import get_dataset, start_training
from src.models.Seq2Seq import Seq2Seq

parser = argparse.ArgumentParser()
parser.add_argument("-mf",
                    "--music_folder",
                    type=str,
                    required=True,
                    help="Music Folder")
args = parser.parse_args()
args_mf = args.music_folder


def main(mf):
    # Parameter settings
    train_params = {
        'skip_test': True,
        'max_epochs': 500,
        'batch_size': 32,
        'early_stopping_patience': 20,
        'class_dim': 1,
        'loss': 'music_multi_loss',
        'optimizer': 'Adam'
    }

    dataset_params = {
        'train_filename':
            path_file +
            '/../data/' + mf + '/input_data/train',
        'validation_filename':
            path_file +
            '/../data/' + mf + '/input_data/val'
    }

    # Load iterable datasets
    train_loader = get_dataset(
       dataset_params['train_filename'],
       train_params['batch_size']
    )
    validation_loader = get_dataset(
       dataset_params['validation_filename'],
       train_params['batch_size'],
       shuffle=False
    )

    # Instantiate model
    filename = 'Seq2Seq_' + mf

    # Train the model
    start_training(Seq2Seq(next(iter(train_loader))[0][0].shape), None, train_params, dataset_params,
                   train_loader, validation_loader, None,
                   "/../data/saved_models", filename)


# e.g. python conf_Seq2Seq.py -mf "edm"
if __name__ == "__main__":
    main(args_mf)
