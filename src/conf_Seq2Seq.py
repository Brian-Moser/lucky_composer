#!/usr/bin/python
# -*- coding: utf-8 -*-

#  Created by Brian B. Moser.
#  Contact: Brian.Moser@DFKI.de

import argparse
import os
import sys

import torch.nn as nn
import torchvision.models as models
import inspect

path_file = os.path.abspath(
    os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
)     # noqa

sys.path.insert(0, path_file + '/..')   # noqa

from lucky_trainer.utils import get_dataset, start_training
from src.models.Seq2Seq import Seq2Seq

parser = argparse.ArgumentParser()
parser.add_argument("-o",
                    "--output",
                    type=str,
                    nargs='?',
                    default=path_file + "/../data/saved_models",
                    help="output directory for the model")
args = parser.parse_args()
args_output = args.output


def main(output_directory):
    # Parameter settings
    train_params = {
        'skip_test': True,
        'max_epochs': 200,
        'batch_size': 8,
        'early_stopping_patience': 20,
        'acc_metric': 'classification',
        'class_dim': 1,
        'top_k': 5,
        'loss': 'MSELoss',
        'optimizer': 'Adam'
    }

    dataset_params = {
        'train_filename':
            path_file +
            '/../data/edm/input_data/train',
        'validation_filename':
            path_file +
            '/../data/edm/input_data/val'
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
    filename = 'Seq2Seq_edm'

    # Train the model
    start_training(Seq2Seq(next(iter(train_loader))[0][0].shape), None, train_params, dataset_params,
                   train_loader, validation_loader, None,
                   output_directory, filename)


if __name__ == "__main__":
    main(args_output)
