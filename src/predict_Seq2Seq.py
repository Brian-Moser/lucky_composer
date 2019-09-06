#!/usr/bin/python
# -*- coding: utf-8 -*-

#  Created by Brian B. Moser.
#  Contact: Brian.Moser@DFKI.de

import os
import sys
import inspect
import pickle
import torch
import random
import argparse
import numpy as np
from music21 import note, chord, stream, duration

path_file = os.path.abspath(
    os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
)     # noqa

sys.path.insert(0, path_file + '/..')   # noqa

from src.models.Seq2Seq import Seq2Seq

# Device configuration (DO NOT EDIT)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("-mf",
                    "--music_folder",
                    type=str,
                    required=True,
                    help="Music Folder")
parser.add_argument("-sl",
                    "--seq_len",
                    type=int,
                    nargs='?',
                    default=25,
                    help="Sequence length")
parser.add_argument("-n",
                    "--numbers",
                    type=int,
                    nargs='?',
                    default=1,
                    help="Amount of compositions added")
args = parser.parse_args()
args_mf = args.music_folder
args_sl = args.seq_len
args_n = args.numbers


def main(mf, sl, n):
    with open(path_file + "/../data/" + mf + "/element_key_dict.pkl", 'rb') as f:
        element_key_dict = pickle.load(f)
    with open(path_file + "/../data/" + mf + "/durations_key_dict.pkl", 'rb') as f:
        durations_key_dict = pickle.load(f)
    with open(path_file + "/../data/" + mf + "/offsets_key_dict.pkl", 'rb') as f:
        offsets_key_dict = pickle.load(f)

    element_key_dict_rev = dict((v, k) for k, v in element_key_dict.items())
    durations_key_dict_rev = dict((v, k) for k, v in durations_key_dict.items())
    offsets_key_dict_rev = dict((v, k) for k, v in offsets_key_dict.items())

    max_key = max(len(element_key_dict), len(durations_key_dict), len(offsets_key_dict))
    # Instantiate model
    model_dict = torch.load(path_file + "/../data/saved_models/Seq2Seq_edm.pth")
    model = Seq2Seq([sl, 3, max_key])
    model.eval()
    model.load_state_dict(model_dict['model_state'])
    model = model.to(device)

    for _ in range(n):
        input_melody = []
        for _ in range(sl):
            np_arr = np.stack(np.array([np.zeros(max_key, dtype=np.float32),
                              np.zeros(max_key, dtype=np.float32),
                              np.zeros(max_key, dtype=np.float32)]))
            np_arr[0][random.randint(0, len(element_key_dict)-1)] = 1
            np_arr[1][random.randint(0, len(durations_key_dict)-1)] = 1
            np_arr[2][random.randint(0, len(offsets_key_dict)-1)] = 1
            input_melody.append(np_arr)
        input_melody = torch.from_numpy(np.array(input_melody))
        input_melody = input_melody.view(1, sl, 3, max_key).to(device)

        prediction = model(input_melody).cpu().detach().numpy()[0]

        decoded_seq = []
        last_offset = 0
        for timestep in range(prediction.shape[0]):
            d = duration.Duration()
            d.quarterLength = durations_key_dict_rev[np.argmax(prediction[timestep][1])]

            element = element_key_dict_rev[np.argmax(prediction[timestep][0])]
            if element[0] == "[":  # Chord
                element = chord.Chord(eval(element), duration=d)
            else:
                element = note.Note(element, duration=d)
            offset = offsets_key_dict_rev[np.argmax(prediction[timestep][2])]
            element.offset = last_offset + offset
            last_offset = last_offset + offset
            decoded_seq.append(element)

        midi_stream = stream.Stream(decoded_seq)

        numerator = len([name for name in os.listdir(path_file + "/../data/" + mf + '/compositions/')])
        midi_stream.write('midi',
                          fp=(path_file + "/../data/" + mf + '/compositions/output_' + str(numerator) + '.mid'))


# e.g. python predict_Seq2Seq.py -mf "edm"
if __name__ == "__main__":
    main(args_mf, args_sl, args_n)
