import os
import sys
import glob
import numpy as np
import pickle
from music21 import converter, instrument, note, chord
import argparse
import inspect

path_file = os.path.abspath(
    os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
)  # noqa

sys.path.insert(0, os.path.abspath('..'))  # noqa
from lucky_trainer.misc.custom_dataset_classes import NumpyDataset

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
args = parser.parse_args()
args_mf = args.music_folder
args_sl = args.seq_len


def build_dictionary(file_name):
    notes = []
    chords = []
    durations = []
    offsets = []
    for file in glob.glob(file_name + "/original_data/*.mid"):
        last_offset = 0
        midi = converter.parse(file)
        print("Read in possible notes/chords of %s" % file)

        try:  # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:  # file has notes in a flat structure
            notes_to_parse = midi.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(element.name)
            elif isinstance(element, chord.Chord):
                chords.append(repr(sorted(set([n.name for n in element.notes]))))
            else:
                continue
            durations.append(element.duration.quarterLength)
            element_offset = element.offset
            if last_offset == 0:
                last_offset = element_offset
            offsets.append(element_offset - last_offset)
            last_offset = element_offset

    l = set.union(set(notes), set(chords))
    mapping_notes = dict([(y, x + 1) for x, y in enumerate(l)])
    mapping_durations = dict([(y, x + 1) for x, y in enumerate(sorted(set(durations)))])
    mapping_offsets = dict([(y, x + 1) for x, y in enumerate(sorted(set(offsets)))])

    return mapping_notes, mapping_durations, mapping_offsets


def encode_folder(music_folder):
    mapping_notes, mapping_durations, mapping_offsets = build_dictionary(music_folder)

    # save dictionary
    f = open(music_folder + "/element_key_dict.pkl", "wb")
    pickle.dump(mapping_notes, f)
    f.close()
    f = open(music_folder + "/durations_key_dict.pkl", "wb")
    pickle.dump(mapping_durations, f)
    f.close()
    f = open(music_folder + "/offsets_key_dict.pkl", "wb")
    pickle.dump(mapping_offsets, f)
    f.close()

    print("Read in completed. \n")

    for file in glob.glob(music_folder + "/original_data/*.mid"):
        midi = converter.parse(file)
        print("Parsing %s" % file)

        try:  # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:  # file has notes in a flat structure
            notes_to_parse = midi.notes

        midi_encoding = []
        last_offset = 0
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                element_key = element.name
            elif isinstance(element, chord.Chord):
                element_key = repr(sorted(set([n.name for n in element.notes])))
            else:
                continue
            element_offset = element.offset
            if last_offset == 0:
                last_offset = element_offset
            element_duration = element.duration
            max_key = max(len(mapping_notes), len(mapping_offsets), len(mapping_durations))
            np_arr = np.stack(np.array([np.zeros(max_key, dtype=np.float32),
                                        np.zeros(max_key, dtype=np.float32),
                                        np.zeros(max_key, dtype=np.float32)]))
            np_arr[0][mapping_notes[element_key] - 1] = 1
            np_arr[1][mapping_offsets[element_offset - last_offset] - 1] = 1
            np_arr[2][mapping_durations[element_duration.quarterLength] - 1] = 1
            midi_encoding.append(np_arr)

            last_offset = element_offset

        file_encoding = np.array(midi_encoding)
        f = open(music_folder + "/encodings/" + os.path.basename(file)[:-4] + "_encoding.pkl", "wb")
        pickle.dump(file_encoding, f)
        f.close()


def get_dl_data(music_folder, seq_len=25):
    x_data = []
    y_data = []
    for file in glob.glob(music_folder + "/encodings/*_encoding.pkl"):
        print("Loading %s" % file)
        with open(file, 'rb') as f:
            data = pickle.load(f)
        all_sequences_length = len(data) - 2 * seq_len
        if all_sequences_length < 0:
            print("File %s is too small. Skipping." % file)
            continue
        for i in range(all_sequences_length):
            x_data.append(data[i:i + seq_len])
            y_data.append(data[i + seq_len:i + 2 * seq_len])

    print("Extraction completed.\n")
    return np.array(x_data), np.array(y_data)


def get_val_split(inputs, targets, split=0.1):
    dataset_length = len(inputs)
    indices = list(range(dataset_length))
    np.random.shuffle(indices)
    if split <= 1:
        mapping_val = indices[:int(split * dataset_length)]
        mapping_train = indices[int(split * dataset_length):]
    else:
        mapping_val = indices[:split]
        mapping_train = indices[split:]

    return (inputs[mapping_train], targets[mapping_train],
            inputs[mapping_val], targets[mapping_val])


def main(mf, sl):
    os.makedirs(mf + '/input_data/', exist_ok=True)
    os.makedirs(mf + '/encodings/', exist_ok=True)
    os.makedirs(mf + '/compositions/', exist_ok=True)

    # Encode dataset
    encode_folder(mf)
    x, y = get_dl_data(mf, sl)
    train_in, train_out, val_in, val_out = get_val_split(x, y, split=0.1)

    # Create PyTorch Dataset
    train = NumpyDataset(train_in, train_out)
    val = NumpyDataset(val_in, val_out)

    # Save the Dataset
    outfile = open(mf + "/input_data/train", 'wb')
    pickle.dump(train, outfile)
    outfile.close()
    outfile = open(mf + "/input_data/val", 'wb')
    pickle.dump(val, outfile)
    outfile.close()
    print(mf + " data saved.")


# e.g. python preprocessing_data.py -mf "edm"
if __name__ == "__main__":
    main(args_mf, args_sl)
