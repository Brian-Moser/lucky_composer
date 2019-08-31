import glob
import numpy as np
from music21 import converter, instrument, note, chord, stream, duration

sequence_length = 25


def get_notes(file_name):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []
    for file in glob.glob(file_name + "/training/*.mid"):
        midi = converter.parse(file)
        print("Parsing %s" % file)

        try:  # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:  # file has notes in a flat structure
            notes_to_parse = midi.notes
        max_offset = 0
        for element in notes_to_parse:
            if element.offset > max_offset:
                max_offset = element.offset
        start_offset = len(notes)
        for _ in range(int(4 * max_offset) + 1):
            notes.append([])
            for _ in range(10 * 12):
                notes[-1].append(0.0)
        min_octave, max_octave = parse_to_notes(notes, notes_to_parse,
                                                start_offset, 0)

        if min_octave > 0:
            for _ in range(int(4 * max_offset) + 1):
                notes.append([])
                for _ in range(10 * 12):
                    notes[-1].append(0.0)
            parse_to_notes(notes, notes_to_parse, start_offset, -1)
        if min_octave < 9:
            for _ in range(int(4 * max_offset) + 1):
                notes.append([])
                for _ in range(10 * 12):
                    notes[-1].append(0.0)
            parse_to_notes(notes, notes_to_parse, start_offset, 1)
    notes = np.array(notes).astype('float32')
    return notes


def parse_to_notes(notes, notes_to_parse, start_offset, octave_offset):
    noteNamesToInt = {"C": 0, "C#": 1, "D": 2, "E-": 3, "E": 4, "F": 5, "F#": 6,
                      "G": 7, "G#": 8, "A": 9, "B-": 10, "B": 11}
    min_octave = 10
    max_octave = 0
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            name = element.nameWithOctave
            octave = int(name[-1]) + octave_offset
            if octave > max_octave:
                max_octave = octave
            if octave < min_octave:
                min_octave = octave
            note_index = noteNamesToInt[name[:len(name) - 1]]
            notes[int(start_offset + 4 * element.offset)][
                12 * octave + note_index] = float(
                element.duration.quarterLength)
            # print(element.duration)
        elif isinstance(element, chord.Chord):
            for i in range(len(element)):
                name = element[i].nameWithOctave
                octave = int(name[-1]) + octave_offset
                if octave > max_octave:
                    max_octave = octave
                if octave < min_octave:
                    min_octave = octave
                note_index = noteNamesToInt[name[:len(name) - 1]]
                notes[int(start_offset + 4 * element.offset)][
                    12 * octave + note_index] = float(
                    element.duration.quarterLength)
    return min_octave, max_octave


def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """

    network_input = []
    network_output = []
    for i in range(0, len(notes) - 2 * sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length: i + 2 * sequence_length]
        network_input.append(sequence_in)
        network_output.append(sequence_out)

    return (np.array(network_input), np.array(network_output))


def get_shape_for_predict():
    return sequence_length, 120


def generate_notes(model, file_name, output_name):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    noteList = ["C", "C#", "D", "E-", "E", "F", "F#", "G", "G#", "A", "B-", "B"]

    # int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    pattern = np.random.randint(5, size=(120, sequence_length))
    # pattern = numpy.reshape(pattern, (1, sequence_length, 120))
    pattern = pattern
    prediction_output = []
    output_notes = []

    # generate 500 notes
    for prediction_index in range(25):
        prediction_output.append([])
        prediction_input = np.reshape(pattern, (1, sequence_length, 120))

        prediction = model.predict(prediction_input, verbose=0)
        for i in range(sequence_length):
            prediction_output.append([])
            for j in range(len(prediction[0][0])):
                prediction[0][i][j] = round(
                    0.25 * round((prediction[0][i][j] / 4) / 0.25), 2)
                if prediction[0][i][j] != 0.0:
                    new_note = note.Note(noteList[j % 12] + str(int(j / 12)),
                                         type=
                                         duration.quarterLengthToClosestType(
                                             prediction[0][i][j])[0])
                    new_note.offset = 0.25 * (
                                i + sequence_length * prediction_index)
                    new_note.storedInstrument = instrument.Piano()
                    prediction_output[-1].append(prediction[0][i][j])
                    output_notes.append(new_note)
                else:
                    prediction_output[-1].append(0.0)

        pattern = prediction_output[-sequence_length:]

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi',
                      fp=(file_name + '/output/' + output_name + '.mid'))
