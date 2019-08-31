from keras.models import Sequential
from src.utils import get_shape_for_predict, generate_notes
from keras.layers import LSTM, RepeatVector, Dense, TimeDistributed, Dropout, Bidirectional

def generate(model_name, file_name, ENC_DIM, output_name):
    """ Generate a piano midi file """

    # Get all pitch names
    #pitchnames = sorted(set(item for item in notes))
    # Get all pitch names
    n_vocab = 120

    shape_0, shape_1 = get_shape_for_predict()
    model = create_network(model_name, shape_0, shape_1, n_vocab, ENC_DIM, file_name)
    generate_notes(model, file_name, output_name)    
    

def create_network(model_name, shape_0, shape_1, n_vocab, ENC_DIM, file_name):
    model = Sequential()
    
    #encoder
    model.add(LSTM(ENC_DIM, input_shape=(shape_0, shape_1), return_sequences = True))
    model.add(Dropout(0.3))
    model.add(LSTM(2*ENC_DIM))
    model.add(RepeatVector(shape_0))
    
    #decoder
    model.add(LSTM(ENC_DIM, return_sequences = True))
    model.add(Dropout(0.3))
    model.add(LSTM(ENC_DIM, return_sequences = True))
    model.add(TimeDistributed(Dense(n_vocab, activation='relu')))
    
    model.load_weights(file_name + "/models/" + model_name + ".hdf5")
    
    return model

if __name__ == '__main__':
    for ENC_DIM in [512]:
        for i in range(10):
            generate('model_'+str(ENC_DIM)+"_"+str(0.5), 'westworld', ENC_DIM, 'model_' + str(ENC_DIM) + '_output_'+str(i)+"_"+str(0.5))
