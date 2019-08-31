from src.utils import get_notes, prepare_sequences
#from keras.models import Sequential
#from keras.layers import LSTM, RepeatVector, Dense, TimeDistributed, Dropout, Bidirectional
#from keras.callbacks import ModelCheckpoint, EarlyStopping
#import tensorflow as tf

def train_network(model_name, file_name, ENC_DIM):
    notes = get_notes(file_name)
    print(notes.shape)
    #10 octaves, 12 notes, 8 bit encoding => 120
    n_vocab = 120                                                               


    network_input, network_output = prepare_sequences(notes, n_vocab)
    #print(network_input)
    model = create_network(network_input, n_vocab, ENC_DIM)

    train(model, network_input, network_output, model_name, file_name)
    

#def custom_objective(y_true, y_pred):
#    sum_t_p = y_true - y_pred
#    return tf.norm(sum_t_p)

    
def create_network(network_input, n_vocab, ENC_DIM):
    #model = Sequential()
    #
    #model.add(LSTM(int(0.5*ENC_DIM), input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    #model.add(Dropout(0.3))
    #model.add(LSTM(ENC_DIM))
    #model.add(RepeatVector(network_input.shape[1]))
    #
    #model.add(LSTM(ENC_DIM, return_sequences = True))
    #model.add(Dropout(0.3))
    #model.add(LSTM(int(0.5*ENC_DIM), return_sequences = True))
    #model.add(TimeDistributed(Dense(n_vocab, activation='relu')))
    #
    ##optimizer
    #model.compile(loss=custom_objective, optimizer="adam")
#
    #model.summary()
    #return model
    return None

def train(model, network_input, network_output, model_name, file_name):
    filepath = file_name + "/models/" + model_name + "-{epoch:02d}-{loss:.4f}.hdf5"
    # checkpoint = ModelCheckpoint(
    #     filepath,
    #     monitor='loss',
    #     verbose=0,
    #     save_best_only=True,
    #     mode='min'
    # )
    
    #model.fit(network_input,
    #          network_output,
    #          epochs=500,
    #          batch_size=256,
    #          shuffle=True,
    #          callbacks= [checkpoint, EarlyStopping(monitor='loss', patience=25, mode='auto')])

    
if __name__ == '__main__':
    for enc_dim in [2048]:
        train_network("model_"+str(enc_dim)+"_"+str(0.5), "westworld", enc_dim)