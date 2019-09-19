from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def lstm_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = LSTM(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def lstm_model_multi(input_dim, activation, output_dim=29, recur_layers=[]):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    layer = input_data
    # Add recurrent layer
    for deepth, hidden_units in enumerate(recur_layers):    
        
        layer = LSTM(hidden_units, activation=activation,
            return_sequences=True, implementation=2, name='rnn'+str(deepth))(layer)
        # TODO: Add batch normalization 
        layer = BatchNormalization()(layer)
        
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(layer)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_rnn_model_dropout(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim*2))(bn_rnn)
    time_dense = TimeDistributed(Dropout(.2))(time_dense)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(time_dense)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model


def cnn_rnn_model_multi(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, hidden_layers, output_dim=29, recur_layers=1):
    """ Build a recurrent + convolutional network for speech 
    """
    
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    layer = input_data
    
    # Add convolutional layer
    layer = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(layer)

    for deepth in range(recur_layers):
        # Add batch normalization
        layer = BatchNormalization(name='bn_'+str(deepth))(layer)
        # Add a recurrent layer
        layer = SimpleRNN(units, activation='relu',
            return_sequences=True, implementation=2, name='rnn_'+str(deepth))(layer)
    
    for deepth, hidden_units in enumerate(hidden_layers):    
        # TODO: Add batch normalization
        layer = BatchNormalization()(layer)
        # TODO: Add a TimeDistributed(Dense(output_dim)) layer
        layer = TimeDistributed(Dense(hidden_units))(layer)
        
        layer = Dropout(.4)(layer)
        layer = Activation('relu', name='relu_hiddenunit_'+str(deepth))(layer)
            
            
    # TODO: Add batch normalization
    layer = BatchNormalization()(layer)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    layer = TimeDistributed(Dense(output_dim))(layer)
        
        
    # Add softmax activation layer
    layer = Activation('softmax', name='softmax')(layer)
    # Specify the model
    model = Model(inputs=input_data, outputs=layer)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_lstm_model_multi(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, hidden_layers, output_dim=29, recur_layers=1):
    """ Build a recurrent + convolutional network for speech 
    """
    
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    layer = input_data
    
    # Add convolutional layer
    layer = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(layer)

    for deepth, rec_units in enumerate(recur_layers):
        # Add batch normalization
        layer = BatchNormalization(name='bn_'+str(deepth))(layer)
        # Add a recurrent layer
        layer = LSTM(rec_units, activation='relu',
            return_sequences=True, implementation=2, name='rnn_'+str(deepth))(layer)
    
    for deepth, hidden_units in enumerate(hidden_layers):    
        # TODO: Add batch normalization
        layer = BatchNormalization()(layer)
        # TODO: Add a TimeDistributed(Dense(output_dim)) layer
        layer = TimeDistributed(Dense(hidden_units))(layer)
        
        layer = Dropout(.4)(layer)
        layer = Activation('relu', name='relu_hiddenunit_'+str(deepth))(layer)
            
            
    # TODO: Add batch normalization
    layer = BatchNormalization()(layer)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    layer = TimeDistributed(Dense(output_dim))(layer)
        
        
    # Add softmax activation layer
    layer = Activation('softmax', name='softmax')(layer)
    # Specify the model
    model = Model(inputs=input_data, outputs=layer)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    layer = input_data
    for deepth in range(recur_layers):
        layer = GRU(units, activation='relu', return_sequences=True, implementation=2, name='rnn'+str(deepth))(layer)
        layer = BatchNormalization()(layer)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(layer)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, activation='relu', return_sequences=True, implementation=2), merge_mode='concat')(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(verbose=True):
    """ Build a deep network for speech 
    """
    input_dim=161
    filters=200
    kernel_size=11
    conv_stride=2
    conv_border_mode='causal'
    units=300
    output_dim=29


    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = Bidirectional(GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn', dropout=.65))(bn_cnn)
    
    # Add batch normalization
    bn_cnn2 = BatchNormalization(name='bn_conv_1d_2')(simp_rnn)
    # Add a recurrent layer
    simp_rnn2 = Bidirectional(GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn_2'))(bn_cnn2)
    
    # Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn2)
    time_dense_hidden = TimeDistributed(Dense(256))(bn_rnn)
    dropout = TimeDistributed(Dropout(.65))(time_dense_hidden)
    
    
    time_dense_hidden2 = TimeDistributed(Dense(256))(dropout)
    dropout2 = TimeDistributed(Dropout(.65))(time_dense_hidden2)
    
    time_dense = TimeDistributed(Dense(output_dim))(dropout2)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: (x + conv_stride - 1) // conv_stride
    
    if verbose:
        print(model.summary())
    return model

