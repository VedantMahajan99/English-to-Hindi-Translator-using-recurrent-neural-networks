from preprocessing import num_encoder_tokens, num_decoder_tokens, decoder_target_data, encoder_input_data, decoder_input_data, decoder_target_data

from tensorflow import keras
# Add Dense to the imported layers
# Our encoder requires two layer types from Keras:
#
# An input layer, which defines a matrix to hold all the one-hot vectors that we’ll feed to the model.
# An LSTM layer, with some output dimensionality.
from keras.layers import Input, LSTM, Dense
from keras.models import Model

# UNCOMMENT THE TWO LINES BELOW IF YOU ARE GETTING ERRORS ON A MAC
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Choose a dimensionality
latent_dim = 256

# Choose a batch size
# and a larger number of epochs:
batch_size = 50  # This determines how many sentences are used at a time for training.
epochs = 100      # One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE

# Encoder training setup
encoder_inputs = Input(shape=(None, num_encoder_tokens))
# For the LSTM layer, we need to select the dimensionality (the size of the LSTM’s hidden states, which helps determine how closely the model molds itself to the training data )
encoder_lstm = LSTM(latent_dim, return_state=True)
#  the only thing we want from the encoder is its final states. We can get these by linking our LSTM layer with our input layer
encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
encoder_states = [state_hidden, state_cell]

# Decoder training setup:
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
# with our decoder, we pass in the state data from the encoder, along with the decoder inputs.
decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(decoder_inputs, initial_state=encoder_states)
# need to run the output through a final activation layer, using the Softmax function, that will give us the probability distribution — where all probabilities sum to one — for each token.
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Building the training model: we define the seq2seq model using the Model() function we imported from Keras. To make it a seq2seq model, we feed it the encoder and decoder inputs, as well as the decoder output:
training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

print("Model summary:\n")
training_model.summary()
print("\n\n")

# Compile the model:
# An optimizer (we’re using RMSprop, which is a fancy version of the widely-used gradient descent) to help minimize our error rate (how bad the model is at guessing the true next word given the previous words in a sentence).
# A loss function (we’re using the logarithm-based cross-entropy function) to determine the error rate.
training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# print("Training the model:\n")
# Train the model: Next we need to fit the compiled model. To do this, we give the .fit() method the encoder and decoder input data (what we pass into the model), the decoder target data (what we expect the model to return given the data we passed in), and some numbers we can adjust as needed:
training_model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size = batch_size, epochs = epochs, validation_split = 0.2)

training_model.save('training_model.h5')