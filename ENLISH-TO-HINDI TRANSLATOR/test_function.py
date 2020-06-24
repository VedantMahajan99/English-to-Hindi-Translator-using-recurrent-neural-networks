from preprocessing import input_features_dict, target_features_dict, reverse_input_features_dict, reverse_target_features_dict, max_decoder_seq_length, input_docs, target_docs, input_tokens, target_tokens
from training_model import encoder_inputs, decoder_inputs, encoder_states, decoder_lstm, decoder_dense, encoder_input_data, num_decoder_tokens, latent_dim

from tensorflow import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model, load_model
import numpy as np

training_model = load_model('training_model.h5')
encoder_inputs = training_model.input[0]
encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output
encoder_states = [state_h_enc, state_c_enc]

# we need a model that will decode step-by-step instead of using teacher forcing. To do this, we need a seq2seq network in individual pieces.
# we’ll build an encoder model with our encoder inputs and the placeholders for the encoder’s output states:
encoder_model = Model(encoder_inputs, encoder_states)

#  we need placeholders for the decoder’s input states, which we can build as input layers and store together
decoder_state_input_hidden = Input(shape=(latent_dim,))
decoder_state_input_cell = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]

# Using the decoder LSTM and decoder dense layer (with the activation function) that we trained earlier, we’ll create new decoder states and outputs:
decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_hidden, state_cell]
decoder_outputs = decoder_dense(decoder_outputs)

#  we can set up the decoder model. This is where we bring together:
#
# the decoder inputs (the decoder input layer)
# the decoder input states (the final states from the encoder)
# the decoder outputs (the NumPy matrix we get from the final output layer of the decoder)
# the decoder output states (the memory throughout the network from one word to the next)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

def decode_sequence(test_input):
  # Encode the input as state vectors.
  # Inside the test function, we’ll run our new English sentence through the encoder model. The .predict() method takes in new input (as a NumPy matrix) and gives us output states that we can pass on to the decoder:
  states_value = encoder_model.predict(test_input)

  # Generate empty target sequence of length 1.
  target_seq = np.zeros((1, 1, num_decoder_tokens))
  # Populate the first token of target sequence with the start token.
  target_seq[0, 0, target_features_dict['<START>']] = 1.

  # Sampling loop for a batch of sequences
  # (to simplify, here we assume a batch of size 1).
  # we’ll decode the sentence word by word using the output state that we retrieved from the encoder (which becomes our decoder’s initial hidden state). We’ll also update the decoder hidden state after each word so that we use previously decoded words to help decode new ones.
  decoded_sentence = ''

  stop_condition = False
  while not stop_condition:
    # Run the decoder model to get possible
    # output tokens (with probabilities) & states
    output_tokens, hidden_state, cell_state = decoder_model.predict(
      [target_seq] + states_value)

    # Choose token with highest probability
    sampled_token_index = np.argmax(output_tokens[0, -1, :])
    sampled_token = reverse_target_features_dict[sampled_token_index]
    decoded_sentence += " " + sampled_token

    # Exit condition: either hit max length
    # or find stop token.
    if (sampled_token == '<END>' or len(decoded_sentence) > max_decoder_seq_length):
      stop_condition = True

    # Update the target sequence (of length 1).
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, sampled_token_index] = 1.

    # Update states
    states_value = [hidden_state, cell_state]

  return decoded_sentence


# CHANGE RANGE (NUMBER OF TEST SENTENCES TO TRANSLATE) AS YOU PLEASE
for seq_index in range(50):
  test_input = encoder_input_data[seq_index: seq_index + 1]
  decoded_sentence = decode_sequence(test_input)
  print('-')
  print('Input sentence:', input_docs[seq_index])
  print('Decoded sentence:', decoded_sentence)