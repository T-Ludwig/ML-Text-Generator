from tensorflow.keras.models import load_model
model.save('shakespeare_gen.h5') 

#Currently our model only expects 128 sequences at a time. We can create a new model that only expects a batch_size=1. We can create a new model with this batch size, then load our saved models weights.
#Then call .build() on the mode

model = create_model(vocab_size, embed_dim, rnn_neurons, batch_size=1)
model.load_weights('shakespeare_gen.h5')
model.build(tf.TensorShape([1, None]))


def generate_text(model, start_seed,gen_size=100,temp=1.0):
  '''
  model: Trained Model to Generate Text
  start_seed: Intial Seed text in string form
  gen_size: Number of characters to generate
  Basic idea behind this function is to take in some seed text, format it so
  that it is in the correct shape for our network, then loop the sequence as
  we keep adding our own predicted characters. Similar to our work in the RNN
  time series problems.
  '''
  # Number of characters to generate
  num_generate = gen_size
  # Vecotrizing starting seed text
  input_eval = [char_to_ind[s] for s in start_seed]
  # Expand to match batch format shape
  input_eval = tf.expand_dims(input_eval, 0)
  # Empty list to hold resulting generated text
  text_generated = []
  # Temperature effects randomness in our resulting text
  # The term is derived from entropy/thermodynamics.
  # The temperature is used to effect probability of next characters.
  # Higher probability == lesss surprising/ more expected
  # Lower temperature == more surprising / less expected
  temperature = temp
  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      # Generate Predictions
      predictions = model(input_eval)
      # Remove the batch shape dimension
      predictions = tf.squeeze(predictions, 0)
      # Use a cateogircal disitribution to select the next character
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
      # Pass the predicted charracter for the next input
      input_eval = tf.expand_dims([predicted_id], 0)
      # Transform back to character letter
      text_generated.append(ind_to_char[predicted_id])
  return (start_seed + ''.join(text_generated))


print(generate_text(model,"thou shall not pass",gen_size=1000))