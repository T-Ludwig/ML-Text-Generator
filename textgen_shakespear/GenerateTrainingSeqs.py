seq_len = 180
total_num_seq = len(text)//(seq_len+1)

# Create Training Sequences
char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)

for i in char_dataset.take(500):
     print(ind_to_char[i.numpy()])

sequences = char_dataset.batch(seq_len+1, drop_remainder=True)

def create_seq_targets(seq):
    input_txt = seq[:-1]
    target_txt = seq[1:]
    return input_txt, target_txt
    
dataset = sequences.map(create_seq_targets)

for input_txt, target_txt in  dataset.take(1):
    print(input_txt.numpy())
    print(''.join(ind_to_char[input_txt.numpy()]))
    print('\n')
    print(target_txt.numpy())
    # There is an extra whitespace!
    print(''.join(ind_to_char[target_txt.numpy()]))