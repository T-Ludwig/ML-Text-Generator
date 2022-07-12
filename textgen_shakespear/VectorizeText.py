vocab = sorted(set(text))
vocab[:10]

char_to_ind = {char:i for i, char in enumerate(vocab)}
ind_to_char = np.array(vocab)
encoded_text = np.array([char_to_ind[c] for c in text])
