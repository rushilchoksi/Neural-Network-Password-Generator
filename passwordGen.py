import os
import sys
import time
import math
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from random import randint
from tabulate import tabulate
from password_strength import PasswordStats
from passwordProfiler import generateWordList

class RNNModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x

class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(values=[-float('inf')]*len(skip_ids), indices=skip_ids, dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        predicted_logits, states = self.model(inputs=input_ids, states=states, return_state=True)
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature
        predicted_logits = predicted_logits + self.prediction_mask

        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        predicted_chars = self.chars_from_ids(predicted_ids)
        return predicted_chars, states

class CustomTraining(RNNModel):
    @tf.function
    def train_step(self, inputs):
        inputs, labels = inputs
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.loss(labels, predictions)
        grads = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return {'loss': loss}

def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

mappingDict = {'a': '@', 'e': '3', 'f': 'ƒ', 'i': '!', 'o': '0', 's': '$', 'y': '¥'}
passwordCount = 5

'''
Provide three options to user for choosing the input
password field, pre-configured, custom URL, wordlist generator,
which would then be passed on to the training process.
'''

menuOptions = ['Pre-Configured Dataset', 'Custom URL', 'Generate Wordlist']
for indexVal, optionVal in enumerate(menuOptions):
    print(f'{indexVal}. {optionVal}')
userChoice = int(input('Enter your choice: '))

print()
while userChoice not in range(0,3):
    userChoice = int(input('Invalid choice, please try again: '))

if userChoice == 0:
    path_to_file = tf.keras.utils.get_file('rockYouWithNoob.txt', 'https://raw.githubusercontent.com/rushilchoksi/Neural-Network-Password-Generator/main/Password%20files/rockYou.txt')
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')[:200000]

    newText = ''
    for i in text:
        # print(i, i in mappingDict.keys())
        if i.lower() in mappingDict.keys():
            if randint(1, 1) == 1:
                newText += mappingDict.get(i.lower())
            else:
                newText += i
        else:
            newText += i

    print(f'Length of text: {len(newText)} characters')
    vocab = sorted(set(newText))
    print(f'{len(vocab)} unique characters')

elif userChoice == 1:
    customURLValue = input('Enter URL to input file: ')
    path_to_file = tf.keras.utils.get_file(customURLValue.split('/')[-1], customURLValue)
    newText = open(path_to_file, 'rb').read().decode(encoding='utf-8')[:200000]
    vocab = sorted(set(newText))
    print(f'{len(vocab)} unique characters')

elif userChoice == 2:
    firstName = input('Enter first name: ')
    lastName = input('Enter last name: ')
    dateOfBirth = input('Enter date of birth (DDMMYYYY): ')
    profiledDataFile = generateWordList(firstName, lastName, dateOfBirth)

    newText = open(profiledDataFile, 'rb').read().decode(encoding='utf-8')
    vocab = sorted(set(newText))
    print(f'{len(vocab)} unique characters')

example_texts = ['abcdefg', 'xyz']

chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')
print(chars)

ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None)

ids = ids_from_chars(chars)
print(ids)

chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

chars = chars_from_ids(ids)
print(chars)

tf.strings.reduce_join(chars, axis=-1).numpy()

all_ids = ids_from_chars(tf.strings.unicode_split(newText, 'UTF-8'))
print(all_ids)
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
for ids in ids_dataset.take(10):
    print(chars_from_ids(ids).numpy().decode('utf-8'))
seq_length = 100

sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

for seq in sequences.take(1):
    print(chars_from_ids(seq))

for seq in sequences.take(5):
    print(text_from_ids(seq).numpy())

split_input_target(list("Tensorflow"))
dataset = sequences.map(split_input_target)
for input_example, target_example in dataset.take(1):
    print("Input :", text_from_ids(input_example).numpy())
    print("Target:", text_from_ids(target_example).numpy())

BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = (dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))

print(dataset)

vocab_size = len(ids_from_chars.get_vocabulary())

embedding_dim = 256
rnn_units = 1024

model = RNNModel(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units)
for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

print(model.summary())

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

print(sampled_indices)

print("Input:\n", text_from_ids(input_example_batch[0]).numpy())
print("\nNext Char Predictions:\n", text_from_ids(sampled_indices).numpy())

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
example_batch_mean_loss = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("Mean loss:        ", example_batch_mean_loss)

tf.exp(example_batch_mean_loss).numpy()
model.compile(optimizer='adam', loss=loss)
checkpoint_dir = './training_checkpoints'

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)
EPOCHS = 5
history = model.fit(dataset, epochs=EPOCHS, callbacks=[tensorboard_callback])

one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

start = time.time()
states = None
next_char = tf.constant(['PASSWORDS:'])
result = [next_char]

for n in range(1000):
    next_char, states = one_step_model.generate_one_step(next_char, states=states)
    result.append(next_char)

result = tf.strings.join(result)
end = time.time()

if userChoice == 0:
    resultantPasswords = result[0].numpy().decode('utf-8').split('\r\n')
elif userChoice == 1:
    resultantPasswords = result[0].numpy().decode('utf-8').split('\r\n')[0].split('\n')
else:
    resultantPasswords = result[0].numpy().decode('utf-8').split('\n')

with open('genPasswords.txt', 'a') as outputFile:
    genPasswords, tempCounter = [], 0
    for i in range(len(resultantPasswords)):
        if len(resultantPasswords[i].replace('PASSWORDS:', '')) > 0:
            genPasswords.append([resultantPasswords[i].replace('PASSWORDS:', ''), math.log(67 ** len(resultantPasswords[i].replace('PASSWORDS:', '')), 2), PasswordStats(resultantPasswords[i].replace('PASSWORDS:', '')).strength()])
            outputFile.write(f"{resultantPasswords[i].replace('PASSWORDS:', '')}\n")
            tempCounter += 1

print('\nRun time:', end - start)

if userChoice == 2:
    print(f'\nTop {passwordCount} passwords generated for {firstName} {lastName}:')
else:
    print(f'\nTop {passwordCount} passwords generated using {menuOptions[userChoice].lower()}:')

dataFrame = pd.DataFrame.from_records(genPasswords)
dataFrame.columns = ['Password', 'Entropy', 'Strength']
dataFrame = dataFrame.sort_values(['Strength', 'Entropy'], ascending = [False, False])
dataFrame.insert(loc = 0, column = 'Rank', value = range(1, tempCounter + 1))

print(dataFrame.head(passwordCount).to_string(index = False))
