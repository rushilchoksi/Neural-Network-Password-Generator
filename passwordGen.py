import os
import sys
import time
import math
import datetime
import threading
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from random import randint
from tabulate import tabulate
from breachAPI import checkPassword
from password_strength import PasswordStats
from passwordProfiler import generateWordList

class RNNModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputData, statesData=None, return_state=False, training=False):
        x = inputData
        x = self.embedding(x, training=training)
        if statesData is None:
            statesData = self.gru.get_initial_state(x)

        x, statesData = self.gru(x, initial_state=statesData, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, statesData
        else:
            return x

class OneStep(tf.keras.Model):
    def __init__(self, model, charsFromIDs, idsFromChars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.charsFromIDs = charsFromIDs
        self.idsFromChars = idsFromChars

        skipIDs = self.idsFromChars(['[UNK]'])[:, None]
        sparseMask = tf.SparseTensor(values=[-float('inf')]*len(skipIDs), indices=skipIDs, dense_shape=[len(idsFromChars.get_vocabulary())])
        self.predictionMask = tf.sparse.to_dense(sparseMask)

    @tf.function
    def genOneStep(self, inputData, statesData=None):
        inputChars = tf.strings.unicode_split(inputData, 'UTF-8')
        inputIDs = self.idsFromChars(inputChars).to_tensor()

        predictedLogits, statesData = self.model(inputData=inputIDs, statesData=statesData, return_state=True)
        predictedLogits = predictedLogits[:, -1, :]
        predictedLogits = predictedLogits/self.temperature
        predictedLogits = predictedLogits + self.predictionMask

        predictedIDs = tf.random.categorical(predictedLogits, num_samples=1)
        predictedIDs = tf.squeeze(predictedIDs, axis=-1)

        predictedChars = self.charsFromIDs(predictedIDs)
        return predictedChars, statesData

class CustomTraining(RNNModel):
    @tf.function
    def trainStep(self, inputData):
        inputData, labelValue = inputData
        with tf.GradientTape() as gradientTape:
            predictionsValue = self(inputData, training=True)
            lossValue = self.loss(labelValue, predictionsValue)

        gradientValue = gradientTape.gradient(lossValue, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradientValue, model.trainable_variables))

        return {'loss': lossValue}

def textFromIDs(idsData):
    return tf.strings.reduce_join(charsFromIDs(idsData), axis=-1)

def splitInputTarget(sequenceData):
    inputText = sequenceData[:-1]
    targetText = sequenceData[1:]
    return inputText, targetText

def showAnimation():
    for tempSymbol in itertools.cycle(['.  \r', '.. \r', '...\r']):
        if taskCompleted:
            break
        sys.stdout.write('\rPlease wait while data is being processed ' + tempSymbol)
        sys.stdout.flush()
        time.sleep(0.5)
    sys.stdout.write('\rData processing completed!\n')

if __name__ == "__main__":
    passwordCount, mappingDict = 5, {'a': '@', 'e': '3', 'f': 'ƒ', 'i': '!', 'o': '0', 's': '$', 'y': '¥'}

    '''
    Provide three options to user for choosing the input
    password field, pre-configured, custom URL, wordlist generator,
    which would then be passed on to the training process.
    '''

    menuOptions = ['Pre-Configured Dataset', 'Custom URL', 'Generate Wordlist']
    for indexVal, optionVal in enumerate(menuOptions):
        print(f'{indexVal}. {optionVal}')
    userChoice = int(input('\nEnter your choice: '))

    print()
    while userChoice not in range(0,3):
        userChoice = int(input('Invalid choice, please try again: '))

    if userChoice == 0:
        filePath = tf.keras.utils.get_file('rockYouWithNoob.txt', 'https://raw.githubusercontent.com/rushilchoksi/Neural-Network-Password-Generator/main/Password%20files/rockYou.txt')
        fileTextContent = open(filePath, 'rb').read().decode(encoding='utf-8')[:200000]

        newText = ''
        for i in fileTextContent:
            if i.lower() in mappingDict.keys():
                if randint(1, 1) == 1:
                    newText += mappingDict.get(i.lower())
                else:
                    newText += i
            else:
                newText += i

        print(f'Length of fileTextContent: {len(newText)} characters')
        trainingVocab = sorted(set(newText))
        print(f'{len(trainingVocab)} unique characters')

    elif userChoice == 1:
        customURLValue = input('Enter URL to input file: ')
        filePath = tf.keras.utils.get_file(customURLValue.split('/')[-1], customURLValue)
        newText = open(filePath, 'rb').read().decode(encoding='utf-8')[:200000]
        trainingVocab = sorted(set(newText))
        print(f'{len(trainingVocab)} unique characters')

    elif userChoice == 2:
        firstName = input('Enter first name: ')
        lastName = input('Enter last name: ')
        dateOfBirth = input('Enter date of birth (DDMMYYYY): ')
        profiledDataFile = generateWordList(firstName, lastName, dateOfBirth)

        newText = open(profiledDataFile, 'rb').read().decode(encoding='utf-8')
        trainingVocab = sorted(set(newText))
        print(f'{len(trainingVocab)} unique characters')

    exampleText = ['abcdefg', 'xyz']
    characterData = tf.strings.unicode_split(exampleText, input_encoding='UTF-8')

    idsFromChars = tf.keras.layers.StringLookup(vocabulary=list(trainingVocab), mask_token=None)
    idsData = idsFromChars(characterData)

    charsFromIDs = tf.keras.layers.StringLookup(vocabulary=idsFromChars.get_vocabulary(), invert=True, mask_token=None)
    characterData = charsFromIDs(idsData)

    tf.strings.reduce_join(characterData, axis=-1).numpy()
    allIDsData = idsFromChars(tf.strings.unicode_split(newText, 'UTF-8'))
    idsDataset = tf.data.Dataset.from_tensor_slices(allIDsData)
    for idsData in idsDataset.take(10):
        print(charsFromIDs(idsData).numpy().decode('utf-8'))

    sequenceLength = 100
    sequencesData = idsDataset.batch(sequenceLength+1, drop_remainder=True)

    for tempSeq in sequencesData.take(1):
        print(charsFromIDs(tempSeq))

    for tempSeq in sequencesData.take(5):
        print(textFromIDs(tempSeq).numpy())

    splitInputTarget(list('Tensorflow'))
    inputDataset = sequencesData.map(splitInputTarget)
    for tempInputExample, tempTargetExample in inputDataset.take(1):
        print(f'Input: {textFromIDs(tempInputExample).numpy()}')
        print(f'Target: {textFromIDs(tempTargetExample).numpy()}')

    # Model Variables
    BATCH_SIZE = 64
    BUFFER_SIZE = 10000
    EMBEDDING_DIMS = 256
    RNN_UNITS = 1024
    EPOCHS = 1

    inputDataset = (inputDataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))
    vocabSize = len(idsFromChars.get_vocabulary())

    rnnModel = RNNModel(vocab_size = vocabSize, embedding_dim = EMBEDDING_DIMS, rnn_units = RNN_UNITS)
    for inputExampleBatch, targetExampleBatch in inputDataset.take(1):
        example_batch_predictions = rnnModel(inputExampleBatch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocabSize)")

    print(rnnModel.summary())

    sampledIndices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampledIndices = tf.squeeze(sampledIndices, axis=-1).numpy()
    print(sampledIndices)

    print(f'Input: \n{textFromIDs(inputExampleBatch[0]).numpy()}')
    print(f'\nNext Char Predictions: \n{textFromIDs(sampledIndices).numpy()}')

    lossValue = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    exampleBatchMeanLoss = lossValue(targetExampleBatch, example_batch_predictions)
    print(f'\nPrediction shape: {example_batch_predictions.shape} # (batch_size, sequence_length, vocabSize)')
    print(f'Mean loss: {exampleBatchMeanLoss}')

    tf.exp(exampleBatchMeanLoss).numpy()
    rnnModel.compile(optimizer='adam', loss=lossValue)
    checkpointDir = './training_checkpoints'

    modelLogDir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir = modelLogDir, histogram_freq = 1)
    checkpointPrefix = os.path.join(checkpointDir, "ckpt_{epoch}")

    checkpointCallback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointPrefix, save_weights_only=True)
    fitModel = rnnModel.fit(inputDataset, epochs=EPOCHS, callbacks=[tensorboardCallback])
    oneStepModel = OneStep(rnnModel, charsFromIDs, idsFromChars)

    startTime, statesData, nextChar = time.time(), None, tf.constant(['PASSWORDS:'])
    resultValue = [nextChar]

    for _ in range(1000):
        nextChar, statesData = oneStepModel.genOneStep(nextChar, statesData = statesData)
        resultValue.append(nextChar)

    resultValue = tf.strings.join(resultValue)
    endTime = time.time()

    if userChoice == 0:
        resultantPasswords = resultValue[0].numpy().decode('utf-8').split('\r\n')
    elif userChoice == 1:
        resultantPasswords = resultValue[0].numpy().decode('utf-8').split('\r\n')[0].split('\n')
    else:
        resultantPasswords = resultValue[0].numpy().decode('utf-8').split('\n')

    taskCompleted = False
    threadVar = threading.Thread(target = showAnimation)
    threadVar.start()
    with open('genPasswords.txt', 'a') as outputFile:
        genPasswords, tempCounter = [], 0
        for i in range(len(resultantPasswords)):
            if len(resultantPasswords[i].replace('PASSWORDS:', '')) > 0:
                genPasswords.append([resultantPasswords[i].replace('PASSWORDS:', ''), math.log(67 ** len(resultantPasswords[i].replace('PASSWORDS:', '')), 2), PasswordStats(resultantPasswords[i].replace('PASSWORDS:', '')).strength(), checkPassword(resultantPasswords[i].replace('PASSWORDS:', ''), False)])
                outputFile.write(f"{resultantPasswords[i].replace('PASSWORDS:', '')}\n")
                tempCounter += 1

    taskCompleted = True
    print(f'\nRun time: {endTime - startTime}')
    tryAgain = 'Y'
    while tryAgain == 'Y':
        passwordLength = int(input('\n\nEnter length of your desired password: '))

        if userChoice == 2:
            print(f'\nTop {passwordCount} passwords generated for {firstName} {lastName} of {passwordLength} characters:')
        else:
            print(f'\nTop {passwordCount} passwords generated using {menuOptions[userChoice].lower()} of {passwordLength} characters:')

        dataFrame = pd.DataFrame.from_records(genPasswords)
        dataFrame.columns = ['Password', 'Entropy', 'Strength', 'Breach Status']
        dataFrame = dataFrame.sort_values(['Strength', 'Entropy'], ascending = [False, False])
        dataFrame.insert(loc = 0, column = 'Rank', value = range(1, tempCounter + 1))
        filterCondition = (dataFrame['Password'].str.len() == passwordLength)
        filteredDF = dataFrame.loc[filterCondition]
        if filteredDF.empty == True:
            print(f'No passwords with a length of {passwordLength} characters were generated. However, these are the top {passwordCount} passwords:')
            print(dataFrame.head(passwordCount).to_string(index = False))
        else:
            print(filteredDF.head(passwordCount).to_string(index = False))
        tryAgain = input('\nDo you wish to change the desired password length [Y/N]: ').upper()
    # print(dataFrame.head(passwordCount).to_string(index = False))
