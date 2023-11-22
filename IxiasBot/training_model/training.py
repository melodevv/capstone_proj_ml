import string
import random
import json
import pickle
import os
import nltk
import numpy as np
import tensorflow as tf


from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from matplotlib import pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# nltk.download("wordnet")
# nltk.download("punkt")


# =====================================================
#                   Preprocessing
# =====================================================

# Get the dataset
data_file = open("./IxiasBot/intents.json").read()

# Variables
lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_words = string.punctuation
intents = json.loads(data_file)

# Tokenization
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent["tag"]))

        # add the tag in our classes list
        if intent["tag"] not in classes:
            classes.append(intent["tag"])


# Lemmatization
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

# Extract the words and classes to a pkl file
pickle.dump(words, open("./IxiasBot/rendered/words.pkl", "wb"))
pickle.dump(classes, open("./IxiasBot/rendered/classes.pkl", "wb"))


# =====================================================
#                   Training Model
# =====================================================

training = []  # For training data

output_empty = [0] * len(classes)  # This is for our output


# Training set, bag of wards for each sentence
for doc in documents:
    bag_of_words = []
    pattern_words = doc[0]
    # lemmatize and convert pattern_words to lower case
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    # create bag of words array,if word match found in current pattern then put 1
    # otherwise 0.[row * colm(263)]
    for w in words:
        bag_of_words.append(1) if w in pattern_words else bag_of_words.append(0)

        # in output array 0 value for each tag ang 1 value for matched tag.[row *
        # colm(8)]
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

    training.append([bag_of_words, output_row])

random.shuffle(training)
training = np.array(training, dtype=list)

# create the traing and test.
train_x = list(training[:, 0])  # patterns(words)
train_y = list(training[:, 1])  # intents(tag)


# =====================================================
#                   Build the model
# =====================================================

model = Sequential()
model.add(
    Dense(128, input_shape=(len(train_x[0]),), activation="relu")
)  # Layer 1 - 128 neurons
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))  # Layer 2 - 64 neurons
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient
# gives good results for this model
# sgd = SGD(learning_rate=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(train_x, train_y, epochs=400, batch_size=10, verbose=1)
model.save("./IxiasBot/rendered/ixias_model.h5", history)


# Display the model performance:
plt.plot(history.history["accuracy"])
plt.plot(history.history["loss"])
plt.title("Model Accurarcy Vs Loss[Bsize 10]")
plt.ylabel("Accurracy and Loss")
plt.xlabel("Epoch")
plt.legend(["Accuracy", "Loss"])
plt.savefig("./IxiasBot/training_model/test_stats.png")
