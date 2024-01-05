# Imports
# Below is a list of all the packages used in the project
import nltk
from nltk.stem import WordNetLemmatizer
import string
import re
import json
import pickle
import warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import (
    Input,
    Embedding,
    Dense,
    Activation,
    Dropout,
    GlobalAveragePooling1D,
)
from keras.optimizers import SGD
from keras.utils import plot_model
from sklearn.preprocessing import LabelEncoder
import random

from matplotlib import pyplot as plt

## Data Preparation
# The dataset is in json format and so to make it more visually appealing
# it's better to convert it to a Pandas dataframe format. This will also
# later on make it easy for us to do some data analysis.


# read the dataset
with open("./IxiasBot/dataset/intents.json", "r") as f:
    data = json.load(f)

# Convert dataset into dataframe
df = pd.DataFrame(data["intents"])


# On the datafram each tag contains a list of question(pattern) and
# answers(responses). The below code breaks up the patterns based on
# their tag and responses and reconvert them into a DataFrame"""
dic = {"tag": [], "patterns": [], "responses": []}
count = 1

for i in range(len(df)):
    patterns = df[df.index == i]["patterns"].values[0]
    responses = df[df.index == i]["responses"].values[0]
    tag = df[df.index == i]["tag"].values[0]

    for j in range(len(patterns)):
        dic["tag"].append(tag)
        dic["patterns"].append(patterns[j])
        dic["responses"].append(responses)

df = pd.DataFrame.from_dict(dic)


# preview of tag (label) in the dataset
print(df["tag"].unique())


# Exploratory Data Analysis

### Distribution of Intents
# Below we analyze the distribution of intents in the dataset
# and visualize this data using bar plot from the Matplotlib
# library. The x-axis represents the intents, and y-axis represent
# the count of patterns or responses associated with each intent
intents_count = df["tag"].value_counts()

fig, ax = plt.subplots(figsize=(15, 9))
ax.bar(intents_count.index, intents_count.values, color="skyblue")
plt.xticks(rotation=-90)
plt.xlabel("Intents")
plt.ylabel("Count")
plt.title("Distribution of Intents")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color("#D8D8D8")
ax.tick_params(bottom=False, left=False)
ax.set_axisbelow(True)
ax.xaxis.grid(False)
ax.yaxis.grid(True, color="#D8D8D8")

plt.savefig("/IxiasBot/rendered/img/distribution_analysis.png")
plt.show()


### Pattern and Response Analysis
# Below we explore the patterns and responses associated with
# each intent then we make use of Matplotlib library to visualize
# the information. X-axis represents the avearge count of patterns
# and responses and the y-axis represents the intents.

# get the str length of each pattern under patterns
df["pattern_count"] = df["patterns"].apply(lambda x: len(x))

# get the list length of each reponse list under responses
df["response_count"] = df["responses"].apply(lambda x: len(x))

# get the average pattern_count and response_count
avg_pattern_count = df.groupby("tag")["pattern_count"].mean()
avg_response_count = df.groupby("tag")["response_count"].mean()

# plot chart
X_axis = np.arange(len(avg_pattern_count.index))
fig, ax = plt.subplots(figsize=(10, 15))

ax.barh(
    X_axis - 0.4,
    avg_pattern_count.values,
    align="edge",
    height=0.4,
    label="Average Pattern Count",
    color="skyblue",
)
ax.barh(
    X_axis,
    avg_response_count.values,
    align="edge",
    height=0.4,
    label="Average Response Count",
    color="#ff8178",
)
ax.invert_yaxis()

# Style chart
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_color("#D8D8D8")
ax.tick_params(bottom=False, left=False)
ax.set_axisbelow(True)
ax.yaxis.grid(False)
ax.xaxis.grid(True, color="#D8D8D8")

plt.yticks(X_axis, avg_pattern_count.index)  # type: ignore
plt.ylabel(ylabel="Intents")
plt.xlabel("Average Count")
ax.legend()

plt.savefig("/IxiasBot/rendered/img/ptrn_rspn_analysis.png")
plt.show()

"""# Data Preprocessing

Now will be apply some text preprocessing techniques such as cleaning/ normalizing, lowering, removing punctuations and then tokenizing the patterns.


"""

lemmatizer = WordNetLemmatizer()
words = []
classes = []
documents = []
ignore_words = string.punctuation

nltk.download("wordnet")
nltk.download("punkt")


# This function converts short-hand texts to long format
def clean_text(txt):
    txt = txt.lower()
    txt = re.sub(r"i'm", "i am", txt)
    txt = re.sub(r"he's", "he is", txt)
    txt = re.sub(r"she's", "she is", txt)
    txt = re.sub(r"that's", "that is", txt)
    txt = re.sub(r"what's", "what is", txt)
    txt = re.sub(r"where's", "where is", txt)
    txt = re.sub(r"\'ll", " will", txt)
    txt = re.sub(r"\'ve", " have", txt)
    txt = re.sub(r"\'re", " are", txt)
    txt = re.sub(r"\'d", " would", txt)
    txt = re.sub(r"won't", " will not", txt)
    txt = re.sub(r"can't", " can not", txt)
    txt = re.sub(r"[^\w\s]", "", txt)
    return txt


# the the pattern text and normalize it
for i in range(len(df)):
    pattern = clean_text(df["patterns"][i])
    df["patterns"][i] = pattern


# Tokenization
for i, pattern in df["patterns"].items():
    w = nltk.word_tokenize(pattern)
    words.extend(w)
    documents.append((w, df["tag"][i]))  # type: ignore

    # add the tag in our classes list
    if df["tag"][i] not in classes:  # type: ignore
        classes.append(df["tag"][i])  # type: ignore


# Lemmatization
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

# documents = combination between patterns and intents
print(len(documents), "documents\n", documents, "\n")

# classes = tags
print(len(classes), "classes\n", classes, "\n")

# words = all words, vocabulary
print(len(words), "unique lemmatized words\n", words, "\n")

# Extract the words and classes to a pkl file
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# Build and Train Model
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

model = Sequential()
model.add(
    Dense(128, input_shape=(len(train_x[0]),), activation="relu")
)  # Layer 1 - 128 neurons
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))  # Layer 2 - 64 neurons
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

# Compile model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# print summary of the model
model.summary()
plot_model(model, show_shapes=True)

history = model.fit(train_x, train_y, epochs=1500, batch_size=5, verbose=1)  # type: ignore
model.save("/IxiasBot/rendered/ixias_model.h5", history)

plt.plot(history.history["accuracy"])
plt.plot(history.history["loss"])
plt.title("Model Accurarcy Vs Loss")
plt.ylabel("Accurracy and Loss")
plt.xlabel("Epoch")
plt.legend(["Accuracy", "Loss"])
plt.savefig("/IxiasBot/rendered/img/train_stats.png")
plt.show()
