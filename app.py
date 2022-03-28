import pandas as pd

#%% Load and Check Data

df = pd.read_csv(r"data.csv", encoding = "latin1")
df.info()

df = pd.concat([df.gender, df.description], axis = 1)
df.gender = [1 if each == "female" else 0 for each in df.gender]
df.head()

#%% Cleaning Data

import re #regular expression
df.dropna(axis = 0, inplace = True)

first_description = df.description[4]
description = re.sub("[^a-zA-Z]", " ", first_description) # replacing non-alphabet characters with space character.
description = description.lower() # making all letters lower.
print(description)

#%% Remove stopwords (irrelevant words) and split the words

import nltk # natural language tool kit
nltk.download("stopwords") # downloading the stopwords into corpus file.
from nltk.corpus import stopwords # importing stopwords from corpus file.

# splitting with tokenizer
description = nltk.word_tokenize(description) # we could have use description.split() but it doesn't split words like this: "shouldn't = should not"
description = [word for word in description if not word in set(stopwords.words("english"))] # I am going to show another alternative way instead of this way.
print(description)

#%% Lemmatization

import nltk as nlp
lemma = nlp.WordNetLemmatizer()
description = [lemma.lemmatize(word) for word in description]
description = " ".join(description)
print(description)

#%% Applying methods to all descriptions

description_list = []

for description in df.description:
    description = re.sub("[^a-zA-Z]", " ", description)
    description = description.lower()
    description = nltk.word_tokenize(description)
    #description = [word for word in description if not word in set(stopwords.words("english"))]
    #this method makes the process very long. So i am going to use another way instead of this. I already mentioned that I am going to show another way above.
    lemma = nlp.WordNetLemmatizer()
    description = [lemma.lemmatize(word) for word in description]
    description = " ".join(description)
    description_list.append(description)

#%% Bag of Words

from sklearn.feature_extraction.text import CountVectorizer # to create bag of words.
max_features = 5000 # using just 5000 words to make the process faster.

count_vectorizer = CountVectorizer(max_features = max_features, stop_words = "english") # preparing 5000 words to create sparse_matrix

sparse_matrix = count_vectorizer.fit_transform(description_list).toarray() # create sparse matrix. There is an example for sparse matrix in the picture above.
#print("{} common used words: {}".format(max_features, count_vectorizer.get_feature_names()))

#%% Classification

y = df.iloc[:,0].values # male or female classes
x = sparse_matrix

# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state = 42)

# naive bayes classification method
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)

#prediction
print("Accuracy:", nb.score(x_test, y_test))
