import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression         # model creation
from sklearn.metrics import confusion_matrix                # metrics evaluation

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('balanced_review.csv')
print(df.head()) # prints the first 5 rows by default
print(f"This dataset has {df.shape[0]} rows and {df.shape[1]} columns")

# we got missing values, so we drop them and clean the data
df1 = df.copy()
df1.dropna(inplace = True)
print(df1.isnull().sum())

# we'll be plotting a pie chart for the same
df1['overall'].value_counts().plot(kind = 'pie', autopct = '%1.1f%%', shadow = False, explode = [0.1, 0, 0, 0, 0])

# the first value (3) will be exploded out of the chart by 0.1 times the radius of the pie
plt.title("Distribution of the Overall Rating")

fig = plt.gcf() # gcf() stands for 'get current figure'

# sets the height and width of the image as 7x7
fig.set_size_inches(7,7)
plt.show()

# rating 3 is neutral, and could be biased
# so we use separate dataframes with 1, 2 as negative and 4, 5 as positive ratings
df2 = df1[df1['overall'] != 3]
print(df2.shape) # retrieving dimensions

print(df2['overall'].value_counts()) # count of all the ratings, except 3

# positivity column is set to 1 if rating>3, otherwise it's 0
df2['positivity'] = np.where(df2['overall'] > 3 , 1, 0)


########################
# TRAINING AND TESTING #
########################

x = df2['reviewText']
y = df2['positivity']

# 20% - testing, 80% - training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# number of samples for each
print(f'Number of samples of x_train: {x_train.shape}')
print(f'Number of samples of x_test: {x_test.shape}')
print(f'Number of samples of y_train: {y_train.shape}')
print(f'Number of samples of y_test: {y_test.shape}')

# applying the bag of words classifier
# CountVectorizer() converts text data to matrix representation
# each row corresponds to a text document
# each column corresponds to to a unique word/token
# fit() analyses the text data (the argument) and learns the vocab from it

print("\n\nLoading...")
vect = CountVectorizer().fit(x_train)
x_train_vectorized = vect.transform(x_train)


###############################################
# CREATING THE FINAL MODEL FOR CLASSIFICATION #
###############################################

# to train and make predictions on binary outcomes
model = LogisticRegression(solver = 'liblinear')

model.fit(x_train_vectorized, y_train)


#############################
# PREDICITONS ON A TEST SET #
#############################

prediction = model.predict(vect.transform(x_test))

# Print the predicted sentiment for each review in the test set
for i in range(len(x_test)):
    print(f"Review: {x_test.iloc[i]}")
    print(f"Predicted Sentiment: {'Positive' if prediction[i] == 1 else 'Negative'}")
    print()


####################
# MODEL EVALUATION #
####################

cm = confusion_matrix(y_test, prediction)
print("Confusion Matrix:")
print(cm)

accuracy = (cm[0, 0] + cm[1, 1])*100 / np.sum(cm)
print(f"Accuracy: {accuracy:.2f}%")

precision = cm[1, 1] * 100 / (cm[1, 1] + cm[0, 1])
print(f"Precision: {precision:.2f}%")

recall = cm[1, 1] * 100 / (cm[1, 1] + cm[1, 0])
print(f"Recall: {recall:.2f}%")

f1 = 2 * (precision * recall) / (precision + recall)
print(f"F1 Score: {f1:.2f}%")
