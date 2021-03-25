#--------------------------------------------------------------------------------#
## Chapter 1: Exploring the raw data
## In this chapter, you'll be introduced to the problem you'll be solving in this course.
## How do you accurately classify line-items in a school budget based on what that money is
## being used for? You will explore the raw text and numeric values in the dataset, both quantitatively
## and visually. And you'll learn how to measure success when trying to predict class labels for each
## row of the dataset.

## Introducing the challenge
## What category of problem is this?

## (Q) Your goal is to develop a model that predicts the probability for each possible label by relying
## on some correctly labeled examples. What type of machine learning problem is this?

## (A) Supervised Learning, because the model will be trained using labeled examples
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## What is the goal of the algorithm?

## (Q) As you know from previous courses, there are different types of supervised machine learning
## problems. In this exercise you will tell us what type of supervised machine learning problem
## this is, and why you think so.

## Remember, your goal is to correctly label budget line items by training a supervised
## model to predict the probability of each possible label, taking most probable label
##as the correct label.

## (A) Classification, because predicted probabilities will be used to select a label class.
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Loading the data

import pandas as pd

df = pd.read_csv("TrainingData.csv", index_col=0)
print(df.info())

# How many rows are there in the training data?
# How many columns are there in the training data?
# How many non-null entries are in the Job_Title_Description column?

## (A)1560 rows, 25 columns, 1131 non-null entries in Job_Title_Description
## (A)400277 rows, 25 columns, 292743 non-null entries in Job_Title_Description
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Summarizing the data
# Print the summary statistics
print(df.describe())

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Create the histogram
plt.hist(df['FTE'].dropna())

# Add title and labels
plt.title('Distribution of %full-time \n employee works')
plt.xlabel('% of full-time')
plt.ylabel('num employees')

# Display the histogram
plt.show()
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Exploring datatypes in pandas

## (Q) How many columns with dtype object are in the data?
## (A) 23

print(df.dtypes.value_counts())
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Encode the labels as categorical variables
LABELS = ['Function',
          'Use',
          'Sharing',
          'Reporting',
          'Student_Type',
          'Position_Type',
          'Object_Type',
          'Pre_K',
          'Operating_Status']


# Define the lambda function: categorize_label
categorize_label = lambda x: x.astype('category')

# Convert df[LABELS] to a categorical type
df[LABELS] = df[LABELS].apply(categorize_label, axis=0)

# Print the converted dtypes
print(df[LABELS].dtypes)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Counting unique labels
# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Calculate number of unique values for each label: num_unique_labels
num_unique_labels = df[LABELS].apply(pd.Series.nunique)

# Plot number of unique values for each label
num_unique_labels.plot(kind="bar")

# Label the axes
plt.xlabel('Labels')
plt.ylabel('Number of unique values')

# Display the plot
plt.show()
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## How do we measure success?

## (Q) Select the ordering of the examples which corresponds to the lowest to highest
## log loss scores. y is an indicator of whether the example was classified correctly.
## You shouldn't need to crunch any numbers!

import numpy as np


def compute_log_loss(predicted, actual, eps=1e-14):

    """ Computes the logarithmic loss between predicted and
    actual when these are 1D arrays.
    :param predicted: The predicted probabilities as floats between 0-1
    :param actual: The actual binary labels. Either 0 or 1.
    :param eps (optional): log(0) is inf, so we need to offset our
    predicted values slightly by eps from 0 or 1.
    """
    predicted = np.clip(predicted, eps, 1 - eps)
    loss = -1 * np.mean(actual * np.log(predicted)
                        + (1 - actual)
                        * np.log(1 - predicted))
    return loss


print(compute_log_loss(predicted=0.85, actual=1))
print(compute_log_loss(predicted=0.99, actual=0))
print(compute_log_loss(predicted=0.51, actual=0))

## (A) Lowest: A, Middle: C, Highest: B.
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Computing log loss with NumPy

actual_labels = np.array([1., 1., 1., 1., 1., 0., 0., 0., 0., 0.])

correct_confident = np.array([0.95, 0.95, 0.95, 0.95, 0.95, 0.05, 0.05, 0.05, 0.05, 0.05])

correct_not_confident = np.array([0.65, 0.65, 0.65, 0.65, 0.65, 0.35, 0.35, 0.35, 0.35, 0.35])

wrong_not_confident = np.array([0.35, 0.35, 0.35, 0.35, 0.35, 0.65, 0.65, 0.65, 0.65, 0.65])

wrong_confident = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.95, 0.95, 0.95, 0.95, 0.95])


# Compute and print log loss for 1st case
correct_confident_loss = compute_log_loss(correct_confident, actual_labels)
print("Log loss, correct and confident: {}".format(correct_confident_loss))

# Compute log loss for 2nd case
correct_not_confident_loss = compute_log_loss(correct_not_confident, actual_labels)
print("Log loss, correct and not confident: {}".format(correct_not_confident_loss))

# Compute and print log loss for 3rd case
wrong_not_confident_loss = compute_log_loss(wrong_not_confident, actual_labels)
print("Log loss, wrong and not confident: {}".format(wrong_not_confident_loss))

# Compute and print log loss for 4th case
wrong_confident_loss = compute_log_loss(wrong_confident, actual_labels)
print("Log loss, wrong and confident: {}".format(wrong_confident_loss))

# Compute and print log loss for actual labels
actual_labels_loss = compute_log_loss(actual_labels, actual_labels)
print("Log loss, actual labels: {}".format(actual_labels_loss))
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Chapter 2: Creating a simple first model
## In this chapter, you'll build a first-pass model. You'll use numeric data only to
## train the model. Spoiler alert - throwing out all of the text data is bad for performance!
## But you'll learn how to format your predictions. Then, you'll be introduced to natural
## language processing (NLP) in order to start working with the large amounts of text in the data.

## Setting up a train-test split in scikit-learn
from multilabel import multilabel_train_test_split

NUMERIC_COLUMNS = ['FTE', 'Total']

# Create the new DataFrame: numeric_data_only
numeric_data_only = df[NUMERIC_COLUMNS].fillna(-1000)

# Get labels and convert to dummy variables: label_dummies
label_dummies = pd.get_dummies(df[LABELS])

# Create training and test sets
X_train, X_test, y_train, y_test = multilabel_train_test_split(numeric_data_only,
                                                               label_dummies,
                                                               size=0.2,
                                                               seed=123)

# Print the info
print("X_train info:")
print(X_train.info())
print("\nX_test info:")
print(X_test.info())
print("\ny_train info:")
print(y_train.info())
print("\ny_test info:")
print(y_test.info())
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Training a model
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Create the DataFrame: numeric_data_only
numeric_data_only = df[NUMERIC_COLUMNS].fillna(-1000)

# Get labels and convert to dummy variables: label_dummies
label_dummies = pd.get_dummies(df[LABELS])

# Create training and test sets
X_train, X_test, y_train, y_test = multilabel_train_test_split(numeric_data_only,
                                                               label_dummies,
                                                               size=0.2,
                                                               seed=123)

# Instantiate the classifier: clf
clf = OneVsRestClassifier(LogisticRegression())

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Print the accuracy
print("Accuracy: {}".format(clf.score(X_test, y_test)))
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Use your model to predict values on holdout data
# Instantiate the classifier: clf
clf = OneVsRestClassifier(LogisticRegression())

# Fit it to the training data
clf.fit(X_train, y_train)

# Load the holdout data: holdout
holdout = pd.read_csv('TestData.csv', index_col=0)

# Generate predictions: predictions
predictions = clf.predict_proba(holdout[NUMERIC_COLUMNS].fillna(-1000))

print(predictions)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Writing out your results to a csv for submission

## https://stackoverflow.com/questions/57507832/unable-to-allocate-array-with-shape-and-data-type

BOX_PLOTS_COLUMN_INDICES = [range(0, 37),
                            range(37, 48),
                            range(48, 51),
                            range(51, 76),
                            range(76, 79),
                            range(79, 82),
                            range(82, 87),
                            range(87, 96),
                            range(96, 104)]

def _multi_multi_log_loss(predicted,
                          actual,
                          class_column_indices=BOX_PLOTS_COLUMN_INDICES,
                          eps=1e-15):
    """ Multi class version of Logarithmic Loss metric as implemented on
    DrivenData.org
    """
    class_scores = np.ones(len(class_column_indices), dtype=np.float64)

    # calculate log loss for each set of columns that belong to a class:
    for k, this_class_indices in enumerate(class_column_indices):
        # get just the columns for this class
        preds_k = predicted[:, this_class_indices].astype(np.float64)

        # normalize so probabilities sum to one (unless sum is zero, then we clip)
        preds_k /= np.clip(preds_k.sum(axis=1).reshape(-1, 1), eps, np.inf)

        actual_k = actual[:, this_class_indices]

        # shrink predictions so
        y_hats = np.clip(preds_k, eps, 1 - eps)
        sum_logs = np.sum(actual_k * np.log(y_hats))
        class_scores[k] = (-1.0 / actual.shape[0]) * sum_logs

    return np.average(class_scores)


def score_submission(pred_path='./',
                     holdout_path='C:/Users/Jjrex8988/pythonProject/TFTest/TrainingData.csv'):
    # this happens on the backend to get the score
    holdout_labels = pd.get_dummies(
        pd.read_csv(holdout_path, index_col=0)
            .apply(lambda x: x.astype('category'), axis=0)
    )

    preds = pd.read_csv(pred_path, index_col=0)

    # make sure that format is correct
    assert (preds.columns == holdout_labels.columns).all()
    assert (preds.index == holdout_labels.index).all()

    return _multi_multi_log_loss(preds.values, holdout_labels.values)


# # Generate predictions: predictions
# predictions = clf.predict_proba(holdout[NUMERIC_COLUMNS].fillna(-1000))
#
# # Format predictions in DataFrame: prediction_df
# prediction_df = pd.DataFrame(columns=pd.get_dummies(df[LABELS]).columns,
#                              index=holdout.index,
#                              data=predictions)
#
#
# # Save prediction_df to csv
# prediction_df.to_csv('predictions.csv')
#
# # Submit the predictions for scoring: score
# score = score_submission(pred_path='predictions.csv')
#
# # Print score
# print('Your model, trained with numeric data only, yields logloss score: {}'.format(score))
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## A very brief introduction to NLP
## Tokenizing text

## (Q) How many tokens (1-grams) are in the string
## Title I - Disadvantaged Children/Targeted Assistance
## if we tokenize on whitespace and punctuation

## (A) 6
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Testing your NLP credentials with n-grams

one_grams = ['petro', 'vend', 'fuel', 'and', 'fluids']

## (Q) In this exercise, your job is to determine the sum of the sizes of 1-grams,
## 2-grams and 3-grams generated by the string petro-vend fuel and fluids, tokenized on punctuation.

## (A) 12
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Representing text numerically
## Creating a bag-of-words in scikit-learn
# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Fill missing values in df.Position_Extra
df.Position_Extra.fillna("", inplace=True)

# Instantiate the CountVectorizer: vec_alphanumeric
vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

# Fit to the data
vec_alphanumeric.fit(df.Position_Extra)

# Print the number of tokens and first 15 tokens
msg = "There are {} tokens in Position_Extra if we split on non-alpha numeric"
print(msg.format(len(vec_alphanumeric.get_feature_names())))
print(vec_alphanumeric.get_feature_names()[:15])
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Combining text columns for tokenization


# Define combine_text_columns()
def combine_text_columns(data_frame, to_drop=NUMERIC_COLUMNS + LABELS):
    """ converts all text in each row of data_frame to single vector """

    # Drop non-text columns that are in the df
    to_drop = set(to_drop) & set(data_frame.columns.tolist())
    text_data = data_frame.drop(to_drop, axis=1)

    # Replace nans with blanks
    text_data.fillna("", inplace=True)

    # Join all text items in a row that have a space in between
    return text_data.apply(lambda x: " ".join(x), axis=1)

print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## What's in a token?
# Import the CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create the basic token pattern
TOKENS_BASIC = '\\S+(?=\\s+)'

# Create the alphanumeric token pattern
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Instantiate basic CountVectorizer: vec_basic
vec_basic = CountVectorizer(token_pattern=TOKENS_BASIC)

# Instantiate alphanumeric CountVectorizer: vec_alphanumeric
vec_alphanumeric = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

# Create the text vector
text_vector = combine_text_columns(df)

# Fit and transform vec_basic
vec_basic.fit(text_vector)


# Print number of tokens of vec_basic
print("There are {} tokens in the dataset".format(len(vec_basic.get_feature_names())))

# Fit and transform vec_alphanumeric
vec_alphanumeric.fit(text_vector)

# Print number of tokens of vec_alphanumeric
print("There are {} alpha-numeric tokens in the dataset".format(len(vec_alphanumeric.get_feature_names())))
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#