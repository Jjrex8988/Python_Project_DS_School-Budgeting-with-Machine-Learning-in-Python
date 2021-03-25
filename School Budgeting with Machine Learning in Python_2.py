#--------------------------------------------------------------------------------#
## Chapter 3: Improving your model
## Here, you'll improve on your benchmark model using pipelines. Because the budget consists of
## both text and numeric data, you'll learn to how build pipielines that process multiple
## types of data. You'll also explore how the flexibility of the pipeline workflow makes testing
## different approaches efficient, even in complicated problems like this one!

## Pipelines, feature & text preprocessing
## Instantiate pipeline
import pandas as pd
import numpy as np

rng = np.random.RandomState(123)

SIZE = 1000

sample_data = {
 'numeric': rng.normal(0, 10, size=SIZE),
 'text': rng.choice(['', 'foo', 'bar', 'foo bar', 'bar foo'], size=SIZE),
 'with_missing': rng.normal(loc=3, size=SIZE)
}

sample_df = pd.DataFrame(sample_data)

sample_df.loc[rng.choice(sample_df.index, size=np.floor_divide(sample_df.shape[0], 5)),
              'with_missing'] = np.nan

foo_values = sample_df.text.str.contains('foo') * 10
bar_values = sample_df.text.str.contains('bar') * -25
no_text = ((foo_values + bar_values) == 0) * 1

val = 2 * sample_df.numeric + -2 * (foo_values + bar_values + no_text) + 4 * sample_df.with_missing.fillna(3)
val += rng.normal(0, 8, size=SIZE)

sample_df['label'] = np.where(val > np.median(val), 'a', 'b')

print(sample_df)

# Import Pipeline
from sklearn.pipeline import Pipeline

# Import other necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Split and select numeric data only, no nans
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric']],
                                                    pd.get_dummies(sample_df['label']),
                                                    random_state=22)

# Instantiate Pipeline object: pl
pl = Pipeline([
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# Fit the pipeline to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - numeric, no nans: ", accuracy)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Preprocessing numeric features
# Import the Imputer object
from sklearn.impute import SimpleImputer

# Create training and test sets using only numeric data
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric', 'with_missing']],
                                                    pd.get_dummies(sample_df['label']),
                                                    random_state=456)
# Insantiate Pipeline object: pl
pl = Pipeline(
    [
        ('imp', SimpleImputer()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ]
)

# Fit the pipeline to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - all numeric, incl nans: ", accuracy)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Text features and feature unions
## Preprocessing text features

# Import the CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Split out only the text data
X_train, X_test, y_train, y_test = train_test_split(sample_df['text'],
                                                    pd.get_dummies(sample_df['label']),
                                                    random_state=456)

# print(type(sample_df['text']))
# print(type(sample_df[['text']]))

# Instantiate Pipeline object: pl
pl = Pipeline([
        ('vec', CountVectorizer()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - just text data: ", accuracy)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Multiple types of processing: FunctionTransformer
# Import FunctionTransformer
from sklearn.preprocessing import FunctionTransformer

# Obtain the text data: get_text_data
get_text_data = FunctionTransformer(lambda x: x['text'], validate=False)

# Obtain the numeric data: get_numeric_data
get_numeric_data = FunctionTransformer(lambda x: x[['numeric', 'with_missing']], validate=False)

# Fit and transform the text data: just_text_data
just_text_data = get_text_data.fit_transform(sample_df)

# Fit and transform the numeric data: just_numeric_data
just_numeric_data = get_numeric_data.fit_transform(sample_df)

# Print head to check results
print('Text Data')
print(just_text_data.head())
print('\nNumeric Data')
print(just_numeric_data.head())
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Multiple types of processing: FeatureUnion
# Import FeatureUnion
from sklearn.pipeline import FeatureUnion

# Split using ALL data in sample_df
X_train, X_test, y_train, y_test = train_test_split(sample_df[['numeric', 'with_missing', 'text']],
                                                    pd.get_dummies(sample_df['label']),
                                                    random_state=22)

# Create a FeatureUnion with nested pipeline: process_and_join_features
process_and_join_features = FeatureUnion(
    transformer_list=[
        ('numeric_features', Pipeline([
            ('selector', get_numeric_data),
            ('imputer', SimpleImputer())
        ])),
        ('text_features', Pipeline([
            ('selector', get_text_data),
            ('vectorizer', CountVectorizer())
        ]))
    ]
)

# Instantiate nested pipeline: pl
pl = Pipeline([
        ('union', process_and_join_features),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])


# Fit pl to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - all data: ", accuracy)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Choosing a classification model
## Using FunctionTransformer on the main dataset
df = pd.read_csv('TrainingData.csv', index_col=0)
# df = pd.read_csv('TrainingSetSample.csv', index_col=0)

# Labels
LABELS = ['Function',
          'Use',
          'Sharing',
          'Reporting',
          'Student_Type',
          'Position_Type',
          'Object_Type',
          'Pre_K',
          'Operating_Status']

NUMERIC_COLUMNS = ['FTE', "Total"]

# Convert object to category for LABELS
df[LABELS] = df[LABELS].apply(lambda x: x.astype('category'))


# Define combine_text_columns() for use in sklearn.preprocessing.FunctionTransformer
def combine_text_columns(data_frame, to_drop=NUMERIC_COLUMNS + LABELS):
    """ Takes the dataset as read in, drops the non-feature, non-text columns and
    then combines all of the text columns into a single vector that has all of
    the text for a row.

    :param data_frame: The data as read in with read_csv (no preprocessing necessary)
    :param to_drop (optional): Removes the numeric and label columns by default.
    """
    # drop non-text columns that are in the df
    to_drop = set(to_drop) & set(data_frame.columns.tolist())
    text_data = data_frame.drop(to_drop, axis=1)

    # replace nans with blanks
    text_data.fillna("", inplace=True)

    # joins all of the text items in a row (axis=1)
    # with a space in between
    return text_data.apply(lambda x: " ".join(x), axis=1)

# Import FunctionTransformer
from sklearn.preprocessing import FunctionTransformer
from multilabel import multilabel_train_test_split

# Get the dummy encoding of the labels
dummy_labels = pd.get_dummies(df[LABELS])
print(dummy_labels)

# Get the columns that are features in the original df
NON_LABELS = [c for c in df.columns if c not in LABELS]

print(NON_LABELS)

X_train, X_test, y_train, y_test = multilabel_train_test_split(df[NON_LABELS],
                                                               dummy_labels,
                                                               size=0.2,
                                                               seed=123)

# Preprocess the text data: get_text_data
get_text_data = FunctionTransformer(combine_text_columns, validate=False)

# Preprocess the numeric data: get_numeric_data
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Add a model to the pipeline
# Complete the pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', SimpleImputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Try a different class of model
# Import random forest classifer
from sklearn.ensemble import RandomForestClassifier

# Edit model step in pipeline
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', SimpleImputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('clf', RandomForestClassifier())
    ])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Can you adjust the model or parameters to improve accuracy?
# Import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# Add model step to pipeline: pl
# Add model step to pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', SimpleImputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('clf', RandomForestClassifier(n_estimators=20))
    ])


# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Chapter 4: Learning from the experts
## In this chapter, you will learn the tricks used by the competition winner, and implement
## them yourself using scikit-learn. Enjoy!

## Learning from the expert: processing
## How many tokens?

## (Q) how many tokens are in the following string from the main dataset?
##  'PLANNING,RES,DEV,& EVAL      '

## (A) 4, because , and & are not tokens
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Deciding what's a word
# Import the CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Create the text vector
text_vector = combine_text_columns(X_train)

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Instantiate the CountVectorizer: text_features
text_features = CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

# Fit text_features to the text vector
text_features.fit(text_vector)

# Print the first 10 tokens
print(text_features.get_feature_names()[:10])
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## N-gram range in scikit-learn
# Import pipeline
from sklearn.pipeline import Pipeline

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Import other preprocessing modules
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import chi2, SelectKBest

# Select 300 best features
chi_k = 300

# Import functional utilities
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from sklearn.pipeline import FeatureUnion

# Perform preprocessing
get_text_data = FunctionTransformer(combine_text_columns, validate=False)
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Instantiate pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', SimpleImputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                   ngram_range=(1, 2))),
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])

# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Learning from the expert: a stats trick
## Which models of the data include interaction terms?

## (Q) Which expression(s) include interaction terms?
## (A) βxy + βx + βy
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Implement interaction modeling in scikit-learn

from SparseInteractions import SparseInteractions

# Instantiate pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', SimpleImputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                   ngram_range=(1, 2))),
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('int', SparseInteractions()),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])
# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Learning from the expert: the winning model

## (Q) Why is hashing a useful trick?

## (A) Some problems are memory-bound and not easily parallelizable, and hashing enforces a
## fixed length computation instead of using a mutable datatype (like a dictionary).
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Implementing the hashing trick in scikit-learn
# Import HashingVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

# Get text data: text_data
text_data = combine_text_columns(X_train)

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'

# Instantiate the HashingVectorizer: hashing_vec
hashing_vec = HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC)

# Fit and transform the Hashing Vectorizer
hashed_text = hashing_vec.fit_transform(text_data)

# Create DataFrame and print the head
hashed_df = pd.DataFrame(hashed_text.data)
print(hashed_df.head())
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Build the winning model
# Import the hashing vectorizer
from sklearn.feature_extraction.text import HashingVectorizer

# Instantiate the winning model pipeline: pl
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list=[
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', SimpleImputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                        # norm=None, binary=False,
                                                     non_negative=True, norm=None, binary=False,
                                                     ngram_range=(1, 2))),
                    ('dim_red', SelectKBest(chi2, chi_k))
                ]))
             ]
        )),
        ('int', SparseInteractions(degree=2)),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])
# Fit to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## What tactics got the winner the best score?
## (Q) Let's take a moment to reflect on why this model did so well. What tactics got the
## winner the best score?

## (A) The winner used skillful NLP, efficient computation, and simple but powerful stats
##tricks to master the budget data.
print("-"*38)
#--------------------------------------------------------------------------------#



#--------------------------------------------------------------------------------#
## Next steps and the social impact of your work
