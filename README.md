# Credentials Finder Model
This repository contains code and documentation related to various data science projects. Each project focuses on different aspects of data collection, preprocessing, and modeling. The following sections provide an overview of each project and its corresponding files.

## Collection of Data

Over 500 source code files were collected from GitHub repos or generated from ChatGPT. A Python script was written to automatically clone over 50 repos to help with the data search. The files were seperated into two folders, Dirty folders when the files contained hardcoded credentials, and clean folders otherwise.
## Data Preprocessing

The "Data Preprocessing" project focuses on cleaning, transforming, and preparing the raw data for analysis. This section provides an overview of the data preprocessing steps performed on the extracted files, as represented by the provided code snippet.

First, the code snippet starts by extracting the files from the "projects.zip" archive into the "Projects" directory:

```
zip_path = "projects.zip"
extract_dir = "Projects"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(".") 
```
Next, the code separates the files into two directories: "Clean" and "Dirty". It collects the file paths and assigns labels based on whether the file is clean or dirty:

```
clean_dir = os.path.join(extract_dir, 'Clean')
dirty_dir = os.path.join(extract_dir, 'Dirty')

file_paths = []
labels = []
```
### Get all file paths and labels for clean files
```
for filename in os.listdir(clean_dir):
    file_paths.append(os.path.join(clean_dir, filename))
    labels.append('clean')
```
### Get all file paths and labels for dirty files
```
for filename in os.listdir(dirty_dir):
    file_paths.append(os.path.join(dirty_dir, filename))
    labels.append('dirty')
```    
After collecting the file paths and labels, the code reads the contents of each file and stores them in the file_contents list:

```
file_contents = []
for file_path in file_paths:
    with open(file_path, 'r', errors='ignore') as file:
        file_contents.append(file.read())
```        
The next step involves splitting the data into training and testing sets using the train_test_split function from scikit-learn:

```
X_train, X_test, y_train, y_test = train_test_split(file_contents, labels, test_size=0.2, random_state=42)
```
To prepare the textual data for modeling, the code utilizes the TfidfVectorizer from scikit-learn to convert the text documents into numerical feature vectors:

```
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train).toarray()
X_test_vectorized = vectorizer.transform(X_test).toarray()
```
The resulting feature vectors are stored in X_train_vectorized and X_test_vectorized. The code then encodes the labels using the LabelEncoder from scikit-learn:
```
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)
```
The encoded labels are stored in y_train_encoded and y_test_encoded, respectively.

Finally, the code snippet prints the shape of the training and testing feature vectors:

```
print(X_train_vectorized.shape)
print(X_test_vectorized.shape)
```
These steps collectively preprocess the data by extracting, labeling, cleaning, and transforming it into a suitable format for further analysis and modeling.

Please let me know if you have any further questions or if there's anything else I can assist you with!
### LSTM Model
The LSTM (Long Short-Term Memory) model is a recurrent neural network architecture commonly used for sequence modeling tasks. This section provides an overview of the LSTM model implemented for the project.

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

max_words = 10000  # Maximum size of the vocabulary
embedding_dim = 128  # Embedding size for each word
max_len = 14508  # Maximum length of an input sequence

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len))
model.add(LSTM(512, return_sequences=True))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(256, activation='gelu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='gelu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='gelu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_vectorized, y_train_encoded, batch_size=65, epochs=25, validation_data=(X_test_vectorized, y_test_encoded))
```
The code above represents the implementation of an LSTM model for the project. The model architecture consists of multiple LSTM layers with different hidden units, followed by dropout layers for regularization. The output is passed through fully connected (Dense) layers with GELU activation. The final output layer utilizes the softmax activation function to produce the predicted class probabilities.

The model is compiled with the Adam optimizer and uses the sparse categorical cross-entropy loss function for multiclass classification. During training, the model is fit to the training data (X_train_vectorized and y_train_encoded) with a specified batch size and number of epochs. The model's performance is monitored on the validation set (X_test_vectorized and y_test_encoded).
The model has 78.02% accuracy

## Naive Bayes Model

The Naive Bayes model is a probabilistic classification algorithm based on Bayes' theorem with the assumption of independence between features. This section provides an overview of the Naive Bayes model implemented for the project.

```
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Create the Naive Bayes model
model = MultinomialNB()

# Define the parameter grid for GridSearchCV
param_grid = {
    'alpha': [0.1, 0.5, 1.0],  # Add more alpha values as needed
}

# Create the KFold cross-validation object
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(model, param_grid, cv=kf, scoring='accuracy')
grid_search.fit(X_train_vectorized, y_train_encoded)
```
The code above represents the implementation of a Naive Bayes model for the project. The MultinomialNB class from scikit-learn is used to create the model. In this example, we use the alpha parameter to control the Laplace smoothing or additive smoothing.

To find the best hyperparameters for the Naive Bayes model, we perform a grid search using the GridSearchCV class from scikit-learn. The param_grid dictionary defines the range of values to search for the alpha parameter. The cross-validation strategy is specified using the KFold object, and we use accuracy as the scoring metric.

After fitting the grid search object to the training data (X_train_vectorized and y_train_encoded), the best hyperparameters can be accessed using grid_search.best_params_, and the best model can be accessed using grid_search.best_estimator_.

You can further evaluate the Naive Bayes model using metrics such as classification report and confusion matrix. Additionally, you can customize the hyperparameter grid (param_grid) or explore other variations of Naive Bayes models, such as Gaussian Naive Bayes, depending on the nature of your data.


## Logistic Regression Model

The Logistic Regression model is a popular linear classification algorithm that utilizes a logistic function to model the relationship between input features and class probabilities. This section provides an overview of the Logistic Regression model implemented for the project.

```
# Create the Logistic Regression model
model = LogisticRegression()

# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1.0, 10.0],  # Add more C values as needed
    'penalty': ['l1', 'l2']  # Add more penalty options as needed
}

# Create the KFold cross-validation object
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(model, param_grid, cv=kf, scoring='accuracy')
grid_search.fit(X_train_vectorized, y_train_encoded)
```
The code above represents the implementation of a Logistic Regression model for the project. The LogisticRegression class from scikit-learn is used to create the model.

To find the best hyperparameters for the Logistic Regression model, we perform a grid search using the GridSearchCV class from scikit-learn. The param_grid dictionary defines the range of values to search for the C (inverse regularization strength) and penalty (regularization type) parameters.

The cross-validation strategy is specified using the KFold object, and we use accuracy as the scoring metric. By fitting the grid search object to the training data (X_train_vectorized and y_train_encoded), the best hyperparameters can be accessed using grid_search.best_params_, and the best model can be accessed using grid_search.best_estimator_.

## Transformer Model
The model represents the implementation of a Transformer Classifier for the project. The architecture consists of a MultiHeadSelfAttention layer, followed by layer normalization, feed-forward neural network (FFN), dropout, and global average pooling.

The Transformer Classifier is created by instantiating the TransformerClassifier class, which takes the embedding dimension (embed_dim), number of attention heads (num_heads), and feed-forward dimension (ff_dim) as input parameters.

The model is built using the Keras functional API, with an input layer (inputs), embedding layer (embedding_layer), Transformer block (transformer_block), and subsequent dense layers. The final output is passed through a softmax activation function for classification.

The training process involves a custom train_step function, which performs forward pass, calculates the loss, and computes gradients for backpropagation. The model is trained using an optimizer (Adam in this example) and a loss function (sparse categorical cross-entropy). Gradients are accumulated and applied in batches to optimize memory usage.
Due to the fact that the files were large in size, and the transformer models require lots of GPU RAM, we were not able to train this model due to running out of RAM.
