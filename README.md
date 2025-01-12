# DNN Classification Script

## Instagram Influencer Categorization using Deep Neural Networks
Overview


This project implements a deep neural network (DNN) to classify Instagram influencer accounts into various categories based on the textual data extracted from their biographies and post captions. The model processes input text, removes stopwords, applies vectorization, and uses a fully connected DNN for classification.


The solution is built using Python, TensorFlow, and Scikit-learn, leveraging TF-IDF vectorization and one-hot encoding for feature and label representation.

Features:
* Preprocessing steps include:
* Text normalization and stopword removal (with Turkish stopwords support).
* Conversion of emojis into descriptive text for better context understanding.
* TF-IDF vectorization for generating word-based feature vectors.
* Label encoding and one-hot encoding for categorical labels.
* DNN model with regularization, dropout layers, and batch normalization to enhance performance and prevent overfitting.
* Stratified train-test split to maintain label distribution consistency.

Dataset:
* Category_name
* Biographies (biography) and captions (captions) of Instagram influencers.
* Labels representing the influencer categories (e.g., "Tech," "Fashion," "Travel").


Processed Data:
* TF-IDF vectorized features for biographies and captions.
* One-hot encoded labels for categories.


Main Libraries Used:

pandas: For data handling and manipulation.

numpy: For numerical operations.

nltk: For natural language processing (e.g., stopword removal).

scikit-learn: For vectorization and label encoding.

tensorflow: For deep learning model implementation.

Preprocessing Workflow:


Text Normalization:

Converts text to lowercase, removes punctuation, numbers, and special characters.
Emojis are converted to descriptive text (e.g., 😊 → "smiling face").


Stopword Removal:

Removes common words (e.g., "ve," "de") using Turkish stopwords.


TF-IDF Vectorization:

Creates numerical representations of text, with a feature limit of 5000.


Label Encoding:

Converts categorical labels to one-hot representations for training.
Model Architecture
The DNN consists of:

Input layer: Matches the dimensionality of the TF-IDF features.

Three hidden layers:

* Fully connected (Dense) layers with ReLU activation.
* Dropout (50%) for regularization.
* Batch normalization for stable training.

Output layer:

Fully connected layer with softmax activation for multi-class classification.


Training:


Optimizer: Adam (learning rate = 0.001).

Loss Function: Categorical cross-entropy. (Given it is a multiclass classification)

Metrics: Accuracy.

Validation Split: 20% of the training set used for validation.

Epochs: 30.

Batch Size: 16.


Results

Training accuracy: ~59%.

Test accuracy: ~58% (as indicated in the script).



# XGBoost Classification Script

## Instagram Influencer Categorization using XGB
Overview


This project implements a XGBoost to classify Instagram influencer accounts into various categories based on the textual data extracted from their biographies and post captions. The model processes input text, removes stopwords, applies vectorization, and uses a CVGrid Searched approach for classification.


Features:
* Preprocessing steps include:
* Text normalization and stopword removal (with Turkish stopwords support).
* Conversion of emojis into descriptive text for better context understanding.
* TF-IDF vectorization for generating word-based feature vectors.
* Label encoding and one-hot encoding for categorical labels.
* Categorization was done with integers since XGBoost cannot mistakenly find relations between integers, as opposed to DNN case.
* Every possible parameter was swapped until convergence for a considerably long time to determine CVGrid results

  
Dataset:
* Category_name
* Biographies (biography) and captions (captions) of Instagram influencers.
* Labels representing the influencer categories (e.g., "Tech," "Fashion," "Travel").


Processed Data:
* TF-IDF vectorized features for biographies and captions.
* integer encoded labels for categories.


Preprocessing Workflow:


Text Normalization:

Converts text to lowercase, removes punctuation, numbers, and special characters.
Emojis are converted to descriptive text (e.g., 😊 → "smiling face").


Stopword Removal:

Removes common words (e.g., "ve," "de") using Turkish stopwords.


TF-IDF Vectorization:

Creates numerical representations of text, with a feature limit of 3000.


Label Encoding:

Converts categorical labels to integer representations for training.


Results

Training accuracy: ~65%.

Test accuracy: ~64% (as indicated in the script).


## Preprocessing the CSV files

The preprocess.ipynb script goes through the main dataset to fetch related information and collects them under output_data.csv(for classification) and output_reg.csv(for regression).
The fields found to be related are 

* "category_name",
* "captions" (of all posts, concatenated) and
* "biography"

## Postprocessing the CSV files
After the predictions are collected from XGB, DNN and Ali's MNB example, they are subjected to a voting scheme in results.py, where if MNB and DNN agree on a specific prediction, this is selected as the final result. If they do not agree, then XGB result is selected. I trust XGB more than the other two, that is why only time I am not selecting it, is the case where other two have created a dominance. This scheme has created for Phase 3, thus I do not know the contribution of accuracy. On my subjective tests, it has increased 65% to 67%.


