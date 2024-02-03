# Sentiment-Analysis-for-Homonyms-Problem
Homonyms are words that share the same spelling or form (characters) but possess distinct meanings. For instance, the term "bank" can assume two disparate contexts, denoting both a "financial institution" and the "edge of a river."

These homonyms hold significant relevance in sentiment analysis, given their capacity to alter the meaning or emotional tone of a sentence entirely. Consider the following examples that highlight this challenge:

Sentence 1: "I hate the selfishness in you"** => Label: "NEGATIVE"
Sentence 2: "I hate anyone who hurts you" => Label: "POSITIVE"
In the first sentence, the word "hate" renders the sentiment as NEGATIVE. Conversely, the same word, "hate" appears in the second sentence, shaping the sentiment of the sentence as POSITIVE. This poses a considerable challenge to models relying on fixed word embeddings. Therefore, employing contextualized embeddings leveraging attention mechanisms from transformers becomes crucial to grasp the comprehensive context within a sentence.

# Tools Used
The project is implemented using the following Python packages:

| Package | Description |
| --- | --- |
| NumPy | Numerical computing library |
| Pandas | Data manipulation library |
| Matplotlib | Data visualization library |
| Sklearn | Machine learning library |
| TensorFlow | Open-source machine learning framework |
| TensorFlow | High-level deep learning API |
| Transformers | Hugging Face package contains state-of-the-art Natural Language Processing models |
| Dataset | Hugging Face package contains datasets |

# Dataset
The Stanford Sentiment Treebank (SST) is a corpus with fully labeled parse trees that allows for a complete analysis of the compositional effects of sentiment in language. The corpus is based on the dataset introduced by Pang and Lee (2005) and consists of 11,855 single sentences extracted from movie reviews. It was parsed with the Stanford parser and includes a total of 215,154 unique phrases from those parse trees, each annotated by 3 human judges.

Binary classification experiments on full sentences (negative or somewhat negative vs somewhat positive or positive with neutral sentences discarded) refer to the dataset as SST-2 or SST binary.

The dataset contains 3 features [`idx`, `sentence`, and `label`] and it comes in 3 different splits [train, validation, and test]. Here is the number of samples per split:

| Split | Number of Samples |
| --- | --- |
| train | 67349 |
| validation | 872 |
| test | 1821 |

# Methodology
## Dataset Preparation
This phase includes the dataset preparation before modeling and this includes: converting the `sentence` column to integers using the `Tokenizer` class from `keras`, then padding these integer sequences to match the highest sequence length in the dataset. 

## Models Selection
The 2 selected models are: LSTM-based and DistilBERT-based. The reason for the choice of these models is I need to see the difference between fixed-embeddings models (LSTM) and the contextualized embeddings models (DistilBERT). I have also trained 2 versions of the LSTM-based model one before addressing the imbalance in the dataset and one after solving it.

# Results
Before training, I did a simple grid search for the optimal hyperparameters like the embedding dimension of the `keras.Embedding` layer and the number of hidden units in the LSTM cell. Here are the detailed results summarized in the following table ordered by the highest validation accuracy:

| Package | Validation Loss | Validation Accuracy |
| --- | --- | --- |
| (768, 128) | 0.471817	 | 0.840596 |
| (768, 32)	| 0.492739	 | 0.839450 |
| (64, 64) | 0.455455	 | 0.833716 |
| (512, 128)	| 0.542884 | 0.833716 |
| (128, 64)	| 0.420377	 | 0.829128 |
| (64, 128)	| 0.479428 | 0.822248 |
| (512, 64)	| 0.487282 | 0.817661 |
| (512, 32)	| 0.444669 | 0.815367 |
| (128, 128)	| 0.600242 | 0.807339 |
| (768, 64)	| 0.498956 | 0.802752 |
| (128, 32)	 | 0.640019 | 0.678899 |
| (64, 32)	| 0.695933 | 0.509174 |

Here is a graphical representation of the above table:
![download](https://github.com/MohammedAly22/Sentiment-Analysis-for-Homonyms-Problem/assets/90681796/f6594b2b-f3c4-4979-9ca6-ae7f25d2e84d)

As we can see, the optimal hyperparameters achieving the best validation accuracy are 768 and 128 for embedding_dim and hidden_dim respectively.

## LSTM-Based Model (Imbalanced)


## LSTM-Based Model (Balanced)

## DistilBERT-Base Model

# Conclusion
