# Sentiment-Analysis-for-Homonyms-Problem
**Homonyms** are words that **share the same spelling** or form (characters) but possess **distinct meanings**. For instance, the term **"bank"** can assume two disparate contexts, denoting both a **financial institution** and **the edge of a river**.

These homonyms hold **significant relevance** in **sentiment analysis**, given their capacity to **alter a sentence's meaning** or **emotional tone** entirely. Consider the following examples that highlight this challenge:

| Sentence | Label |
| --- | --- |
| I **hate** the selfishness in you | NEGATIVE |
| I **hate** anyone who hurts you | POSITIVE |

In the first sentence, the word **"hate"** renders the sentiment as **NEGATIVE**. Conversely, the same word, "hate" appears in the second sentence, shaping the sentiment of the sentence as **POSITIVE**. This poses a **considerable challenge** to models relying on **fixed word embeddings**. Therefore, **employing contextualized embeddings leveraging attention mechanisms from transformers becomes crucial to grasping the comprehensive context within a sentence**.

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
| Datasets | Hugging Face package contains datasets |

# Dataset
The [Stanford Sentiment Treebank (SST)](https://huggingface.co/datasets/sst2) is a corpus with fully labeled parse trees that allows for a complete analysis of the compositional effects of sentiment in language. The corpus is based on the dataset introduced by Pang and Lee (2005) and consists of **11,855** single sentences extracted from **movie reviews**. It was parsed with the Stanford parser and includes a total of **215,154** unique phrases from those parse trees, each annotated by 3 human judges.

Binary classification experiments on full sentences (negative or somewhat negative vs somewhat positive or positive with neutral sentences discarded) refer to the dataset as SST-2 or SST binary.

The dataset contains **3** features [`idx`, `sentence`, and `label`] and it comes in 3 different splits [train, validation, and test]. Here is the number of samples per split:

| Split | Number of Samples |
| --- | --- |
| train | 67349 |
| validation | 872 |
| test | 1821 |

# Methodology
## Dataset Preparation
This phase encompasses dataset preparation preceding the modeling phase. It involves transforming the `sentence` column into integers utilizing the `Tokenizer` class from the `keras` library. Subsequently, the integer sequences are padded to conform to the maximum sequence length within the dataset.

## Models Selection
The two models selected are **LSTM-based** and **DistilBERT-based**. The rationale behind choosing these models lies in the necessity to observe distinctions between **fixed-embedding** models, exemplified by **LSTM**, and **contextualized embedding** models, exemplified by **DistilBERT**. Additionally, two iterations of the **LSTM-based** model were trained: **one before addressing the dataset imbalance and another after resolving it**.

# Results
Before training, I conducted a **straightforward grid search** to determine the **optimal hyperparameters**. This included parameters such as the **embedding dimension** for the Keras `Embedding` layer and the number of hidden units in the LSTM cell. The detailed results, ordered by the highest validation accuracy, are presented in the following table:

| Package | Validation Loss | Validation Accuracy |
| --- | --- | --- |
| (768, 128) | 0.471817	 | **0.840596** |
| (768, 32)	| 0.492739	 | 0.839450 |
| (64, 64) | 0.455455	 | 0.833716 |
| (512, 128)	| 0.542884 | 0.833716 |
| (128, 64)	| **0.420377**	 | 0.829128 |
| (64, 128)	| 0.479428 | 0.822248 |
| (512, 64)	| 0.487282 | 0.817661 |
| (512, 32)	| 0.444669 | 0.815367 |
| (128, 128)	| 0.600242 | 0.807339 |
| (768, 64)	| 0.498956 | 0.802752 |
| (128, 32)	 | 0.640019 | 0.678899 |
| (64, 32)	| 0.695933 | 0.509174 |

Here is a graphical representation of the above table:
![download](https://github.com/MohammedAly22/Sentiment-Analysis-for-Homonyms-Problem/assets/90681796/f6594b2b-f3c4-4979-9ca6-ae7f25d2e84d)

As we can see, the optimal hyperparameters achieving the best validation accuracy are **768** and **128** for `embedding_dim` and `hidden_dim` respectively.

## Model Curves
### LSTM-Based Model (Imbalanced)
| Loss Curve | Accuracy Curve | Confusion Matrix |
| --- | --- | --- |
| ![download](https://github.com/MohammedAly22/Sentiment-Analysis-for-Homonyms-Problem/assets/90681796/428b0c31-0911-469f-aa41-91e0c98b0248) | ![download](https://github.com/MohammedAly22/Sentiment-Analysis-for-Homonyms-Problem/assets/90681796/2f953c98-684c-497a-93a0-b0aacab1a5fc) | ![download](https://github.com/MohammedAly22/Sentiment-Analysis-for-Homonyms-Problem/assets/90681796/cb328b64-437f-45ad-bac4-335aa4cd4132)|

Classification Report:
|  | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| 0 | 0.83 | **0.87** | 0.85 | 428 |
| 1 | **0.87** | 0.82 | 0.85 | 444 |
| accuracy |  | | **0.85** | 872 |
| macro avg | 0.85 | 0.85 | 0.85 | 872 |
| weighted avg | 0.85 | 0.85 | 0.85 | 872 |

### LSTM-Based Model (Balanced)
| Loss Curve | Accuracy Curve | Confusion Matrix |
| --- | --- | --- |
| ![download](https://github.com/MohammedAly22/Sentiment-Analysis-for-Homonyms-Problem/assets/90681796/f0af9cf2-d27c-4419-9e8c-beafd38e7b76)| ![download](https://github.com/MohammedAly22/Sentiment-Analysis-for-Homonyms-Problem/assets/90681796/fbb1c9d6-4692-4181-aa22-056bc0d53035) | ![download](https://github.com/MohammedAly22/Sentiment-Analysis-for-Homonyms-Problem/assets/90681796/440e8ef2-1ff0-4462-aa1c-8070ddd8e60e) |

Classification Report:
|  | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| 0 | **0.85** | 0.79 | 0.82 | 428 |
| 1 | 0.81 | **0.86** | 0.83 | 444 |
| accuracy |  | | **0.83** | 872 |
| macro avg | 0.83 | 0.83 | 0.83 | 872 |
| weighted avg | 0.83 | 0.83 | 0.83 | 872 |

### DistilBERT-Base Model
| Loss Curve | Accuracy Curve | Confusion Matrix |
| --- | --- | --- |
| ![download](https://github.com/MohammedAly22/Sentiment-Analysis-for-Homonyms-Problem/assets/90681796/a56d787d-bb73-4f63-9fc4-8f1c26afb3ac) | ![download](https://github.com/MohammedAly22/Sentiment-Analysis-for-Homonyms-Problem/assets/90681796/ec274a41-392d-40e2-b397-eae8d83afe93) | ![download](https://github.com/MohammedAly22/Sentiment-Analysis-for-Homonyms-Problem/assets/90681796/cff1dc1e-12af-486f-999a-97af469edba4) |

Classification Report:
|  | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| 0 | **0.92** | 0.89 | 0.90 | 428 |
| 1 | 0.89 | **0.92** | 0.91 | 444 |
| accuracy |  | | **0.90** | 872 |
| macro avg | 0.91 | 0.90 | 0.90 | 872 |
| weighted avg | 0.91 | 0.90 | 0.90 | 872 |

## Models Testing
to test the models' abilities to solve the homonyms problems in sentiment analysis, I prepared some confusing test cases summarized in the following table:
![models_predictions](https://github.com/MohammedAly22/Sentiment-Analysis-for-Homonyms-Problem/assets/90681796/11c6083b-7cd0-406b-aaed-549e724a658d)

The analysis of the **LSTM-based** model with an **imbalanced dataset** indicates **strong performance on straightforward sentences**, but its effectiveness **diminishes** in **complex cases** due to the constraints of **fixed embeddings**. Notably, the model attempts to **refine its comprehension** by **incorporating additional elements**, showcasing a **nuanced understanding**. Surprisingly, the **LSTM-based** model with a **balanced dataset** exhibits **no improvement** and tends to **degrade**, emphasizing the **impact of data removal during undersampling**. The use of **fixed embeddings** is **effective** for **simpler sentences** but **falls** short of addressing the identified issues. In contrast, the **DistilBERT** model, utilizing **contextualized embeddings**, demonstrates **significant improvement**, achieving around **91% accuracy** after **three epochs**. DistilBERT excels in handling words with **multiple meanings**, showcasing its ability to make **nuanced decisions** even when classifying a sentence like `I hate anyone hurting you` as `NEGATIVE` with a **low confidence score** of about **51%**. This highlights the transformative power of **contextualized embeddings** in enhancing sentiment analysis.

# Conclusion
Ultimately, the decision to opt for fixed embedding models or contextualized embeddings hinges on the nature of the data the model will encounter in real-world scenarios. Fixed embeddings might suffice when dealing with straightforward data, offering good performance while requiring less memory than transformer-based models. However, in cases where the data is more complex, as demonstrated in our test cases, leveraging a transformer-based model with a self-attention mechanism can yield substantial performance improvements. It's crucial to note that this advantage comes at the expense of a higher memory footprint.
