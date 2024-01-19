# Overview
This repository contains scripts and code used for a machine learning project aimed at predicting student grades based on interactions with ChatGPT. The project utilizes datasets consisting of ChatGPT conversation histories in HTML format and a CSV file containing student grades.

# Methodology
## **Text Preprocessing**

In our project, text preprocessing has been instrumental in data preparation for analysis. By employing a series of techniques including lowercasing, removal of punctuation and special characters, tokenization, stop-word elimination, stemming, and text normalization on both the conversations and questions, we have prepared our text data for feature engineering. These preprocessing steps include:

**Lowercasing:** All text data, including conversations and questions, was converted to lowercase to ensure uniformity and prevent the model from treating the same words differently based on case.

**Removing Punctuation and Special Characters:** Punctuation and special characters, such as commas, periods, question marks, and exclamation marks, were removed from the text. This step helps in focusing on the actual words and reduces noise in the data.

**Tokenization:** Tokenization is the process of breaking text into words or tokens. We used a tokenizer to split the text into individual words, which are then used as features for our machine learning model.

**Stop-Words Removal:** Stop words are common words that do not carry significant meaning, such as "the," "is," "at," etc. We removed these stop words from the text to reduce the dimensionality of the data and focus on the most meaningful words.

**Stemming:** Stemming is the process of reducing words to their root or base form. We used a stemming algorithm to normalize words by removing suffixes and prefixes, which helps in treating similar words as the same base word.

**Text Normalization:** Text normalization includes various steps to ensure consistency in the text data. We converted the stemmed tokens into space-separeted strings to ensure consistency between conversations and questions

## **Prompt Matching**

**Prompt and Question Comparison:** We compared each user prompt with a set of predefined questions using techniques like cosine similarity. Cosine similarity measures the cosine of the angle between two vectors, which in our case, represent the prompts and questions in a multi-dimensional space.

**Cosine Similarity Calculation:** The cosine similarity score is calculated between the vector representations of the user prompt and each question. This score indicates how similar the prompt is to each question, with higher scores indicating greater similarity.

**Establishing Correlations:** Based on the cosine similarity scores, we establish correlations between user prompts and specific questions. This allows us to identify which questions are most relevant or closely related to a given user prompt.

A part of the resulted Table is shown below.

| code                                | Q_0     | Q_1     | Q_2     | Q_3     | Q_4     | Q_5     | Q_6     | Q_7     | Q_8     |
|-------------------------------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| 0031c86e-81f4-4eef-9e0e-28037abf9883| 0.111065| 0.281578| 0.618961| 0.432243| 0.510732| 0.551559| 0.139785| 0.181310| 0.123534|
| 0225686d-b825-4cac-8691-3a3a5343df2b| 0.164216| 0.802719| 0.756955| 0.849414| 0.614525| 0.989534| 0.852307| 0.636646| 0.578007|
| 041f950b-c013-409a-a642-cffff60b9d4b| 0.089521| 0.247172| 0.502018| 0.329575| 0.627776| 0.399371| 0.445345| 0.481055| 0.268303|

## **Feature Engineering**
**Number of Prompts User Asked:** This feature reflects the number of prompts asked by the user. 

**Average Length of User and Assistant Prompts:** The average length of prompts from both the user and the assistant provides insight into the conversational dynamics. Differences in prompt lengths may indicate variations in communication styles or the complexity of the conversation.

**Ratio of Words Between User and Assistant:** This feature measures the balance of interaction between the user and the assistant. It calculates the ratio of the number of words in user prompts to the number of words in assistant responses, highlighting the distribution of communication between the two parties.

**Sentiment Analysis:** Sentiment analysis categorizes prompts as positive, negative, or neutral, capturing the emotional context of the conversation. This feature helps in understanding the tone and sentiment of the interaction.

**Ratio of Error Prompts from User:** This feature helps us understand potential challenges faced by the user. By counting the number of prompts that may indicate errors or confusion on the user's part, we can identify areas where the system can be improved or where users may need additional support. Instead of using the total number, we used ratio in order to reflect the data better.

**Conversation Length:** This feature provides an indication of the length of the conversation, which can be useful for analyzing user engagement patterns and system performance over time.

**Merging Conversation Features with Grades:** At the end of Feature Engineering, we have combined resulted data frame with the grades data from CSV file.

A part of the resulted table is shown below.
| code                                | Q_0     | Q_1     | Q_2     | Q_3     | Q_4     | Q_5     | Q_6     | Q_7     | Q_8     | #UserPrompts | User_Prompts_Avg_Num_Chars | Assistant_Prompts_Avg_Num_Chars | User Assistant Ratio | Positive | Negative | Neutral | Error Ratio | Length | grade |
|-------------------------------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------------|-----------------------------|----------------------------------|----------------------|----------|----------|---------|--------------|--------|-------|
| 0031c86e-81f4-4eef-9e0e-28037abf9883| 0.111065| 0.281578| 0.618961| 0.432243| 0.510732| 0.551559| 0.139785| 0.181310| 0.123534| 14            | 244.500000                  | 1401.071429                      | 0.212783             | 6        | 4        | 4       | 0.357143     | 2998   | 48.0  |
| 0225686d-b825-4cac-8691-3a3a5343df2b| 0.164216| 0.802719| 0.756955| 0.849414| 0.614525| 0.989534| 0.852307| 0.636646| 0.578007| 18            | 156.277778                  | 1073.000000                      | 0.182164             | 8        | 2        | 8       | 0.000000     | 2797   | 99.0  |
| 041f950b-c013-409a-a642-cffff60b9d4b| 0.089521| 0.247172| 0.502018| 0.329575| 0.627776| 0.399371| 0.445345| 0.481055| 0.268303| 9             | 417.444444                  | 1101.555556                      | 0.480653             | 3        | 0        | 6       | 0.555556     | 1722   | 90.0  |

**Detecting and Treating Outliers:** Before proceeding Model Training part, we have detected the outliers and replaced those values with mean of that spesific column, in order to to mitigate their impact on our analysis. 

## **Model Training**
**Dataset Splitting:** We divided our dataset into training and testing subsets, allocating 80% for training and 20% for testing. This splitting strategy ensures that our model is trained on a sufficient amount of data while also having a separate set for evaluation to avoid overfitting.

### **Regression Algorithms Implementation**

**Base Case:** A regression algortihm that is provided to us by the instructor. We have used this algorithm as a benchmark against our own implementations. 
**Linear Regression:** A baseline regression algorithm used for comparison.
**Decision Tree Regression:** A non-linear regression algorithm that uses a decision tree to partition the data.
**Random Forest Regression:** An ensemble learning method that constructs multiple decision trees and merges their predictions to improve accuracy.
**k Nearest Neighbors (KNN) Regression:** A non-parametric method used for regression tasks, which predicts the value of a data point by averaging the values of its k nearest neighbors.
## **Model Evaluation**
We evaluated the performance of each regression algorithm using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R2) score on both the training and testing datasets. 
