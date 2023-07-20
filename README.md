# Fake News Detection using NLP Techniques

## Overview

Fake news has become a pervasive issue in today's digital age, making it challenging to distinguish between reliable and fabricated information. This project aims to address this problem by leveraging Natural Language Processing (NLP) techniques to detect fake text news. NLP is a subfield of artificial intelligence that focuses on the interaction between computers and human language. By applying NLP techniques, we can extract meaningful insights from textual data and build machine learning models to classify news articles as either real or fake.

This repository contains the code and resources necessary to implement and deploy a Flask web application that utilizes a Decision Tree model for fake news detection. The project structure is as follows:

```
- Static: This folder contains CSS styles and images for the Flask web application that implements the ML prediction model.
- Templates: This folder contains the index.html file for the Flask web application.
- DT_model.pkl: The Decision Tree machine learning model with the highest accuracy serialized using the pickle module.
- ML_application_data_visualization.ipynb: A Jupyter notebook that implements various NLP techniques and provides visualizations to explore the data and understand its content during the model training process.
- app.py: The Python Flask web app file.
- articles_examples_fake_real_news.csv: A dataset containing examples of news articles from the test dataset to obtain predictions and test the application.
- vectorizer.pkl: The TF-IDF vectorizer used for word normalization when converting text to understandable data by the machine.
```

## Data

This project uses a dataset of news articles labeled as fake or real. The dataset is split into a training set and a test set. The training set is used to train the machine learning model, while the test set is used to evaluate the performance of the model.

The data is preprocessed using NLP techniques such as text preprocessing, tokenization, and vectorization before being fed into the machine learning model.

## Evaluation

The performance of the machine learning model is evaluated using metrics such as accuracy, precision, recall, and F1-score. These metrics provide insight into how well the model is able to correctly classify news articles as fake or real.


## NLP Techniques and Workflow

### 1. Data Exploration and Preprocessing

The project begins with data exploration and preprocessing to prepare the dataset for training the model. The `ML_application_data_visualization.ipynb` notebook provides a step-by-step walkthrough of this process. Some of the techniques involved are:

- Loading the dataset: Reading the news articles dataset, which includes both real and fake news samples.
- Exploratory Data Analysis (EDA): Analyzing the dataset to gain insights into its structure, distribution, and characteristics.
- Text Preprocessing: Applying techniques such as tokenization, removing stopwords, stemming, and lemmatization to clean the text data.
- Data Visualization: Visualizing the preprocessed data using various plots and graphs to gain a better understanding of the dataset.

### 2. NLP Techniques

In this project, we use several NLP techniques to detect fake news. These techniques include:

- **Text Preprocessing**: This involves cleaning and normalizing the text data, such as removing punctuation, converting all text to lowercase, and removing stopwords.
- **Tokenization**: This involves breaking down the text into individual words or tokens.
- **Vectorization**: This involves converting text data into numerical data that can be understood by machine learning algorithms. In this project, we use the TF-IDF vectorizer to achieve this.
- **Model Training**: We train a Decision Tree machine learning model on preprocessed and vectorized text data to classify news articles as fake or real.

  
### 3. Feature Extraction: TF-IDF Vectorization

To enable machine learning models to process textual data effectively, the text documents need to be converted into a numerical representation. In this project, the TF-IDF (Term Frequency-Inverse Document Frequency) technique is used for feature extraction. TF-IDF assigns weights to words based on their frequency in a document and their rarity in the entire corpus. The `vectorizer.pkl` file contains the serialized TF-IDF vectorizer used for converting text into machine-readable format.

### 4. Model Training: Decision Tree Classification

The Decision Tree model is a popular choice for text classification tasks. In this project, a Decision Tree classifier is trained on the preprocessed and vectorized dataset. The `DT_model.pkl` file contains the serialized Decision Tree model with the highest accuracy achieved during training.

### 5. Flask Web Application

The Flask web application, implemented in `app.py`, allows users to input news articles and receive predictions regarding their authenticity. The web application utilizes the trained Decision Tree model to classify the input text as either real or fake. The `index.html` file in the `Templates` folder provides the structure and design of the web application, while the `Static` folder contains CSS styles and images to enhance the user interface.

#### User Interface

This user interface allows you to input the title and content of a news article, and then predicts whether the news is fake or real using our trained model.

#### Usage

1. Open the user interface in your web browser.
2. Enter the title and content of the news article you want to analyze.
3. Click on the "Predict" button to get the prediction result.

#### Interface Overview

The user interface is designed with simplicity in mind. It consists of the following components:

- **Title Input:** Enter the title of the news article in this input field.

- **Content Input:** Enter the content of the news article in this input field.

- **Predict Button:** Click on this button to trigger the prediction process.

- **Prediction Result:** Once you click on the "Predict" button, the interface will display the prediction result, indicating whether the news is fake or real.<br><br>

  <img width="800" alt="index_interface" src="https://github.com/AITYOUB-Abdelmoughit/fake-news-detection-using-nlp-technics/assets/94485789/c77c7701-adb3-475f-bf6c-ef93429d54e1">


#### Prediction Results

The model uses a machine learning algorithm trained on a labeled dataset of fake and real news articles. The prediction result will be one of the following:<br><br>

- **Fake News:** The model predicts that the provided news article is likely to be fake.<br><br>
<div style="display: flex; justify-content: center;">
  <img alt="fake_news_query" src="https://github.com/AITYOUB-Abdelmoughit/fake-news-detection-using-nlp-technics/assets/94485789/97f6295e-ba87-48ef-9933-0852d7b74bf9" style="width: 450px;"><br><br>
  <img alt="fake_new_prediction" src="https://github.com/AITYOUB-Abdelmoughit/fake-news-detection-using-nlp-technics/assets/94485789/8b366f3c-27d5-4eed-aeb6-845b61513886" style="width: 450px;">
<div><br><br>

- **Real News:** The model predicts that the provided news article is likely to be real.<br><br>
<div style="display: flex; justify-content: center;">
  <img alt="real_news_query" src="https://github.com/AITYOUB-Abdelmoughit/fake-news-detection-using-nlp-technics/assets/94485789/974f79d3-17d5-414d-8547-5fcf582b75cb" style="width: 450px;"><br><br>
  <img alt="real_news_prediction_clean_text" src="https://github.com/AITYOUB-Abdelmoughit/fake-news-detection-using-nlp-technics/assets/94485789/ceb64d88-7a52-4146-b3d2-d09d03305c13" style="width: 450px;">
<div>
<br><br>
Please note that while the model is trained to make accurate predictions, it may not be 100% accurate. Always exercise critical thinking and consider multiple sources before forming opinions based on news articles.

### 6. Testing and Deployment

To evaluate the performance of the model and ensure its effectiveness, a set of news articles is provided in the `articles_examples_fake_real_news.csv` file. These examples can be used to test the application and verify its accuracy in detecting fake news. Once the application has been thoroughly tested, it can be deployed to a server or cloud platform to make it accessible to users.

## Usage

To run the Flask web application locally, follow these steps:

1. Ensure you have Python installed (preferably Python 3.7 or higher).
2. Install the required dependencies.
3. Run the Flask application using the command `python app.py`.
4. Access the application in your web browser by visiting `http://localhost:5000`.

Once the application is running, enter a news article into the input field, and the application will provide a prediction indicating whether the article is real or fake.

## Conclusion

This project demonstrates the power of NLP techniques in detecting fake news. By leveraging text preprocessing, feature extraction, and machine learning algorithms such as Decision Trees, we can build effective models for identifying deceptive information. The Flask web application makes it easy for users to interact with the model and receive predictions in real-time. As fake news continues to be a significant concern, the application of NLP techniques is crucial in combating its spread and ensuring the dissemination of accurate information.

For further improvements, additional NLP techniques like word embeddings or advanced models such as recurrent neural networks (RNNs) or transformer-based models like BERT can be explored. Additionally, expanding the training dataset and regularly updating the model with new data will enhance its accuracy and generalization capabilities.

Feel free to contribute to this project by expanding the dataset, implementing new NLP techniques, or improving the web application's design and functionality. Together, we can combat fake news and promote a more informed and trustworthy digital landscape.
