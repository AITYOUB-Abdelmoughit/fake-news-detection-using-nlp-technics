#importing the needed libs
import numpy as np
import pandas as pd
from flask import Flask,request,jsonify,render_template
import pickle,nltk,string,re
from nltk.corpus import stopwords
from random import randrange

#***********************************************************************#

#let's create a flask app
app=Flask(__name__)

#the flask app varibales
news_title=""
news_text=""
color="black"


#***********************************************************************#


#stopward set
nltk.download('stopwords')
nltk.download('tagsets')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
stop = set(stopwords.words('english'))

#les punctuations
punctuations=list(string.punctuation)

#loading the example dataset
data=pd.read_csv("articles_exemples_fake_real_news.csv")

#load the pickle model
DT_model=pickle.load(open("DT_model.pkl",'rb'))

#loading the count vectorizer to transform the text into numeric form
count_vectorizer=pickle.load(open("vectorizer.pkl",'rb'))

#***********************************************************************#

#remove extra white spaces
def remove_extra_spaces(text):
  return str(text).strip()

#lower case
def lower_case(text):
  return str(text).lower()

#punctuation removal
def remove_punctuation(text):
  text="".join(car for car in text if car not in punctuations)
  return str(text)

#removing the urls
def remove_urls(text):
    return re.sub(r'http\S+', '', text)

# Removing the stopwords from text
def remove_stopwords(text):
    final_text = []
    for word in str(text).split():
        if word.strip().lower() not in stop:
            final_text.append(word.strip())
    return " ".join(final_text)

def pos_tagging(text):
  # Tokenize the article
  tokens = nltk.word_tokenize(text)

  # Perform part-of-speech tagging
  pos_tags = nltk.pos_tag(tokens)
  text=""
  for token, pos in pos_tags:
    text+=token+"-"+pos+" "
  return str(text)

#clean text function
def text_cleaned(text):
  text=remove_urls(text)
  text=remove_stopwords(text)
  text=remove_punctuation(text)
  text=pos_tagging(text)
  text=remove_extra_spaces(text)
  text=lower_case(text)
  return text

#***********************************************************************#

#this functoin will return the predicted valued for our target variable which is 0 or 1
def predictor(to_predict_list):
  to_predict = np.array(to_predict_list)
  #the result of prediction
  result =DT_model.predict(to_predict)
  print(result)
  return result[0]

@app.route("/")
def Home():
	return render_template("index.html")


@app.route("/example", methods=["GET"])
def example():
    index=randrange(0,len(data)-1,1)
    news_title=data.loc[index]["title"]
    news_text=data.loc[index]["text"]
    return render_template("index.html",news_title=news_title,news_text=news_text)

@app.route("/predict",methods=["POST"])
def predict():
  #on clicking on the predict button
  if request.method=='POST' :
    #getting the title from the form
    title=request.form["news_title"]
  
    #getting the text from the form
    text=request.form["news_text"]
  
    #concatinating the title with the text
    article_content=title+" "+text

    #cleaning the article_content
    article_content=text_cleaned(article_content)

    #saving the result into a list
    to_predict_list=[article_content]

    #converting the text into a numeric vector
    to_predict_list=count_vectorizer.transform(to_predict_list).toarray()

    #getting the result of prediction
    result=predictor(to_predict_list)
    
    #checking the result if 1 then the news are fake else then the news are real depending on the given text
    if int(result)==1:
        color="#fa2e1f"
        prediction=' Prediction: fake news ! '
    else:
        color="#76ee3a"
        prediction=" Prediction: real news ! "
  return render_template("index.html",clean_text_label="The clean article content:",clean_text=article_content, prediction = prediction,p_color=color,news_title=news_title,news_text=news_text)


#***********************************************************************#


if __name__=="__main__":
  app.run(debug=True)