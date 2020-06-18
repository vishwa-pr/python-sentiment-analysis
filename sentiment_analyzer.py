import pandas as pd
import json
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import flair


#flair model will be downloaded first time . After download change to downloaded path
flair_sentiment = flair.models.TextClassifier.load('en-sentiment')


#text list
text_list = ["It was a great movie.I enjoyed it a lot",
             "The movie was horrible.0 stars from me",
             "It's a good day to watch a movie"]


nltk_list=[]
tb_list=[]
flair_list=[]
round_off_val =4

for text in text_list:

    #nltk sentiment
    sid = SentimentIntensityAnalyzer()
    nltk_dict = sid.polarity_scores(text)
    print(sid.polarity_scores(text))
    del nltk_dict['compound']
    max_senti_score = max(nltk_dict, key=nltk_dict.get)
    if max_senti_score == "pos":
      nltk_list.append("positive ("+str(round(nltk_dict[max_senti_score],round_off_val))+")")
    elif max_senti_score == "neu":
      nltk_list.append("neutral ("+str(round(nltk_dict[max_senti_score],round_off_val))+")")
    elif max_senti_score == "neg":
      nltk_list.append("negative ("+str(round(nltk_dict[max_senti_score],round_off_val))+")")



    #text blob sentiment
    tb_polarity = TextBlob(text).sentiment[0]

    if tb_polarity == 0:
        tb_list.append('neutral '+"("+str(round(tb_polarity,round_off_val))+")")
    elif tb_polarity < 0:
        tb_list.append('negative '+"("+str(round(tb_polarity,round_off_val))+")")
    elif tb_polarity > 0:
        tb_list.append('positive '+"("+str(round(tb_polarity,round_off_val))+")")

    #flair sentiment
    s = flair.data.Sentence(text)
    flair_sentiment.predict(s)
    total_sentiment = s.labels
    flair_list.append(str(total_sentiment[0]).lower())

#save to dataframe
df = pd.DataFrame({'Text': text_list, 'nltk': nltk_list,'textblob': tb_list,'flair': flair_list})
print(df)
df.to_csv('multi-sentiment.csv', index=False, encoding='utf-8')


