# Hate Speech Detection Web App

This project focuses on applying Machine Learning techniques to categorize a piece of text into three distinct categories, which are "hate speech", "offensive language" and "neither".

The predictive model is then deployed in a Web App, allowing users to enter any text they please in order to get a prediction about its category. 

The app features an interpretation schema that aims to explain the given prediction applying the LIME method, that utilizes white box models in order to interpret decisions of black box models.

The app is hosted by HEROKU and can be found at: 
 
 Dataset originates from the paper cited below and can be found at: https://github.com/t-davidson/hate-speech-and-offensive-language. 
 
 @inproceedings{hateoffensive,
  title = {Automated Hate Speech Detection and the Problem of Offensive Language},
  author = {Davidson, Thomas and Warmsley, Dana and Macy, Michael and Weber, Ingmar}, 
  booktitle = {Proceedings of the 11th International AAAI Conference on Web and Social Media},
  series = {ICWSM '17},
  year = {2017},
  location = {Montreal, Canada},
  pages = {512-515}
  }
