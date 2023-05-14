# Hate Speech Detection Web App

This project focuses on applying Machine Learning techniques to categorize a piece of text into three distinct categories, which are "hate speech", "offensive language" and "neither".

The predictive model is then deployed in a Web App, allowing users to enter any text they please in order to get a prediction about its category. 

The app features an interpretation schema that aims to explain the given prediction applying the LIME method, that utilizes white box models in order to interpret decisions of black box models.

### Online version:
The app was hosted by HEROKU on a free plan, but unfortunately the plan is not available anymore.

### To DO:
- [x] Train one-vs-rest model (1 model for each label -> select the most probable prediction as the final one)
- [x] Apply above schema utilizing Logistic Regression with L2 regularization (as proposed by the authors of cited paper)
 
 ### References:
 Dataset originates from the paper cited below and can be found at: https://github.com/t-davidson/hate-speech-and-offensive-language. 
 
 Davidson, T., Warmsley, D., Macy, M. and Weber, I., 2017. Automated Hate Speech Detection and the Problem of Offensive Language. ArXiv. https://arxiv.org/pdf/1703.04009.pdf
