#!/usr/bin/env python
# coding: utf-8

# # 📰 News Classification using spaCy Word Vectors & Machine Learning
# 
# This project focuses on building a machine learning model that can automatically classify news articles as **Fake** or **Real**. With the increasing spread of misinformation across digital platforms, developing automated systems that can detect and filter fake news has become essential.  
#  
# To achieve this, the project uses:
# 
# - **spaCy’s `en_core_web_lg` word vectors** to convert text into meaningful numerical embeddings.
# - **Classical ML models** such as Multinomial Naive Bayes, K-Nearest Neighbors, and Random Forest for classification.
# - **Evaluation metrics and visualizations** such as confusion matrices, F1 score comparison, and ROC curves to analyze model performance.
# 
# This topic was chosen because fake news detection is a highly relevant real-world problem, and combining NLP embeddings with machine learning provides a strong baseline for text classification tasks. The code demonstrates how raw text data can be processed, vectorized, trained, and evaluated end-to-end, making it an excellent learning and portfolio project.
# 

# In[36]:


import pandas as pd
import numpy as np
import spacy


# In[37]:


df = pd.read_csv("fake_real_news.csv")
print(df.shape)
df.head()


# In[38]:


df.info()


# In[39]:


#we don't require 'title','subject','date' these columns as they do not affect whether the news is real or fake.
df = df.drop(columns=['title','subject','date'])
df.head()


# In[40]:


print(df.shape)
df.info()


# In[41]:


df.category.value_counts()


# In[42]:


df['label_num'] = df['category'].map({'fake':0, 'real':1})


# In[43]:


nlp = spacy.load("en_core_web_lg") 


# In[44]:


df['vector'] = df['text'].apply(lambda x : nlp(x).vector)
df.head()


# In[51]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.vector.values,df.label_num,
                                                   test_size = 0.2,
                                                   random_state = 2022,
                                                   stratify = df.label_num)


# In[52]:


# X_train & X_test is an array which also has arrays as its elements , so we have to convert it into a 2d array


# In[53]:


x_train = np.stack(x_train)
x_test = np.stack(x_test)


# In[54]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

clf = MultinomialNB()

#clf.fit(x_train,y_train)


# In[55]:


#naive bayes does not allow negative values so we have to scale using MinMaxScalar 


# In[59]:


from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()


# In[61]:


scaled_train = scalar.fit_transform(x_train)
scaled_test = scalar.transform(x_test)


# In[62]:


clf.fit(scaled_train,y_train)


# In[63]:


y_pred_nb = clf.predict(scaled_test)


# In[64]:


print(classification_report(y_test, y_pred_nb))


# In[71]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(scaled_train,y_train)

y_pred_knn = knn.predict(scaled_test)

print(classification_report(y_test, y_pred_knn))


# In[72]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(scaled_train,y_train)

y_pred_rf = rf.predict(scaled_test)

print(classification_report(y_test, y_pred_rf))


# In[74]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc


# In[75]:


models = {
    "Multinomial NB": y_pred_nb,
    "KNN": y_pred_knn,
    "Random Forest": y_pred_rf
}

for model_name, preds in models.items():
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# In[76]:


from sklearn.metrics import f1_score

f1_scores = {
    "Multinomial NB": f1_score(y_test, y_pred_nb),
    "KNN": f1_score(y_test, y_pred_knn),
    "Random Forest": f1_score(y_test, y_pred_rf)
}

plt.figure(figsize=(7,5))
plt.bar(f1_scores.keys(), f1_scores.values(), color=["#2980b9", "#27ae60", "#8e44ad"])
plt.title("F1 Score Comparison")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
for i, v in enumerate(f1_scores.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
plt.show()


# In[77]:


plt.figure(figsize=(7,6))

# Models
roc_models = {
    "Multinomial NB": y_pred_nb,
    "KNN": y_pred_knn,
    "Random Forest": y_pred_rf
}

for name, preds in roc_models.items():
    fpr, tpr, _ = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0,1], [0,1], 'k--')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()


# # 🏁 Conclusion
# 
# In this project, we developed a complete end-to-end pipeline for classifying news articles as Fake or Real using spaCy word vectors and classical machine learning models. By converting text into dense semantic embeddings and applying multiple classifiers (Multinomial NB, KNN, and Random Forest), we demonstrated how traditional ML techniques can effectively tackle real-world NLP problems.
# 
# The evaluation metrics and visualizations — including confusion matrices, F1 score comparisons, and ROC curves — provided deeper insights into each model’s performance and helped identify the most effective approach for fake news detection.  
# 
# This project highlights the importance of automated misinformation detection and shows how combining modern NLP embeddings with machine learning can serve as a solid baseline for more advanced deep learning or transformer-based models in the future. It also serves as a strong learning example of handling text data, vectorization, model building, and performance analysis in a structured and interpretable manner.
# 

# In[ ]:




