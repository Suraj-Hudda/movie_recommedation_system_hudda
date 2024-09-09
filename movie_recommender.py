#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head()


# In[4]:


movies.shape


# In[5]:


credits.head(2)


# In[6]:


credits.shape


# In[7]:


movies=movies.merge(credits,on='title')


# In[8]:


movies.info()


# In[9]:


movies.original_language.value_counts()


# In[10]:


# genres,id,keywords,overview,title,cast,crew


# In[11]:


movies=movies[['genres','id','keywords','overview','title','cast','crew']]


# In[12]:


movies


# In[13]:


movies.isnull().sum()


# In[14]:


movies.dropna(inplace=True)


# In[15]:


movies.shape


# In[16]:


movies.isnull().sum()


# In[17]:


movies.duplicated().sum()


# In[18]:


movies.loc[0].genres


# In[19]:


# convert string to dict
# '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"},
#  {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
import ast

def extract_tags(genre_string):
    genre_list = ast.literal_eval(genre_string)  # Parse the string into a list of dictionaries
    list = []
    for i in genre_list:
        list.append(i['name'])
    return list

movies['genres'] = movies.genres.apply(extract_tags)



# In[20]:


movies.head(5)


# In[21]:


movies.keywords.iloc[0]


# In[22]:


movies['keywords']=movies.keywords.apply(extract_tags)


# In[23]:


movies.head(2)


# In[24]:


movies.cast.iloc[0]


# In[25]:


import ast

def cast_extraction(cast_string):
    cast_list = ast.literal_eval(cast_string)  # Parse the string into a list of dictionaries
    list = []
    for cast in cast_list[:3]:  # Only take the first three elements (order numbers 0, 1, and 2)
        list.append(cast['name'])
    return list

movies.cast.apply(cast_extraction)


# In[26]:


movies['cast']=movies.cast.apply(cast_extraction)


# In[27]:


movies.head(3)


# In[28]:


print(movies.crew[0])


# In[29]:


import ast
def find_director(crew_string):
    crew_list=ast.literal_eval(crew_string)
    list=[]
    for i in crew_list:
        if i['job']=='Director':
            list.append(i['name'])
    return list


# In[30]:


movies.crew=movies.crew.apply(find_director)


# In[31]:


movies.head(3)


# In[32]:


movies.overview[0]


# In[33]:


movies.overview=movies.overview.apply(lambda x:x.split())


# In[34]:


movies.head(3)


# In[35]:


movies.genres=movies.genres.apply(lambda x:[i.replace(" ","") for i in x])
movies.cast=movies.cast.apply(lambda x:[i.replace(" ","") for i in x])
movies.crew=movies.crew.apply(lambda x:[i.replace(" ","") for i in x])
movies.keywords=movies.keywords.apply(lambda x:[i.replace(" ","") for i in x])


# In[36]:


movies.head(3)


# In[37]:


movies['tags']=movies.genres+movies.keywords+movies.cast+movies.crew+movies.overview


# In[38]:


movies.head(2)


# In[39]:


df=movies[['id','title','tags']]


# In[40]:


df


# In[41]:


df.tags=df.tags.apply(lambda x:" ".join(x))


# In[42]:


df


# In[43]:


df.tags[0]


# In[44]:


df.tags=df.tags.apply(lambda x:x.lower())


# In[45]:


df


# In[46]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(stop_words='english',max_features=5000)


# In[47]:


cv.fit_transform(df['tags'])


# In[48]:


cv.fit_transform(df['tags']).toarray()


# In[49]:


cv.fit_transform(df['tags']).toarray().shape


# In[50]:


vectors=cv.fit_transform(df['tags']).toarray()


# In[51]:


vectors[0]


# In[52]:


cv.get_feature_names_out()


# In[53]:


# nlp library
get_ipython().system('pip install nltk')


# In[54]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[55]:


def stem(text):
    list=[]
    for i in text.split():
       list.append( ps.stem(i))
    return " ".join(list)


# In[56]:


df.tags=df.tags.apply(stem)


# In[57]:


df.head(3)


# In[58]:


from sklearn.metrics.pairwise import cosine_similarity


# In[59]:


cosine_similarity(vectors)


# In[60]:


cosine_similarity(vectors).shape


# In[61]:


similarity=cosine_similarity(vectors)


# In[62]:


similarity


# In[63]:


similarity[0] # first movie ka relationship apne sath toh one he hoga 


# In[64]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[65]:


def recommendation(movie):
    movie_index=df[df.title==movie].index[0]
    distances=similarity[movie_index]
    movie_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

    for i in movie_list:
        print(df.iloc[i[0]].title)
        
    


# In[66]:


df[df.title=='Avatar'].index[0]


# In[67]:


recommendation('Avatar')


# In[68]:


df.iloc[539].title


# In[69]:


import pickle


# In[70]:


pickle.dump(df,open('movies.pkl','wb'))


# In[71]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[72]:


df


# In[73]:


recommendation('Batman')


# In[74]:


pickle.dump(df.to_dict(),open('movie_dict.pkl','wb'))


# In[79]:


recommendation('Lucy')


# In[75]:


# df.to_dict()


# In[ ]:




