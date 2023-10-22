#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# In[5]:


print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))


# In[6]:


import streamlit as st


# In[7]:


df = pd.read_csv("C:\mgtu\ebw_data.csv")


# In[8]:


st.sidebar.header('Введите значения необходимые для расчета параметров сварного шва')


# In[9]:


def user_input_param():
    IW = st.sidebar.number_input('Величина сварочного тока(IW)')
    IF = st.sidebar.number_input('Ток фокусировки электронного пучка (IF)')
    VW = st.sidebar.number_input('Скорость сварки (VW)')
    FP = st.sidebar.number_input('Расстояние от поверхности образцов до электронно-оптической системы (FP)')
    data = {'IW': IW,
            'IF': IF,
            'VW': VW,
            'FP': FP}
    param = pd.DataFrame(data, index=[0])
    return param


# In[10]:


df_one = user_input_param()


# In[11]:


X = df.drop(["Width", "Depth"], axis=1)
y = df[["Width", "Depth"]].copy()


# In[12]:


X1_train, X1_test, y1_train, y1_test = train_test_split(X, y,
                                                    train_size=0.65,
                                                    test_size=0.35,
                                                    random_state=5000,
                                                    shuffle=True)


# In[13]:


RanForReg = RandomForestRegressor(bootstrap=False, criterion='friedman_mse', max_depth=7,
                      max_features='log2', n_estimators=51)
RanForReg.fit(X1_train, y1_train)


# In[14]:


y4_pred = RanForReg.predict(X1_test)


# In[15]:


prediction = RanForReg.predict(df_one)


# In[16]:


st.write("""
# Результат прогнозирования параметров сварного шва
""")


# In[17]:


st.subheader('Вы ввели вот это:')
st.write(df_one)


# In[18]:


st.subheader('Значение ширины шва')
st.write(prediction[:,[0]])


# In[19]:


st.subheader('Значение глубины шва')
st.write(prediction[:,[1]])


# In[20]:


st.subheader('Точность прогноза 95%')


# In[ ]:




