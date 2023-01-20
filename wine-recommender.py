#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import random

import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
 

st.set_page_config(layout="wide",page_title='Wine Recommender', page_icon="wine3.png")

header = st.container()
dataset = st.container()
model_training = st.container()
recommender = st.container()

with header:
    st.markdown("<h1 style='text-align: center;'>🍷 Welcome to my Amazing Wine Recommender 🍷 </h1>", unsafe_allow_html=True)

with dataset:
    st.markdown("<h2 style='text-align: center;'>Data set in progress...for now this is the info that we have!</h2>", unsafe_allow_html=True)

    clustered = pd.read_csv('new_clustered.csv')
    varietal = pd.read_csv('new_varietal.csv')
    # st.write(clustered.head())

    col1, col2 = st.columns([1,5])
    
    types = ['All','Red','White','Sparkling','Rosé']
    wine_type = col1.radio('Select a type of wine:',types)
    if wine_type == 'All':
       wine_country = pd.DataFrame(varietal['country'].value_counts())
    else:
        wine_country = pd.DataFrame(varietal[varietal['type']== wine_type]['country'].value_counts())
    
    col2.bar_chart(wine_country,height=400)


with model_training:
    st.header("Let's see what features would we include in the model")
    st.write('Choose the parameters of your desired wine and get your recommendation:')

    col_1, col_2, col_3, col_4 = st.columns(4)
    
    country = ['All', 'Portugal', 'Spain', 'Italy', 'France', 'Argentina', 'Australia',
    'Brazil', 'Chile', 'New-Zealand', 'South-Africa', 'United-States',
    'Israel', 'Germany', 'Switzerland']

    variety =['All','Red', 'Cabernet Sauvignon', 'Garnacha', 'Mencía', 'Merlot',
    'Monastrell', 'Blend del Ródano', 'Rioja', 'Syrah', 'Tempranillo',
    'Amarone', 'Barbaresco', 'Barolo', 'Bolgheri', 'Brunello',
    'Chianti', 'Nebbiolo', 'Pinot Noir', 'Ripasso', 'Primitivo',
    'Barbera', 'Sangiovese', 'Montepulciano', 'Cannonau', 'Albariño',
    'Bonarda', 'Blend de Burdeos', 'White', 'Cabernet', 'Carménère',
    'Chablis', 'Chardonnay', 'Chenin blanc', 'Côte-Rotie', 'Gavi',
    'Gewürztraminer', 'Malbec', 'Müller Thurgau', 'Pinot Blanc',
    'Pinot Gris', 'Grauburgunder', 'Spätburgunder', 'Pinotage',
    'Riesling', 'Sauvignon Blanc', 'Jerez', 'Soave', 'Shiraz',
    'Verdejo', 'Vinho verde', 'Viognier', 'Zinfandel',
    'Cabernet Franc', 'Silvaner', 'Médoc', 'Margaux', 'Pauillac',
    'Pomerol', 'Libourne', 'Torrontés', 'Moscatel', 'Saint-Estèphe',
    'Cote Nuits', 'Cote Beaune', 'Cote Chalonnaise', 'Maconnais',
    'Condrieu', 'Cornas', 'Crozes-Hermitage', 'Hermitage',
    'Saint-Péray', 'Pedro Ximenez', 'Chasselas', 'Vin Jaune',
    'Vino espumoso', 'Dornfelder', 'Asti', 'Cava', 'Champagne',
    "Moscato d'Asti", 'Prosecco', 'Sparkling', 'Sparkling - Rosé',
    'Crémant', 'Rosé']

    grapes = ['All', 'Cabernet Sauvignon','Carménère','Chardonnay','Chasselas',
    'Garnacha','Malbec','Merlot','Pinot Noir','Riesling','Sangiovese',
    'Sauvignon Blanc','Shiraz/Syrah','Spätburgunder','Tempranillo',
    'Tinta Roriz','Touriga Franca','Touriga Nacional','Weissburgunder']

    body = ['All','Very Strong','Strong','Medium','Weak','Very Weak']
    acidity = ['All','High','Medium','Low','Very Low']
    
    body_sel = col_1.radio('Body',body)
    ac_sel = col_2.radio('Acidity',acidity)
    types_wine = col_3.radio('Type',types)
    country_sel = col_4.selectbox('Country',country)
    variety_sel = col_4.selectbox('Variety',variety)
    grapes_sel = st.multiselect('Grapes',grapes)



    # using varietal dataframe
    if (body_sel == 'All') & (ac_sel == 'All'):
        # selecting type
        if types_wine == 'All':
            df = varietal.copy()
        else:
            df = varietal[varietal['type']==types_wine]
        # selecting country
        if country_sel == 'All':
            pass
        else:
            df = df[df['country']==country_sel]
        # selecting variety
        if variety_sel=='All':
            pass
        elif len(df[df['varietal_name']==variety_sel])>0 :
           df = df[df['varietal_name']==variety_sel]
        else:
            no_var = "Sorry, we don't have this variety for this features"
        # selecting grapes:
        if grapes_sel == []:
            pass
        elif len(df[(df['grapes_1_name'].isin(grapes_sel))|(df['grapes_2_name'].isin(grapes_sel))|(df['grapes_3_name'].isin(grapes_sel))])>0:
            df = df[(df['grapes_1_name'].isin(grapes_sel))|(df['grapes_2_name'].isin(grapes_sel))|(df['grapes_3_name'].isin(grapes_sel))]
        else:
            no_grape = "Sorry, we don't have those grapes from that country or variety"
    else:
        # selecting type
        if types_wine == 'All':
            df = clustered.copy()
        else:
            df = clustered[clustered['type']==types_wine]
        # selecting country
        if country_sel == 'All':
            pass
        else:
            df = df[df['country']==country_sel]
        # body and acidity
        if (body_sel!='All')&(ac_sel!='All')&(len(df[(df['body_description']==body_sel)&(df['acidity_description']==ac_sel)])>0):
            df = df[(df['body_description']==body_sel)&(df['acidity_description']==ac_sel)]
        elif (body_sel!='All')&(ac_sel=='All')&(len(df[df['body_description']==body_sel])>0):
            df = df[df['body_description']==body_sel]
        elif (body_sel=='All')&(ac_sel!='All')&(len(df[df['acidity_description']==ac_sel])>0):
            df = df[df['acidity_description']==ac_sel]
        else:
            no_b_ac = "Sorry, we don't have wines with that combination of body and acidity from that country or of that type"
        # selecting variety
        if variety_sel=='All':
            pass
        elif len(df[df['varietal_name']==variety_sel])>0 :
           df = df[df['varietal_name']==variety_sel]
        else:
            no_var = "Sorry, we don't have that variety for this features"
        # selecting grapes:
        if grapes_sel == []:
            pass
        elif len(df[(df['grapes_1_name'].isin(grapes_sel))|(df['grapes_2_name'].isin(grapes_sel))|(df['grapes_3_name'].isin(grapes_sel))])>0:
            df = df[(df['grapes_1_name'].isin(grapes_sel))|(df['grapes_2_name'].isin(grapes_sel))|(df['grapes_3_name'].isin(grapes_sel))]
        else:
            no_grape = "Sorry, we don't have those grapes from that country or variety"
    

    st.header('')
    st.markdown("<h5 style='text-align: center; color: purple;'>More features coming soon...</h5>", unsafe_allow_html=True)

   
with recommender:
    st.header('Recommendations')
    rec_1, rec_2 = st.columns([3,2])

    try:
        rec_2.warning(no_b_ac)
    except:
        no_b_ac = None
    try:
        rec_2.warning(no_var)
    except:
        no_var = None
    try:
        rec_2.warning(no_grape)
    except:
        no_grape = None

    rec_1.write('Number of possible recommendations:   '+str(df.shape[0]))
    rec_1.write(df[['wine_name','type','country','region']])


    # varietal = pd.read_csv('new_varietal.csv')
    #     country = ['All','Portugal', 'Spain', 'Italy', 'France', 'Argentina', 'Australia',
    #    'Brazil', 'Chile', 'New-Zealand', 'South-Africa', 'United-States',
    #    'Israel', 'Germany', 'Georgia', 'Switzerland', 'Romania', 'China','Belgium']

    #    variety = ['All', 'Red', 'Cabernet Sauvignon', 'Garnacha', 'Mencía', 'Merlot',
    #    'Monastrell', 'Blend del Ródano', 'Rioja', 'Syrah', 'Tempranillo',
    #    'Amarone', 'Barbaresco', 'Barolo', 'Bolgheri', 'Brunello',
    #    'Chianti', 'Nebbiolo', 'Pinot Noir', 'Ripasso', 'Primitivo',
    #    'Barbera', 'Sangiovese', 'Montepulciano', 'Cannonau', 'Albariño',
    #    'Bonarda', 'Blend de Burdeos', 'White', 'Cabernet', 'Carménère',
    #    'Chablis', 'Chardonnay', 'Chenin blanc', 'Côte-Rotie', 'Gavi',
    #    'Gewürztraminer', 'Malbec', 'Müller Thurgau', 'Pinot Blanc',
    #    'Pinot Gris', 'Grauburgunder', 'Spätburgunder', 'Pinotage',
    #    'Riesling', 'Sauvignon Blanc', 'Jerez', 'Soave', 'Vino espumoso',
    #    'Shiraz', 'Verdejo', 'Vinho verde', 'Viognier', 'Zinfandel',
    #    'Cabernet Franc', 'Silvaner', 'Médoc', 'Margaux', 'Pauillac',
    #    'Pomerol', 'Libourne', 'Torrontés', 'Moscatel', 'Saint-Estèphe',
    #    'Cote Nuits', 'Cote Beaune', 'Cote Chalonnaise', 'Maconnais',
    #    'Condrieu', 'Cornas', 'Crozes-Hermitage', 'Hermitage',
    #    'Saint-Péray', 'Godello', 'Treixadura', 'Ribeiro', 'Pedro Ximenez',
    #    'Chasselas', 'Vin Jaune', 'Dornfelder', 'Rosé', 'Sparkling',
    #    'Sparkling - Rosé', 'Asti', 'Cava', 'Champagne', "Moscato d'Asti",
    #    'Prosecco', 'Crémant', 'Sekt', "Cerasuolo d'Abruzzo",'Pinot Noir Rosé']

    #    grapes = ['All','Cabernet Sauvignon','Carménère','Chardonnay','Chasselas','Garnacha',
    #     'Malbec','Merlot','Pinot Noir','Riesling','Sangiovese','Sauvignon Blanc',
    #     'Shiraz/Syrah','Spätburgunder','Tempranillo','Tinta Roriz','Touriga Franca','Touriga Nacional','Weissburgunder',]

    

