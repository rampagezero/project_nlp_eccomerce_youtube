import pandas as pd
import plotly.express as px
from pymongo import MongoClient
import streamlit as st
hasil=pd.read_csv('streamlit/hasil.csv')
f=open('streamlit/password_mongo.txt','r')
import plotly.express as px


df = pd.DataFrame(hasil.merk.value_counts()).reset_index()
fig7 = px.pie(df, values=df['merk'], names=df['index'])


temp=hasil.groupby(['merk']).mean()['hasil'].reset_index()
fig1=px.bar(temp,x=temp['merk'],y=temp['hasil'],title='Average Sentiment Rating Based on Handphone Brand Youtube Comment',color='merk')
fig1.update_layout(barmode='stack', xaxis={'categoryorder':'total descending'})
hasil['tanggal']=pd.to_datetime(hasil['tanggal'])
from plotly import graph_objects as go
timeline=hasil.groupby([hasil['tanggal'].dt.year,'merk']).mean().reset_index()
fig2=px.line(timeline,'tanggal','hasil',color='merk')
fig2.update_layout(title='Average Brand Sentiment Rating From Youtube Comment Year on Year')
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# Create and generate a word cloud image:
fig3= WordCloud(max_font_size=50, max_words=100, background_color="white").generate(','.join(hasil['komentar'].to_list()))




j=f.readlines()[0]
client=MongoClient(f"mongodb+srv://andikakristianto95:{j}@cluster0.nnjfxtd.mongodb.net/?retryWrites=true&w=majority")
client.list_database_names()
db=client['hasil_project']
collection2=db['data_gather_tokopedia_penjualan']
df2=pd.DataFrame.from_dict(collection2.find())

def tambah_merk(s):
  handphone=['xiaomi','samsung','iphone','oppo','vivo','infinix','sony','lg','nokia','redmi']
  for i in handphone:
    if s.__contains__(i):
      return i
df2['merk']=df2['merk'].apply(tambah_merk)
pengganti=df2[df2['jumlah_terjual'].str.contains('rb')]['jumlah_terjual'].apply(lambda x:int(x.split()[1])*1000)
series={}
for i in pengganti.index:
  series[i]=pengganti[i]
series1=pd.Series(series)
series2=df2[~df2.index.isin(series.keys())]['jumlah_terjual'].apply(lambda x:x.split()[1]).apply(lambda x:x.replace('+',''))


df2['jumlah_terjual']=series1.append(series2).sort_index()

df2=df2.dropna().reset_index(drop=True)
df2['jumlah_terjual']=df2['jumlah_terjual'].astype('int')

y=df2.groupby('merk').mean()[['jumlah_terjual']].reset_index()

data_korelasi=pd.merge(y,temp,on='merk')
fig4=px.scatter(data_korelasi,'hasil','jumlah_terjual',trendline='ols',color='merk',trendline_scope = 'overall',
                 trendline_color_override = '#6074A1')
fig4.update_layout(title='Correlation between Selling and Rating NLP LSTM Based Youtube Comment')

df2['harga']=df2['harga'].apply(lambda x:x.replace('Rp','').replace('.','')).astype('int')

harga=df2.groupby('merk').mean()[['harga']].reset_index().sort_values('harga')
harga=pd.merge(harga,temp,on='merk')
fig5=px.bar(harga,x='merk',y='harga',color='merk')
fig5.update_layout(title='Average Handphone Price Based On Tokopedia')
fig6=px.scatter(harga,x='hasil',y='harga',trendline='ols',color='merk',trendline_scope = 'overall',
                 trendline_color_override = '#6074A1')
fig6.update_layout(title='Correlation between Price and Rating Youtube Based NLP')
st.set_page_config(page_title='E Commerce and Youtube Comment',layout='wide')
df_acc=pd.read_csv('streamlit/Accuracy_NLP.csv',sep=';')
fig_acc=px.bar(df_acc,x=df_acc['Accuracy'],y=df_acc['Model'],color=df_acc['Model'])
col_utama, col_kedua,col_ketiga=st.tabs(['Dashboard','Predictor','Komparasi Model'])
with col_utama:
  st.title('E Commerce and Youtube Comment')
  st.write("Dashboard")
  with st.container():
    col1,col2=st.columns([3,1])
    with col1:
      st.plotly_chart(fig1,use_container_width=True,)
    with col2:
      st.subheader('Brand Frequency Proportion Based On Youtube Comment')
      st.plotly_chart(fig7,use_container_width=True)
    
  with st.container():
    col1,col2=st.tabs(['Trendline Rating','Average Price'])
    with col1:
      st.plotly_chart(fig2,use_container_width=True)
    with col2:
      st.plotly_chart(fig5,use_container_width=True)
      
  with st.container():
    col1,col2=st.columns(2)
    with col1:
      st.subheader('Youtube Comment on Handphone Product WordCloud')
      fig, ax = plt.subplots(figsize = (12, 8))
      ax.imshow(fig3)
      plt.axis("off")
      st.pyplot(fig)
    with col2:
      with st.container():
        col1,col2=st.tabs(['col 1','col 2'])
        with col1:
          st.plotly_chart(fig6,use_container_width=True)
        with col2:
          st.plotly_chart(fig4,use_container_width=True)
  with st.container():
      st.subheader('Model Accuracy')
      st.plotly_chart(fig_acc,use_container_width=True)
with col_kedua:
  import tensorflow as tf
  from tensorflow import keras
  import streamlit as st
  import pickle
  import keras.preprocessing as keras_preprcessing
  
  model_gru=keras.models.load_model('streamlit/model_lstm_hasil_tokopedia_baru (1)')

  
  # model_gru.summary()

  import pickle
  with open("streamlit/tokenizer (1).json", 'rb') as handle:
        b = handle.read()
  b=tf.keras.preprocessing.text.tokenizer_from_json(b)



  st.title('TextReview-Rating Predictor')
  text=st.text_input('Masukan Review Disini')
  def predict(text):
      text=str(text)
      text=[text]
      token_matrix=b.texts_to_sequences(text)
      token_sequence=keras.preprocessing.sequence.pad_sequences(token_matrix,padding='post',maxlen=365)
      pre=model_gru.predict(token_sequence)
      global x1
      for i,j in enumerate(pre.T,1):
        if j==pre.max():
          x1=(f'Rating:{i}',f'Prob:{str(j)}')
      return x1
  if st.button('Predict Rating'):
      hasil=predict(text)
      st.write(x1)
with col_ketiga:
  data_akurasi_twitter=pd.read_csv('streamlit/Akurasi Analisis Twitter.csv',sep=';')
  data_akurasi_algoritma=pd.read_csv('streamlit/Akurasi Komparasi Algoritma.csv',sep=';')
  fig_akurasi_twitter=px.line(data_akurasi_twitter,x=data_akurasi_twitter['model'],y=data_akurasi_twitter['akurasi'],markers=True)
  fig_akurasi_twitter.update_layout(title='Algoritma Akurasi Twitter')
  fig_akurasi_algoritma=px.line(data_akurasi_algoritma,x=data_akurasi_algoritma['model'],y=data_akurasi_algoritma['akurasi'],markers=True)
  fig_akurasi_twitter.update_layout(title='Algoritma Akurasi Algoritma Comment E Commerce')
  with st.container():
    st.plotly_chart(fig_akurasi_twitter)
  with st.container():
    st.plotly_chart(fig_akurasi_algoritma)
      






      



    

