import pandas as pd
import plotly.express as px
from pymongo import MongoClient
import streamlit as st
hasil=pd.read_csv('streamlit/hasil.csv')
f=open('streamlit/password_mongo.txt','r')
import plotly.express as px
from PIL import Image

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
col_utama, col_kedua,col_ketiga,col_empat=st.tabs(['Dashboard','Predictor','Model Comparison  JD.ID','Model Comparison E Commerce'])
with col_utama:
  st.title('E Commerce and Youtube Comment')
  st.write("Dashboard")
  with st.container():
    col1,col2=st.columns([3,1])
    with col1:
      st.plotly_chart(fig1,use_container_width=True,)
      st.write('Sentimen penilaian yang didapat dari penilaian handphone berasal dari tautan komentar yang tersedia pada Channel Youtube tech reviewer. Setiap komentar yang terdapat pada database diambil menggunakan youtube open data API. Sebelum menganalisa sentiment pada brand, pembuatan model dilakukan dengan menggunakan deep learning tensorflow dengan menggunakan data komentator review beserta rating yang dibangun pada Tokopedia dan Lazada. Beberapa algoritma yang dipilih untuk menjadi sentiment rating predictor diantaranya LSTM, GRU , Random Forest serta Decision Tree. Setelah model sudah terbuat maka setiap komen akan dipilah berdasarkan brandnya lalu diprediksi nilai sentiment yang muncul pada brand average rating graphic diatas. Dapat terlihat bahwa Xiaomi merupakan brand yang memiliki tingkat sentiment paling baik diantara brand brand lain diikuti oleh Vivo dan Sony. ')
    with col2:
      st.subheader('Brand Frequency Proportion Based On Youtube Comment')
      st.plotly_chart(fig7,use_container_width=True)
      st.write('Melalui perhitungan data komentar yang terdapat pada youtube didapatkan brand Samsung merupakan brand yang paling sering di bicarakan pada komentar youtube dengan proporsi 25 % dari total komentar yang tersedia.')
      
    
  with st.container():
    col1,col2=st.tabs(['Trendline Rating','Average Price'])
    with col1:
      st.plotly_chart(fig2,use_container_width=True)
      st.write('Pada graphic trendline dapat terlihat sentiment rating dari waktu ke waktu. Pengambilan data youtube didapatkan dari tahun 2014 hingga video terbaru 2022. Data ini didapatkan dari API data youtube lalu diproses menggunakan NLP model untuk mendapatkan penilaian sentimentnya. Setiap brand dihitung rata rata per tahunnya sehingga didapatkan data diatas. Pada tahun 2023 setiap brand memiliki penurunan sentiment kecuali pada brand Redmi dan Vivo.')
      
    with col2:
      st.plotly_chart(fig5,use_container_width=True)
      st.write('Pada graphic diatas merupakan graphic rata rata harga yang ditawarkan setiap produk pada brand brand yang tersedia. Dapat terlihat bahwa Iphone memiliki rata rata harga tertinggi, disusul dengan brand LG lalu Samsung. ')
      
  with st.container():
    col1,col2=st.columns(2)
    with col1:
      st.subheader('Youtube Comment on Handphone Product WordCloud')
      fig, ax = plt.subplots(figsize = (12, 8))
      ax.imshow(fig3)
      plt.axis("off")
      st.pyplot(fig)
      st.write('Pada graphic wordcloud diatas terdapat sebagian besar kata kata yang terkandung pada komentar youtube. Setiap kata kata yang sering bermunculan akan memiliki ukuran yang besar dan juga sebaliknya untuk kata kata yang jarang bermunculan.')
      
    with col2:
      with st.container():
        col1,col2=st.tabs(['col 1','col 2'])
        with col1:
          st.plotly_chart(fig6,use_container_width=True)
          st.write('Pada graphic regresi diatas merupakan hasil penggambaran hubungan antara harga masing masing brand dengan sentiment rating NLP based pada komentar yang tersedia pada youtube. Dapat terlihat pada gambar semakin tinggi rating semakin rendah harga yang dimiliki. Dapat dikatakan bahwa setiap pelanggan memiliki kecendrungan untuk memberikan komntar positif pada barang barang yang cukup murah sehingga tidak perlu untuk membeli barang dengan harga mahal untuk mendapatkan barang dengan kualitas baik. Brand uang memiliki nilai harga terendah dengan penilaiain yang cukup baik adalah pada brand Xiaomi. ')
          
        with col2:
          st.plotly_chart(fig4,use_container_width=True)
          st.write('Pada graphic selanjutnya terdapat gambar yang menunjukan korelasi antara jumlah penjualan yang terjual pada Tokopedia serta korelasinya dengan hasil penilaian sentiment NLP LSTM pada komentar youtube. Dapat terlihat bahwa Xiaomi menunjukan jumlah rating yang paling tinggi, lalu disusul dengan vivo dengan jumlah penjualan tertinggi dengan rating yang cukup baik. ')
          
  with st.container():
      st.subheader('Model Accuracy')
      st.plotly_chart(fig_acc,use_container_width=True)
      st.write('Pada pembuatan model NLP (Natural Language Processing) digunakan 2 data komentar serta rating diantaranya eCommerce Tokopedia dan Lazada. Kedua data tersebut akan dijadikan bahan untuk membuat model dengan algo4tima yang sama sehingga nantinya dapat dipilih algortima serta sumber data yang memiliki accuracy terbaik. Melalui graphic diatas dapat dilihat bahwa algoritma LSTM dengan data Tokopedia memiliki model dengan akurasi terbaik. Oleh karena itu model dengan data train Tokopedia akan digunakan untuk memprediksi sentiment yang terdapat pada komentar youtube. ')
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
  df_komparasi=pd.read_csv('sentiment_ecommerce.csv',sep=";")
  df_komparasi_fix=df_komparasi.melt(id_vars=df_komparasi.iloc[:,0:1],var_name=['ecommerce'],value_vars=df_komparasi.loc[:,["Shopee","Tokopedia","Bukalapak","Lazada"]])
  fig_komparasi_sentiment=px.line(df_komparasi_fix,x=df_komparasi_fix.iloc[:,0],color='ecommerce',y='value',markers='o',text='value')
  fig_komparasi_sentiment.update_traces(textposition="top right")
  fig_komparasi_sentiment.update_layout(
  autosize=False,
  width=1200,
  height=500,xaxis_title='Trisemester (x)',yaxis_title='Sentiment Negative (y)')
  fig_komparasi_sentiment.update_layout(title='E Commerce Sentiment Trend 2020-2022')
  data_akurasi_twitter=pd.read_csv('streamlit/Akurasi Analisis Twitter.csv',sep=';')
  data_akurasi_algoritma=pd.read_csv('streamlit/Akurasi Komparasi Algoritma.csv',sep=';')
  fig_akurasi_twitter=px.line(data_akurasi_twitter,x=data_akurasi_twitter['model'],y=data_akurasi_twitter['akurasi'],markers=True)
  fig_akurasi_twitter.update_layout(title='Accuracy algorithm comparison on buyer comments on JD.ID e-commerce accounts',xaxis_title="Model", yaxis_title="Accuracy")
  fig_akurasi_algoritma=px.line(data_akurasi_algoritma,x=data_akurasi_algoritma['model'],y=data_akurasi_algoritma['akurasi'],markers=True)
  fig_akurasi_algoritma.update_layout(title='Accuracy algorithm comparison on buyer comments on Tokopedia,buka lapak,shopee <br> and lazada accounts e-commerce',xaxis_title="Model", yaxis_title="Accuracy")
  df_proporsi_komen=pd.read_csv('Review.csv',sep=';')
  fig_proporsi_komen=px.bar(df_proporsi_komen,x='Source',y='Review',color='Sentimen',barmode='group',color_discrete_map={"Negativ":"Red","Positiv":"Green"})
  fig_proporsi_komen.update_layout(title='Comment Sentiment Proportion')
  wordcloud=Image.open('download.png')
  df_jd_id=pd.read_csv('proporsi_jd_id.csv',sep=';')
  fig_jd_id_prop=px.pie(df_jd_id,values="Label",names='Sentiment',color='Sentiment',color_discrete_map={"Negative":"Red","Positive":"Green","Neutral":"Blue"})
  fig_jd_id_prop.update_layout(font_size=20)
  with st.container():
    st.plotly_chart(fig_akurasi_twitter)
  with st.container():
    st.title("Wordloud JD.ID Reviews")
    im=wordcloud.resize(300,200)
    st.image(im)
  with st.container():
    st.title("Proportion JD ID Sentiment Comment")
    st.plotly_chart(fig_jd_id_prop)
    
with col_empat:
  with st.container():
    st.plotly_chart(fig_proporsi_komen)
  with st.container():
    st.plotly_chart(fig_akurasi_algoritma)
  with st.container():
    st.plotly_chart(fig_komparasi_sentiment)
    
      






      



    

