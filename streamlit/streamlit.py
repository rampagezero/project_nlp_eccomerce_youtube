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
fig1=px.bar(temp,x=temp['merk'],y=temp['hasil'],title='Rata Rata Sentimen Penilaian Data Youtube')
hasil['tanggal']=pd.to_datetime(hasil['tanggal'])
from plotly import graph_objects as go
timeline=hasil.groupby([hasil['tanggal'].dt.year,'merk']).mean().reset_index()
fig2=px.line(timeline,'tanggal','hasil',color='merk')
fig2.update_layout(title='Rata Rata Penilaian Merk Handphone Dari Waktu ke Waktu')
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
fig4=px.scatter(data_korelasi,'hasil','jumlah_terjual',trendline='ols')
fig4.update_layout(title='Correlation between Selling and Rating NLP LSTM Based Youtube Comment')

df2['harga']=df2['harga'].apply(lambda x:x.replace('Rp','').replace('.','')).astype('int')

harga=df2.groupby('merk').mean()[['harga']].reset_index().sort_values('harga')
harga=pd.merge(harga,temp,on='merk')
fig5=px.bar(harga,x='merk',y='harga',color='merk')
fig5.update_layout(title='Harga Rata Rata masing masing merk pada E Commerce')
fig6=px.scatter(harga,x='hasil',y='harga',trendline='ols')
fig6.update_layout(title='Correlation between Price and Rating Youtube Based NLP')
st.set_page_config(page_title='E Commerce and Youtube Comment',layout='wide')
st.title('E Commerce and Youtube Comment')
st.write("Dashboard")

with st.container():
  col1,col2=st.columns([3,1])
  with col1:
    st.plotly_chart(fig1,use_container_width=True,)
  with col2:
    st.subheader('Proporsi Frekuensi Brand pada Komentar Youtube')
    st.plotly_chart(fig7,use_container_width=True)
  
with st.container():
  col1,col2=st.tabs(['trendline rating','rata rata harga'])
  with col1:
    st.plotly_chart(fig2,use_container_width=True)
  with col2:
    st.plotly_chart(fig5,use_container_width=True)
    
with st.container():
  col1,col2=st.columns(2)
  with col1:
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

    



    

