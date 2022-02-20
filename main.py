from flask import Flask, render_template, request, flash, redirect, url_for, session
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
import yfinance as yf
from yahoofinancials import YahooFinancials
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.python.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Sequential
from keras.models import load_model
import requests
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 10, 5
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
app = Flask(__name__)

@app.route('/crypto-prediction', methods = ['GET','POST'])
def crypto_data():
    headers={
        'X-CMC_PRO_API_KEY': '28939384-79cf-4fef-ac83-a5c216a00b74',
        'Accepts' : 'application/json'
    }
    params = {
        'start' : '1',
        'limit' : '10',
        'convert' : 'USD'
    }
    url  = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
    json = requests.get(url, params=params, headers=headers).json()
    coins=json['data']
    for i in coins:
        if i['symbol']=='BTC':
            btc_price= i['quote']['USD']['price']
            btc_volume=i['quote']['USD']['volume_24h']
        if i['symbol']=='ETH':
            eth_price= i['quote']['USD']['price']
            eth_volume=i['quote']['USD']['volume_24h']
        if i['symbol']=='ADA':
            ada_price= i['quote']['USD']['price']
            ada_volume=i['quote']['USD']['volume_24h']
        if i['symbol']=='BNB':
            bnb_price= i['quote']['USD']['price']
            bnb_volume=i['quote']['USD']['volume_24h']
        if i['symbol']=='USDT':
            usdt_price= i['quote']['USD']['price']
            usdt_volume=i['quote']['USD']['volume_24h']
        if i['symbol']=='XRP':
            xrp_price= i['quote']['USD']['price']
            xrp_volume=i['quote']['USD']['volume_24h']
        if i['symbol']=='SOL':
            sol_price= i['quote']['USD']['price']
            sol_volume=i['quote']['USD']['volume_24h']
        if i['symbol']=='DOT':
            dot_price= i['quote']['USD']['price']
            dot_volume=i['quote']['USD']['volume_24h']
        if i['symbol']=='USDC':
            usdc_price= i['quote']['USD']['price']
            usdc_volume=i['quote']['USD']['volume_24h']
        if i['symbol']=='DOGE':
            doge_price= i['quote']['USD']['price']
            doge_volume=i['quote']['USD']['volume_24h']
    return render_template('crypto-prediction.html', btc_price=np.round(btc_price,2), btc_volume=np.round(btc_volume,2), eth_price=np.round(eth_price,2), eth_volume=np.round(eth_volume,2), ada_price=np.round(ada_price,2), ada_volume= np.round(ada_volume,2), bnb_price=np.round(bnb_price,2), bnb_volume=np.round(bnb_volume,2), usdt_price=np.round(usdt_price,2), usdt_volume=np.round(usdt_volume,2), xrp_price=np.round(xrp_price,2), xrp_volume=np.round(xrp_volume,2), sol_price=np.round(sol_price,2), sol_volume=np.round(sol_volume,2), dot_price=np.round(dot_price,2), dot_volume=np.round(dot_volume,2), usdc_price=usdc_price, usdc_volume=np.round(usdc_volume,2), doge_price=np.round(doge_price,2), doge_volume=np.round(doge_volume, 2 ))



@app.route('/crypto-prediction-results',methods = ['GET','POST'])
def insertintotable():
    nm = request.form['nm']
    def get_historical(quote):
        df= pd.DataFrame()
        temp_df = yf.download(quote,start = '2014-01-01',end = '2021-12-31',progress = False)
        df= df.append(temp_df)
        df= df.reset_index(drop=False)
        df.head()
        return df
    def LSTM_ALGO(df):
        fig1=plt.figure(figsize=(16,8))
        ax = df.plot(x='Date', y='Close')
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price (USD)")
        plt.title('Recent Trends')
        plt.savefig('static/Trends.png')
        plt.close(fig1)

        if quote=="BTC-USD":
            model = load_model('static/Bitcoin.h5')
            data = [6404, 6296, 5193]
        if quote=="ETH-USD":
            model = load_model('static/Ethereum.h5')
            data = [5398, 4864, 7387]
        if quote=="BNB-USD":
            model = load_model('static/Binance.h5')
            data = [7396, 6950, 2294]
        if quote=="ADA-USD":
            model = load_model('static/Cardano.h5')
            data = [1709, 13031, 2983]
        if quote=="USDT-USD":
            model = load_model('static/Tether.h5')
            data = [5471, 794, 10912]
        if quote=="XRP-USD":
            model = load_model('static/XRP.h5')
            data = [5348, 2991, 9330]
        if quote=="SOL1-USD":
            model = load_model('static/Solana.h5')
            data = [4741, 3267, 6674]
        if quote=="DOT1-USD":
            model = load_model('static/Polkadot.h5')
            data = [3468, 994, 6827]
        if quote=="USDC-USD":
            model = load_model('static/USD.h5')
            data = [1040, 1409, 3531]
        if quote=="DOGE-USD":
            model = load_model('static/Dogecoin.h5')
            data = [6602, 1171, 9269]
        
        
        
        #CRYPTO TRAINING

        #1.MinMaxScaler
        scaler = MinMaxScaler()
        close_price = df.Close.values.reshape(-1, 1)
        scaled_close = scaler.fit_transform(close_price)
        scaled_close = scaled_close[~np.isnan(scaled_close)]
        scaled_close = scaled_close.reshape(-1, 1)



        #2.Training & Test Split
        SEQ_LEN = 100
        def to_sequences(data, seq_len):
            d = []
            for index in range(len(data) - seq_len):
                d.append(data[index: index + seq_len])
            return np.array(d)

        def preprocess(data_raw, seq_len, train_split):
            data = to_sequences(data_raw, seq_len)
            num_train = int(train_split * data.shape[0])
            X_train = data[:num_train, :-1, :]
            y_train = data[:num_train, -1, :]
            X_test = data[num_train:, :-1, :]
            y_test = data[num_train:, -1, :]
            return X_train, y_train, X_test, y_test
        X_train, y_train, X_test, y_test = preprocess(scaled_close, SEQ_LEN, train_split = 0.95)


        #5.Evaluation
        y_hat = model.predict(X_test)
        y_test_inverse = scaler.inverse_transform(y_test)
        y_hat_inverse = scaler.inverse_transform(y_hat)
        fig2=plt.figure(figsize=(16,8))
        plt.plot(y_test_inverse, label="Actual Price", color='green')
        plt.plot(y_hat_inverse, label="Predicted Price", color='red')
 
        plt.title(quote+ ' price prediction')
        plt.xlabel('Time [days]')
        plt.ylabel('Price')
        plt.legend(loc='best')
        plt.savefig('static/LSTM.png')
        plt.close(fig2)

        sentiment = ['Positive', 'Negative', 'Neutral']
  
        wp = { 'linewidth' : 1, 'edgecolor' : "green" }
        # # Creating plot
        def func(pct, allvalues):
            absolute = int(pct / 100.*np.sum(allvalues))
            return "{:.1f}%\n({:d} Tweets )".format(pct, absolute)
  
        # Creating plot
        fig3, ax = plt.subplots(figsize =(10, 7))
        wedges, texts, autotexts = ax.pie(data, autopct = lambda pct: func(pct, data), labels = sentiment,wedgeprops = wp, textprops = dict(color ="white"))
  
        # Adding legend
        ax.legend(wedges, sentiment ,loc ="center left",bbox_to_anchor =(1, 0, 0.5, 1))
        plt.setp(autotexts, size = 8, weight ="bold")
        ax.set_title("Sentiment Analysis")
        plt.savefig('static/sentiment.png')
        # plt.close(fig3)
        return


    quote=nm
    #Try-except to check if valid price symbol
    try:
        df = get_historical(quote)
    except:
        return render_template('.html',not_found=True)
    else:
        #************** PREPROCESSUNG ***********************
        # temp_df = pd.read_csv(''+quote+'.csv')
        # # today_price=temp_df.iloc[-1:]
        # df = pd.DataFrame()
        # #
        # # df = df.reset_index(drop=False)
        
        LSTM_ALGO(df)
        # today_price=today_price.round(2)
        return render_template('results.html',quote=quote)
if __name__ == '__main__':
   app.run()