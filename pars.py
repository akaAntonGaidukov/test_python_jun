#import
import sys
import requests
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

import time
from datetime import datetime
import threading


api_key = "dcb39226-229c-404e-8603-220ac76a05fd"

# Сбор данных для дальнейшего анализа
def pars_data(coin_name_list,interval="h2",v=False):
  """"valid intervals: m1, m5, m15, m30, h1, h2, h6, h12, d1"""
  
  time_now = datetime.timestamp(datetime.utcnow())*1000
  data = []

  for coin_name in coin_name_list:
    url = f"https://api.coincap.io/v2/assets/{coin_name}/history"

    params = {
        "interval":f"{interval}",
        "start":time_now-2629743000,
        "end":time_now
    }

    headers = {
      'Authorization': f'Bearer {api_key}'
    }

    response = requests.request("GET", url, headers=headers,params=params)
    if v:
        print(response.status_code)
    df_tmp = pd.DataFrame(response.json()["data"])
    df_tmp["time"] = df_tmp["time"].astype("int64")
    df_tmp["date"] = [datetime.fromtimestamp(int(i)*0.001) for i in df_tmp["time"]]
    df_tmp["priceUsd"] = df_tmp["priceUsd"].astype('float')
    df_tmp["circulatingSupply"]=df_tmp["circulatingSupply"].astype('float')

    df_tmp = df_tmp.rename(columns={"priceUsd":f"{coin_name}_priceUSD","circulatingSupply":f"{coin_name}_Supply"})
    df_tmp.set_index("time",inplace=True)
    
    data.append(df_tmp)
    time.sleep(1)
  
  data = pd.concat(data,axis=1)
  data = data.T.drop_duplicates().T

  return data

# Создание базовой модели для определения изменения собственной цены фьючерса
def LinReg(x,y,v=False):
    """Input x, y\n v- verbose[bool] - deafult False"""
    bitcoin_prices = np.array(x).reshape((-1, 1)).astype(float)
    ethereum_prices = np.array(y).astype(float)

    # Create a random forest regression model and fit it to the training data
    model = LinearRegression()
    model.fit(bitcoin_prices,ethereum_prices)
    if v:
        print(f"coef - {model.coef_[0]}\n",
        f"intercept - {model.intercept_}"
    )

    # Use the model to predict the Ethereum price based on the test data
    y_pred = model.predict(bitcoin_prices)

    # Evaluate the model's performance using R-squared and Mean Squared Error
    r_squared = r2_score(ethereum_prices, y_pred)
    mse = mean_squared_error(ethereum_prices, y_pred)

    if v:
        print(f'R-squared: {round(r_squared, 3)}, Mean Squared Error: {round(mse, 3)}')
    return model

# Моментальный сбор данных для отслеживания изменения цены в реальном времени
def fast_pars_data(coin_name_list,sleep_time=0,querry="assets"):
    """coin name orded matters"""
    time.sleep(sleep_time)
    if querry == "assets":
        url = "https://api.coincap.io/v2/assets?"
        params = {
            "ids":f"{','.join(coin_name_list)}"
        }
        headers = {
        'Authorization': f'Bearer {api_key}'
        }

    if querry == "history":
        url = f"https://api.coincap.io/v2/assets/{coin_name_list[0]}/history"

        params = {
            "interval":"h1",

        }

        headers = {
        'Authorization': f'Bearer {api_key}'
        }


    response = requests.request("GET", url, headers=headers, params=params)

    return response.json()["data"]

# Функция для обновления модели по историческим данным (отдельный поток)
def get_data(verbose=False):
    global h2_data,model, stop_thread
    while not stop_thread:
        h2_data = pars_data(["bitcoin","ethereum"],"h2")
        model = LinReg(x=h2_data["bitcoin_priceUSD"],y=h2_data["ethereum_priceUSD"],v=False)

        if verbose:
            print("Vars are updated")
    
        time.sleep(3600) # 3600 sec Ожидание 1 часа


if __name__ == '__main__':

    animation = "|/-\\"
    idx = 0
    stop_thread = False
    flag = True
    h2_data=0
    model=0

    print(f"Program is starting...{' '*50}",end="\r")
    t = threading.Thread(target=get_data)
    t.start() # Запуск потока
    
    while flag:

        if model == 0:
            time.sleep(1)
        else: 
            flag = False

    while True:
        

        try:
                
            data = fast_pars_data(["bitcoin","ethereum"],2)
            
            btc_price = float(data[0]["priceUsd"])
            eth_price = float(data[1]["priceUsd"])
            eth_lr = model.intercept_ + model.coef_[0]*btc_price
            eth_var = eth_price/eth_lr*100-100
            
            if eth_var>1:
                text = [np.round(eth_var,2)]
                for i in data:
                        price = f"{i['id']} - {np.round(float(i['priceUsd']),2)}"
                        text.append(price)

                print(f"Real time price change {animation[idx % len(animation)]} \033[32m{' | '.join(text)}\033[0m {animation[idx % len(animation)]}",end="\r")
            else:
                print(f"Real time price change {animation[idx % len(animation)]} \033[31m{np.round(eth_var,2)}\033[0m % {animation[idx % len(animation)]}{' '*50}",end="\r")
            idx +=1
      
        except KeyboardInterrupt:
                
                stop_thread = True
                break


    t.join()
    print(f"joining{' '*50}", end="\r")
    time.sleep(2)
    print(f"exiting{' '*50}", end = "\r")
    print(' '*50)
    sys.exit()