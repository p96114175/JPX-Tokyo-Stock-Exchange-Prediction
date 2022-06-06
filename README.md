# JPX-Tokyo-Stock-Exchange-Prediction

# 目標

對確定日期和股票代碼進行Rank預測，Rank表示2000隻股票中每隻股票的第二天收盤價和第二天收盤價的變化率的排名

# Kaggle team

![image](https://user-images.githubusercontent.com/48245648/172189051-da1caa8e-d62a-412b-b12f-232c8a936da8.png)

# 最後上傳的Ranking

![image](https://user-images.githubusercontent.com/48245648/172189839-c3c32802-e3b2-4a3b-82ce-2e7cc0382d42.png)

# Feature Engineering
透過對原始資料的分析與加工，使得AI Model可以處理更為有意義的資料，並且提升Ｍodel的準確性，以下針對我們有使用的Feature介紹
* Upper & Lower shadow:有些技術分析方法，是藉由這些K線的型態、排列方式，來了解現在市場情緒。
![image](https://user-images.githubusercontent.com/102530486/172061802-61da38a6-c1f1-4b04-8d28-cc67d6947641.png)
* Return:計算與過去股價差距之百分比，如果為正，則為上漲趨勢，反之為負，則下跌趨勢。
* MovingAvg:MA線是一條平滑的曲線，所以可以利用斜率來判斷目前股價的發展趨勢。
* Volatility:波動率高的特點是價格變化節奏極快，交易量較大，市場出現意外重大價格變動。另一方面，波動率較低往往趨於穩定，並且價格波動較小。


# 使用資料說明

![image](https://user-images.githubusercontent.com/102530486/172060974-1a6b8760-2831-4e29-92e0-a87ad188bf34.png)

data_specification: 給出數據表的各列具體意義（僅給出各列的具體含義）

example_test_files: 測試集的數據文件夾，用於預測提交，與train_files格式一致，只是缺少'Target'

jpx_tokyo_market_prediction: 啟動測試提交的API，需5分鐘內提交所有行並少於0.5GB內存（我們不必考慮文件內容，與比賽數據無關）

supplemental_files: 包含補充訓練數據的動態窗口

train_files: 訓練集，主要文件夾包含了各類股票信息

stock_list.csv: SecuritiesCode(即股票id)和公司名稱之間的映射，以及有關公司所在行業的一般信息
## train_files 
### stock_prices.csv

> Open：開盤價
> Close：收盤價
> Volume：一天內股票交易量
> AdjustmentFactor：理論價格/成交量
> ExpectedDividend：除權日
> Target = [Close(t+2)-Close(t+1)] / Close(t+1) (Target越大Rank排名越靠前)

### secondary_stock_prices.cs
> stock_prices是核心數據集，包含了2000種最常交易的股票。但許多流動性較低的股票也在東京市場上交易，他們雖然沒有評分，但可以幫住評估整個市場
   
### options.csv    
> 基於大盤的期權狀況數據
### financials.csv   
> 針對4071隻股票的季度收益報告的結果
     
### trades.csv
>上一個商業星期的總交易量。共1712條數據，缺失嚴重，約44.68%的行不能使用
    
### supplemental_files
> 與train_files文件內容格式完全一致，將會在5月初、6月初的競賽主階段，以及提交文件鎖定前大約一周，用新數據更新。
> 在最終模型訓練過程中，需要將這裡面的數據與train_files合併作為我們最終的訓練集使用。

# 驗證

提交是根據每日點差回報的夏普比率評估的。您需要對給定日期的每隻活躍股票進行排名。單日回報將排名最高的 200 只（例如 0 到 199）股票視為買入，將排名最低（例如 1999 到 1800）的 200 隻股票視為賣空。然後根據股票的排名對股票進行加權，並假設股票在第二天購買並在第二天賣出，從而計算投資組合的總回報。您可以在此處找到該指標的 python 實現。

您必須使用提供的 python 時間序列 API 提交本次比賽，以確保模型不會及時向前窺視。要使用 API，請遵循 Kaggle Notebooks 中的此模板：
```python=
    import jpx_tokyo_market_prediction
    env = jpx_tokyo_market_prediction.make_env()   # initialize the environment
    iter_test = env.iter_test()    # an iterator which loops over the test files
    for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
    sample_prediction_df['Rank'] = np.arange(len(sample_prediction))  # make your predictions here
    env.predict(sample_prediction_df)   # register your predictions
```
