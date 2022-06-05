# JPX-Tokyo-Stock-Exchange-Prediction

# 目標

對確定日期和股票代碼進行Rank預測，Rank表示2000隻股票中每隻股票的第二天收盤價和第二天收盤價的變化率的排名

# 資料前處理

應用 upper_shadow、lower_shadow 完成特徵提取

![image](https://user-images.githubusercontent.com/102530486/172061802-61da38a6-c1f1-4b04-8d28-cc67d6947641.png)

## upper_shadow
   
   當日最高價 - 從開盤價、收盤價中取最大值
   
## lower_shadow

   在兩者中取最小值，分別為收盤價、開盤價扣除最低價
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

    Open：開盤價
    Close：收盤價
    Volume：一天內股票交易量
    AdjustmentFactor：理論價格/成交量
    ExpectedDividend：除權日
    Target = [Close(t+2)-Close(t+1)] / Close(t+1) (Target越大Rank排名越靠前)

### secondary_stock_prices.cs

    stock_prices是核心數據集，包含了2000種最常交易的股票。但許多流動性較低的股票也在東京市場上交易，他們雖然沒有評分，但可以幫住評估整個市場
   
### options.csv
    
    基於大盤的期權狀況數據
### financials.csv
     
     針對4071隻股票的季度收益報告的結果
     
### trades.csv

    上一個商業星期的總交易量。共1712條數據，缺失嚴重，約44.68%的行不能使用
    
## supplemental_files

    與train_files文件內容格式完全一致，將會在5月初、6月初的競賽主階段，以及提交文件鎖定前大約一周，用新數據更新。
    在最終模型訓練過程中，需要將這裡面的數據與train_files合併作為我們最終的訓練集使用。
# 描述
任何金融市場的成功都需要確定可靠的投資。當股票或衍生品被低估時，購買是有意義的。如果它被高估，也許是時候出售了。雖然這些財務決策歷來是由專業人士手動做出的，但技術為散戶投資者帶來了新的機會。具體來說，數據科學家可能有興趣探索量化交易，其中決策是根據訓練模型的預測以編程方式執行的。

現有大量量化交易工作用於分析金融市場和製定投資策略。創建和執行這樣的策略需要歷史和實時數據，尤其是散戶投資者很難獲得這些數據。本次大賽將為日本市場提供金融數據，讓散戶對市場進行最全面的分析。

Japan Exchange Group, Inc. (JPX) 是一家控股公司，運營著世界上最大的證券交易所之一、東京證券交易所 (TSE) 以及衍生品交易所大阪交易所 (OSE) 和東京商品交易所 (TOCOM)。JPX 是本次比賽的主辦方，並得到了 AI 技術公司 AlpacaJapan Co.,Ltd. 的支持。

本次比賽將在訓練階段完成後將您的模型與真實的未來回報進行比較。比賽將涉及從符合預測條件的股票（約 2,000 隻股票）中建立投資組合。具體來說，每個參與者從最高到最低的預期回報對股票進行排名，並根據前 200 隻股票和後 200 隻股票之間的回報差異進行評估。您可以訪問日本市場的財務數據，例如股票信息和歷史股票價格，以訓練和測試您的模型。

所有獲獎模型將被公開，以便其他參與者可以向優秀模型學習。優秀的模型也可能會增加散戶投資者對市場的興趣，包括那些想要進行量化交易的人。同時，您將對程序化投資方法和投資組合分析有自己的見解——您甚至可能會發現自己對日本市場情有獨鍾。
https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction

# 評估

提交是根據每日點差回報的夏普比率評估的。您需要對給定日期的每隻活躍股票進行排名。單日回報將排名最高的 200 只（例如 0 到 199）股票視為買入，將排名最低（例如 1999 到 1800）的 200 隻股票視為賣空。然後根據股票的排名對股票進行加權，並假設股票在第二天購買並在第二天賣出，從而計算投資組合的總回報。您可以在此處找到該指標的 python 實現。

您必須使用提供的 python 時間序列 API 提交本次比賽，以確保模型不會及時向前窺視。要使用 API，請遵循 Kaggle Notebooks 中的此模板：

    import jpx_tokyo_market_prediction
    env = jpx_tokyo_market_prediction.make_env()   # initialize the environment
    iter_test = env.iter_test()    # an iterator which loops over the test files
    for (prices, options, financials, trades, secondary_prices, sample_prediction) in iter_test:
    sample_prediction_df['Rank'] = np.arange(len(sample_prediction))  # make your predictions here
    env.predict(sample_prediction_df)   # register your predictions

