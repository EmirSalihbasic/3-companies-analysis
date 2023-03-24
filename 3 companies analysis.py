# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 21:47:25 2023

@author: Korisnik
"""

# -- Import librairies --  
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Extract data https://pypi.org/project/fundamentalanalysis/ 
import fundamentalanalysis as fa
api_key = "bc0466dc3a4561fb5679c9578bd39689"

# Ticker we use
#ticker = "MSFT"
list_ticker = ["MSFT","AAPL","GOOGL"]
n_ticker = len(list_ticker)

# ========== Financial Rotios for our tickers : Show a large set of in-depth ratios
d_T1 = fa.financial_ratios(list_ticker[0], api_key, period="annual")
d_T2 = fa.financial_ratios(list_ticker[1], api_key, period="annual")
d_T3 = fa.financial_ratios(list_ticker[2], api_key, period="annual")

# ========== Key Metrics for our tickers 
d_T1_key_metrics = fa.key_metrics(list_ticker[0], api_key, period="annual")
d_T2_key_metrics = fa.key_metrics(list_ticker[1], api_key, period="annual")
d_T3_key_metrics = fa.key_metrics(list_ticker[2], api_key, period="annual")

year_filter_min = 2021-5
year_filter_max = 2021

def transform_df(df):
# Def Transform Financial ratios DataFrame 
  # Transpose to put year in rows and ratios in columns
  df = df.T
  #Index is now a column
  df.reset_index(inplace=True)
  #Rename column index into years
  df = df.rename(columns={"index": "year"})
  #Drop column 'period'
  df = df.drop(columns=['period'])
  # Convert all columns of DataFrame into float
  n_columns_d = np.size(df,axis=1)
  for i in range(n_columns_d): 
    df.iloc[:,i] = pd.to_numeric(df.iloc[:,i], downcast="float")
  # Column year is int
  df['year'] = df['year'].astype(int)
  # Filter data until year_filter
  df = df[(df['year'] >= year_filter_min) & (df['year'] <= year_filter_max)]
  return df

#New dataFrame with our def transform_df
d_T1 = transform_df(d_T1)
d_T2 = transform_df(d_T2)
d_T3 = transform_df(d_T3)


d_T1_key_metrics = transform_df(d_T1_key_metrics)
d_T2_key_metrics = transform_df(d_T2_key_metrics)
d_T3_key_metrics = transform_df(d_T3_key_metrics)


#Merging dataframe by ticker
def merging_df(df1,df2) :
  for col in df1.columns :
    if col not in df1.columns.difference(df2.columns) and col != 'year' :
      df1 = df1.drop(columns=[col])
  df1 = pd.merge(df1, df2, how="outer" , on="year")
  #--------------------------------------------------------To calculate and add a new ratio or key metrics !!! [ADD NEW RATIO]
  #dataframe['new ratio'] = Calculation between dataframe['metric_x']
  df1['ebitda'] = df1['enterpriseValue']/df1['enterpriseValueOverEBITDA']
  return df1

d_T1 = merging_df(d_T1,d_T1_key_metrics)
d_T2 = merging_df(d_T2,d_T2_key_metrics)
d_T3 = merging_df(d_T3,d_T3_key_metrics)


# Set index to year
d_T1 = d_T1.set_index('year')
d_T2 = d_T2.set_index('year')
d_T3 = d_T3.set_index('year')


''' LETs in comment
# Merging all dataFrame
d_T1T2 = pd.merge(d_T1, d_T2, how="outer" , on="year",suffixes=(None, '_T2'))
d_T1T2T3 = pd.merge(d_T1T2, d_T3, how="outer" , on="year",suffixes=(None, '_T3'))
d_T1T2T3T4 = pd.merge(d_T1T2T3, d_T4, how="outer",on="year",suffixes=(None, '_T4'))

#Copy from merging to work with
dfPlot = d_T1T2T3T4.copy()'''

#Function Plot Line
def plot_line(name_ratio) :
  # Visualization RATIOS OF ALL the stocks

  # to set the plot size
  plt.figure(figsize=(16, 8), dpi=150)
  '''
  #To have plot separate
  d_T1.plot(x='year',y=name_ratio,label=list_ticker[0], color='orange')
  d_T2.plot(x='year',y=name_ratio,label=list_ticker[1],color='red')
  d_T3.plot(x='year',y=name_ratio, kind='line',label=list_ticker[2],color='black')
  d_T4.plot(x='year',y=name_ratio, kind='line',label=list_ticker[3],color='blue')
  '''
  # in plot method we set the label and color of the curve.
  d_T1[name_ratio].plot(label=list_ticker[0], color='orange')
  d_T2[name_ratio].plot(label=list_ticker[1],color='red')
  d_T3[name_ratio].plot(label=list_ticker[2],color='black')
 
  
  # adding title to the plot  
  plt.title(name_ratio)

  # adding Label to the x-axis
  plt.xlabel('Years')

  # adding legend to the curve
  plt.legend()
  return

#Enter ratio you want to plot
plot_line('returnOnAssets')
plot_line('returnOnCapitalEmployed')
plot_line('returnOnEquity')
plot_line('netProfitMargin')

plot_line('currentRatio')
plot_line('quickRatio')
plot_line('cashRatio')

plot_line('debtEquityRatio')
plot_line('debtRatio')

plot_line('debtEquityRatio')
plot_line('priceToSalesRatio')
plot_line('dividendPayoutRatio')
plot_line('priceEarningsToGrowthRatio')
plot_line('enterpriseValueMultiple')
plot_line('priceSalesRatio')
plot_line('ebitda')
plot_line('interestCoverage')


# Date
begin_date = "2016-01-01"
end_date = "2022-01-01"

# ========== Stocks data detailled : Show a open, close,high, low
d_T1_stock_init = fa.stock_data_detailed(list_ticker[0], api_key, begin= begin_date, end=end_date)
d_T2_stock_init = fa.stock_data_detailed(list_ticker[1], api_key, begin= begin_date, end=end_date)
d_T3_stock_init = fa.stock_data_detailed(list_ticker[2], api_key, begin= begin_date, end=end_date)
d_T4_stock_init = fa.stock_data_detailed(list_ticker[3], api_key, begin= begin_date, end=end_date)  

# --- TRANSFORM DATA ---
#Function that transform df prices
def transform_df_prices(df,lsT):
  df = df[[ 'adjClose']]
  # make columns names prettier
  df.columns = ['price']
  # Convert price columns of DataFrame into float
  df["price"] = pd.to_numeric(df["price"],downcast="float")
  #Sort by ascending date
  df = df.sort_index(ascending=True)
  df["ticker"] =lsT
  return df



# --- COMPUTE CUMMULATE DAILY RETURN AND YEARLY RETURNS ---
# Function that compute cumulative daily returns
def calculate_cum_daily_returns (df):
  #N rows of df
  n_rows_d = np.size(df,axis=0)
  #Init column Daily Returns
  df['Daily_Returns'] = np.zeros(n_rows_d)
  # Daily Return = (Price - Previous Price )/ Previous Price
  for i in range(1,n_rows_d,1):
    df.iloc[i, df.columns.get_loc('Daily_Returns')] = ((df.iloc[i, df.columns.get_loc('price')] - df.iloc[i-1, df.columns.get_loc('price')])/df.iloc[i-1, df.columns.get_loc('price')])
  # skip first row with NA 
  df['Cum_Daily_Returns'] = df.iloc[1:,df.columns.get_loc('Daily_Returns')]
  # Calculate the cumulative daily returns
  df['Cum_Daily_Returns'] = (1 + df['Cum_Daily_Returns']).cumprod() - 1
  # Calculate the cumulative daily returns in pourcentage
  df['Cum_Daily_Returns_Pourcentage'] = df['Cum_Daily_Returns']*100
  # Date in a column
  df = df.reset_index()
  # Rename column
  df = df.rename(columns={"index": "date"})
  #Format to datetime
  df['date'] = pd.to_datetime(df['date'])
  # --- Yearly Compute ---
  #Create a column name year to join for yearly returns
  df['year'] = df['date'].dt.year 
  #---Create a new dataFrale dfYearly
  dfYearly = df.groupby(["ticker", "year"])["date"].agg(["min", "max"]).reset_index()
  dfYearly = dfYearly.merge(df[["ticker", "date", "price"]], how="left", left_on=["ticker", "min"], right_on=["ticker","date"])
  dfYearly.rename(columns={"price": "Price-Begin"}, inplace=True)
  dfYearly.drop("date", axis=1, inplace=True)
  dfYearly = dfYearly.merge(df[["ticker", "date", "price"]], how="left", left_on=["ticker", "max"], right_on=["ticker","date"])
  dfYearly.rename(columns={"price": "Price-End"}, inplace=True)
  dfYearly.drop(["ticker","date", "min", "max"], axis=1, inplace=True)
  dfYearly["Annual_Return_Pourcentage"] = np.round(dfYearly["Price-End"].values / dfYearly["Price-Begin"].values - 1, 3)*100
  # Merging Df with dfYearly in order to have Yearly_Returns for each year
  df = pd.merge(left=df, right=dfYearly, how='inner', left_on='year', right_on='year')
  return df

#New dataFrame computed
d_T1_stock = calculate_cum_daily_returns(d_T1_stock)
d_T2_stock = calculate_cum_daily_returns(d_T2_stock)
d_T3_stock = calculate_cum_daily_returns(d_T3_stock)
d_T4_stock = calculate_cum_daily_returns(d_T4_stock)

# Concatenation of all the data
data_built = [d_T1_stock, d_T2_stock,d_T3_stock,d_T4_stock]
data = pd.concat(data_built)

# DATAFRAME Price
prices = data.pivot_table(index=['date'], columns='ticker', values=['price'])

# DATAFRAME Daily_Returns  
daily_returns = data.pivot_table(index=['date'], columns='ticker', values=['Daily_Returns'])

# DATAFRAME Cum_Daily_Returns  
cum_daily_returns = data.pivot_table(index=['date'], columns='ticker', values=['Cum_Daily_Returns'])

# DATAFRAME Cum_Daily_Returns_Pourcentage   
cum_daily_returns_pourcentage = data.pivot_table(index=['date'], columns='ticker', values=['Cum_Daily_Returns_Pourcentage'])

# DATAFRAME Cum_Daily_Returns_Pourcentage   
yearly_returns = data.pivot_table(index=['year'], columns='ticker', values=['Annual_Return_Pourcentage'])

print(yearly_returns)

plt.figure(figsize=(16, 8), dpi=150)
for c in prices.columns.values:
   plt.plot(prices.index, prices[c], label = c, lw = 2, alpha = .7)
plt.title('Daily Prices')
plt.ylabel('Prices ($)')
plt.xlabel('Years')
plt.legend(prices.columns.values, loc= 'upper right')
plt.show()


plt.figure(figsize=(16, 8), dpi=150)
for c in daily_returns.columns.values:
   plt.plot(daily_returns.index, daily_returns[c], label = c, lw = 2, alpha = .7)
plt.title('Daily Simple Returns')
plt.ylabel('Percentage (in decimal form')
plt.xlabel('Years')
plt.legend(daily_returns.columns.values, loc= 'upper right')
plt.show()

import seaborn as sns
plt.subplots(figsize= (15,15))
sns.heatmap(daily_returns.corr(), annot= True, fmt= '.2%')

plt.figure(figsize=(16, 8), dpi=150)
for c in cum_daily_returns.columns.values:
  plt.plot(cum_daily_returns.index, cum_daily_returns[c], lw=2, label= c)
plt.title('Daily Cumulative Simple Return')
plt.xlabel('Years')
plt.ylabel('Growth of $1 investment')
plt.legend(cum_daily_returns.columns.values, loc = 'upper left', fontsize = 10)
plt.show()

yearly_returns.plot(kind='bar',figsize=(30, 15))
plt.ylim(-50,170)
plt.title('Yearly Returns')
plt.ylabel('Percentage')
plt.xlabel('Years')
plt.legend(yearly_returns.columns.values, loc= 'upper right')
plt.show()