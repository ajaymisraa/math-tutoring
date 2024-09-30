"""

By Ajay Misra - Sept 1, 2024. 

Intended for Calculus 3 / Linear Students. Challenging!!! 

Topics: 
- Understanding rates of change (returns), optimization (gradient descent),
and basic integration (cumulative returns).
- Working with sequences and series (lagged features), and approximating functions.
- Handling functions of multiple variables (multivariate regression), partial
derivatives (gradient computation), and probabilistic models (HMM).
- Research on quantitative finance topics! This is the hardest part- you have
to figure out what you don't know! 

PROBLEM STATEMENT:

Using historical financial data (HYG, IEF, SPY, VIX)*, develop a model to predict
credit spreads and create a trading strategy. Apply calculus concepts including:

- Rate of change for daily returns
- Multivariate analysis for credit spread prediction
- Time series analysis with lagged features
- Probabilistic modeling with Hidden Markov Models
- Gradient descent optimization in XGBoost
- Integration for cumulative returns

*Other features that would be interesting to add to a confusion matrix:

BDSpread_long (Bond Spread Long-term), BDSpread_short (Bond Spread Short-term), MKM3, 
SPY_DIFF (S&P 500 Difference), TEDRATE (TED Rate - difference between interest 
rates on interbank loans and short-term government debt), Y1M6 (Yield difference between 
1-year and 6-month), Y2Y1 (Yield difference between 2-year and 1-year), leverage, buy bond, 
TB3MS (3-Month Treasury Bill Secondary Market Rate), USRQE, TERMCBAUTO48NS, PCE (Personal 
Consumption Expenditures), CPAUCSL (Consumer Price Index for All Urban Consumers), GDP
(Gross Domestic Product), Notes_Gross_Issues, Notes_Net, Bonds_Gross_Issues, Bonds_Net, 
10-year issuance, Bill_Net, Monetary policy, Fiscal Policy, National security, Sovereign 
debt, y_lag1 (Lagged variable of choice), market regime_cs, vix_new1 (Volatility Index)

(for the sake of simplicity, for the problem solution, we only did HYG, SPY, and VIX) 

Compare your model's performance against a linear regression benchmark and evaluate
the trading strategy against a buy-and-hold approach. Explain the calculus concepts
applied at each step.

(CHALLENGE) 

Add your own flair to your code!! Add layers! Incorperate niche calculus terms and ideas!
Be your own person!!! 

Or try in another programming language! 

---- 

Solution video on CANVAS!!! Come to my math sessions every Sunday in Phillips 214! 

Reminder: The solution does not have to be done with code! You can write it out in your
own language and submit it. That is a more fun route tbh. 

SOLUTION AS I SEE IT::::

"""



# Import necessary libraries
import numpy as np  # For numerical computations
import pandas as pd  # For data manipulation
import yfinance as yf  # For data acquisition from Yahoo Finance
from hmmlearn.hmm import GaussianHMM  # For Hidden Markov Model
import xgboost as xgb  # For XGBoost model
from sklearn.linear_model import LinearRegression  # For linear regression benchmark
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.metrics import mean_squared_error  # For model evaluation
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For advanced plotting

"""
We import libraries that will help us perform numerical computations
(numpy), manipulate data (pandas), acquire financial data (finance),
model hidden states (hmmlearn), and build machine learning models
(xgboost, sklearn).
"""

# Step 1: Data Acquisition

# Define the time period for data
start_date = '2015-01-01'
end_date = '2023-01-01'

# Download data for High Yield Corporate Bonds ETF (HYG) and Treasury Bonds ETF (IEF)
hyg = yf.download('HYG', start=start_date, end=end_date)
ief = yf.download('IEF', start=start_date, end=end_date)

"""
- We select the time period from January 1, 2015, to January 1, 2023.

(more info on the following in the slideshow in canvas)
- HYG is an ETF that tracks high-yield (junk) corporate bonds, representing
risky securities.
- IEF is an ETF that tracks 7-10 year Treasury bonds, representing risk-free
securities.
- By downloading the adjusted closing prices, we can compute yields and
thus the credit spread.
"""

# Step 2: Compute Daily Returns to Approximate Yields

# Calculate daily returns for HYG and IEF
hyg['Return'] = hyg['Adj Close'].pct_change()
ief['Return'] = ief['Adj Close'].pct_change()

"""
The daily return is calculated using

Return(t) = [P(t)-P(t-1)]/(P(t-1)

where P(t) is the price at time t. 

This is basically a derivative; a discrete approximation of 
the derivative of price w/ respect to time. 
"""

# Step 3: Compute Credit Spread

# Compute the credit spread as the difference in returns
data = pd.DataFrame()
data['Date'] = hyg.index
data['Credit_Spread'] = hyg['Return'] - ief['Return']
data.set_index('Date', inplace=True)

"""
The credit spread is calculated as the difference between the returns
of the risky security (HYG) and the risk-free security (IEF):

Credit Spread(t) = Return(HYG,t) - Return(IEF,t)

Difference measures the risk premium & economic uncertainty.

"""

# Step 4: Acquire Additional Features

# Download S&P 500 Index (SPY) and Volatility Index (VIX)
spy = yf.download('SPY', start=start_date, end=end_date)
vix = yf.download('^VIX', start=start_date, end=end_date)

# Calculate returns for SPY
spy['Return'] = spy['Adj Close'].pct_change()

# Merge all data into a single DataFrame
data['SPY_Return'] = spy['Return']
data['VIX_Close'] = vix['Adj Close']

"""

There are a bunch of additional features you can add into your
model. We will do the VIX (the volitility indicator) and the 
SPY (the S&P 500) returns & closing prices. Several variables
lead to several dimensions; hence several ordered derivatives.

Credit spread is a function of all of these variables. 

"""

# Step 5: Data Preprocessing

# Handle missing values
data.dropna(inplace=True)

# Feature Engineering: Add lagged features
data['Credit_Spread_Lag1'] = data['Credit_Spread'].shift(1)
data['Credit_Spread_Lag2'] = data['Credit_Spread'].shift(2)
data.dropna(inplace=True)

# Step 6: Hidden Markov Model for Regime Detection

# Prepare data for HMM
hmm_data = data[['Credit_Spread']]

# Fit HMM with two hidden states (e.g., Bull and Bear markets)
model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000).fit(hmm_data)
hidden_states = model.predict(hmm_data)
data['Regime'] = hidden_states

"""

This mainly hits on Hidden Markov Models, an idea we hit on in class 
- in this perspective, we handle probablistic processes with hidden
states & handle credit spreads as such. It's really used to detect 
"market regimes" as the textbook says.

This relates a lot towards probability distributions and stochastic
processes.

VERY similar to transition matricies * probability density functions like
Sept 8's lesson. 
"""

# Step 7: Prepare Data for Modeling

# Select features and target variable
features = ['SPY_Return', 'VIX_Close', 'Credit_Spread_Lag1', 'Credit_Spread_Lag2', 'Regime']
X = data[features]
y = data['Credit_Spread']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)


# Step 8: Linear Regression Benchmark Model

# Initialize and train the linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict on the test set
y_pred_lr = lr_model.predict(X_test)

# Evaluate the model
mse_lr = mean_squared_error(y_test, y_pred_lr)
print(f"Linear Regression MSE: {mse_lr}")


"""

Linear regression models the relationship between the dependent variable
and independent variables using a linear equation:

like y= B0+ B1(x1) + B2(x2) + ... 

Mean Squared Error is used primarily for this: 

1/n sum (n) (i=1) of [yi-yhat(i)]^2

"""

# Step 9: XGBoost Model

# Prepare data for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters for XGBoost
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.1,
    'max_depth': 5
}

# Train the XGBoost model
xgb_model = xgb.train(params, dtrain, num_boost_round=100)

# Predict on the test set
y_pred_xgb = xgb_model.predict(dtest)

# Evaluate the model
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f"XGBoost MSE: {mse_xgb}")

"""
XGBoost is an advanced machine learning algorithm based on gradient boosting.

It minimizes the loss function using GRADIENT DESCENT!!! Told you we would
get to it at some point. 

The loss function for regression is often the squared error, and gradient 
descent involves taking derivatives to find the minimum:

L = 1/n sum (n) (i=1) of [yi-yhat(i)]^2

learning_rate corresponds to the step size 𝛼 (alpha) in gradient descent.

"""

# Step 10: Compare Models

print(f"Linear Regression MSE: {mse_lr}")
print(f"XGBoost MSE: {mse_xgb}")

# Plot actual vs predicted credit spread
plt.figure(figsize=(14,7))
plt.plot(y_test.index, y_test, label='Actual Credit Spread', color='blue')
plt.plot(y_test.index, y_pred_lr, label='Predicted Credit Spread (Linear Regression)', color='red', alpha=0.7)
plt.plot(y_test.index, y_pred_xgb, label='Predicted Credit Spread (XGBoost)', color='green', alpha=0.7)
plt.legend()
plt.title('Actual vs Predicted Credit Spread')
plt.xlabel('Date')
plt.ylabel('Credit Spread')
plt.show()

# Step 11: Develop Trading Strategy Based on Predictions

# Assume we can trade an ETF correlated with credit spread, e.g., HYG

# Generate trading signals based on XGBoost predictions
data_test = X_test.copy()
data_test['Predicted_Credit_Spread'] = y_pred_xgb
data_test['Signal'] = np.where(data_test['Predicted_Credit_Spread'] > data_test['Credit_Spread_Lag1'], 1, -1)

# Calculate strategy returns
hyg_test = hyg.loc[X_test.index]
hyg_test['Strategy_Return'] = data_test['Signal'].shift(1) * hyg_test['Return']
hyg_test['Cumulative_Strategy_Return'] = (1 + hyg_test['Strategy_Return']).cumprod()

# Calculate buy-and-hold returns
hyg_test['Cumulative_Buy_and_Hold_Return'] = (1 + hyg_test['Return']).cumprod()

# Plot cumulative returns
plt.figure(figsize=(14,7))
plt.plot(hyg_test.index, hyg_test['Cumulative_Strategy_Return'], label='Strategy Return', color='green')
plt.plot(hyg_test.index, hyg_test['Cumulative_Buy_and_Hold_Return'], label='Buy and Hold Return', color='blue')
plt.legend()
plt.title('Trading Strategy Performance')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.show()

"""
- We generate trading signals: buy (1) if the predicted credit spread
is increasing, sell (-1) if decreasing.
- the strategy return is calculated by multiplying the signal with the HYG return:

stat return(t) = signal(t-1) * return(HYG,t)

Cumulative return is calculated using the product of (1 + strategy return), which 
HEAVILY relates to exponential growth and integration!!
"""


# Step 12: Evaluate Strategy Performance

# Calculate total returns
total_strategy_return = hyg_test['Cumulative_Strategy_Return'].iloc[-1] - 1
total_buy_and_hold_return = hyg_test['Cumulative_Buy_and_Hold_Return'].iloc[-1] - 1

print(f"Total Strategy Return: {total_strategy_return * 100:.2f}%")
print(f"Total Buy and Hold Return: {total_buy_and_hold_return * 100:.2f}%")
