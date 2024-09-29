"""
From Ajay Misra Sept 17, 2024 -- 

Designed to show Calc 1-3 concepts in a quant-trading scenario. 

I know a lot of students are "pre-quant" and this is a little bit of 
what it is like to design an optimization model. Moreso, I designed
this lesson so you can get a little grip of what it is like to write 
a quant algorithm. 

These are concepts you've been studying all semester (plus some calc
1-2 concepts) for some quant finance applications. 

In general, I want to demystify calculus and that you are actually using
the building blocks for cutting-edge financial modeling and machine learning 
algorithms used on Wall Street. I simplified it heavily by using frameworks
like PyTorch / sci-kit learn / yfinance to make the code easier and more
coherent but the general jist of it is here. 

----

Problem Statement: 

You are a quantitative analyst intern at a major hedge fund. The fund manager 
has asked you to develop a comparative analysis of different machine learning models 
for predicting stock returns, with a specific focus on Apple Inc. (AAPL) stock. Your
task is to implement and evaluate three distinct models: 
- a linear model (Ridge Regression),
- a tree-based model (Gradient Boosting), and
- a neural network.

Requirements:

- Use 5 years of historical data for AAPL stock.
- Engineer relevant features including returns, log returns, volatility, moving 
averages, and RSI.
- Implement and train the three specified models.
- Evaluate the models using appropriate metrics (MSE and R-squared).
- Develop a simple trading strategy based on each model's predictions and backtest it.
- Compare the performance of the three models in terms of predictive accuracy
and trading strategy profitability.
- Visualize the results of your backtesting.


(CHALLENGE)
Additionally, the fund manager, who is passionate about nurturing new talent, wants 
you to document how concepts from calculus (particularly from Calc 1-3) are applied in your 
implementation. This documentation will be used for an upcoming intern training session.

Your code should be well-commented, explaining both the financial and mathematical concepts 
involved. The final deliverable should be a Python script that can be run to reproduce your
entire analysis, from data fetching to final visualization.


----

If you want to try to make a harder version:

Try to implement your own neural network / your own libraries using python 
(or a language of your choice!) to recreate this! 

or: 

Try to recreate this in another language! This will be tricky but languages
like C++ and OCaml provide great quant resources. Books I recommend to study
are on this list:

https://www.quantstart.com/articles/Quant-Reading-List-C-Programming/

Most of these books are available for rent at Student Stores! Or ask me or
Professor McLaughlin (or someone that specializes in quant like Cherednik!).

Have fun! 

----


Solution:


"""


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    return df

def prepare_data(df):
    """
    Derivatives: Returns approximate the derivative of price w.r.t. time.
    f'(x) = lim(h->0) [f(x+h) - f(x)] / h
    Here, h = 1 day: Returns = (Price(t) - Price(t-1)) / Price(t-1)
    
    Logarithms: Log returns use the natural logarithm.
    Log_Returns = ln(Price(t) / Price(t-1))
    Preferred in finance for its additive property over time.
    
    Multivariable Functions: Volatility as a function of multiple returns.
    Uses standard deviation, involving partial derivatives in multiple dimensions.
    
    Integrals: Moving averages approximate integrals.
    SMA = (1/n) * Σ[t-n+1 to t] Price(i) is a Riemann sum approximation.
    """
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'], window=14)
    df['Target'] = df['Returns'].shift(-1)
    
    df = df.dropna()
    features = ['Returns', 'Log_Returns', 'Volatility', 'SMA_20', 'SMA_50', 'RSI']
    return df[features], df['Target']

def calculate_rsi(prices, window=14):
    """
    Sequences and Series: RSI involves averages over a sequence of price changes.
    Related to convergence of sequences and series in calculus.
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Fetch and prepare data
ticker = 'AAPL'  # Apple Inc. stock
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)  # 5 years of data

df = fetch_stock_data(ticker, start_date, end_date)
X, y = prepare_data(df)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1)

# Ridge Regression
"""
Gradient and Optimization: Solves min_w ||Xw - y||^2 + α||w||^2
Uses gradient descent: w_new = w_old - η * ∇L(w_old)
Analogous to finding minimum of multivariable function using partial derivatives.
"""
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
ridge_pred = ridge_model.predict(X_test_scaled)

# Gradient Boosting
"""
Taylor Series and Multivariable Optimization: Uses functional gradient descent.
Approximates loss function: L(y, F + h) ≈ L(y, F) + ∂L/∂F * h
Similar to Taylor series expansion extended to functions of multiple variables.
"""
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

# Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        return self.layers(x)

"""
Chain Rule: Backpropagation applies the chain rule from multivariable calculus.
∂L/∂w_i = ∂L/∂fn * ∂fn/∂f(n-1) * ... * ∂f(i+1)/∂fi * ∂fi/∂w_i
This is the chain rule applied to the composite function L(fn(f(n-1)(...f1(x)...))).
"""

nn_model = NeuralNetwork(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)

# Training loop,, left as n because I lowkey don't want it to run for reasons I mentioned in lecture.
epochs = n
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = nn_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

# Make predictions
nn_model.eval()
with torch.no_grad():
    nn_pred = nn_model(X_test_tensor).numpy().flatten()

def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - MSE: {mse:.6f}, R2: {r2:.4f}")

evaluate_model(y_test, ridge_pred, "Ridge Regression")
evaluate_model(y_test, gb_pred, "Gradient Boosting")
evaluate_model(y_test, nn_pred, "Neural Network")

def backtest_strategy(y_true, y_pred, initial_capital=10000):
    """
    Infinite Series: Cumulative return relates to infinite series.
    Total Return = Π(1 + r_t) - 1
    Analogous to convergence of infinite products.
    
    Optimization: Sharpe ratio maximization is an optimization problem.
    max_w Sharpe(w) = (E[R_p] - R_f) / σ_p
    E[R_p]: expected portfolio return, R_f: risk-free rate, σ_p: portfolio std dev.
    """
    position = np.where(y_pred > 0, 1, -1)
    strategy_returns = position * y_true
    cumulative_returns = (1 + strategy_returns).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1
    sharpe_ratio = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    
    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    return cumulative_returns * initial_capital

print("\nBacktesting Results:")
print("Ridge Regression Strategy:")
ridge_portfolio = backtest_strategy(y_test, ridge_pred)
print("\nGradient Boosting Strategy:")
gb_portfolio = backtest_strategy(y_test, gb_pred)
print("\nNeural Network Strategy:")
nn_portfolio = backtest_strategy(y_test, nn_pred)


"""
MPL part left out. We used WandB in class. 
"""
