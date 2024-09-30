# Importing necessary libraries for advanced mathematical modeling and machine learning
import numpy as np  # NumPy for efficient numerical computations and array operations
import pandas as pd  # Pandas for data manipulation and analysis
import matplotlib.pyplot as plt  # Matplotlib for creating visualizations
from scipy import stats, optimize, integrate  # SciPy for statistical functions, optimization, and integration
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.ensemble import RandomForestRegressor  # Random Forest algorithm for regression
from sklearn.metrics import mean_squared_error, r2_score  # Metrics for model evaluation
import sympy as sp  # SymPy for symbolic mathematics

"""
PROBLEM STATEMENT: 

As a data engineer at Amazon, you face the monumental task of revolutionizing the company's
price optimization system to maintain market leadership in an increasingly dynamic e-commerce
landscape. The current system struggles to efficiently handle the scale and diversity of 
Amazon's vast product catalog, which spans millions of items across hundreds of categories. 
It lacks the agility to respond in real-time to rapidly changing market conditions, including
competitor pricing, demand fluctuations, and supply chain disruptions. Furthermore, the existing
framework fails to adequately account for complex product interdependencies, seasonal trends,
and the unique characteristics of Amazon's multiple sales channels. These limitations often result
in suboptimal pricing decisions, potentially leading to lost sales and eroded profit margins.

Your challenge is to develop an advanced, scalable, and mathematically rigorous price optimization
system that addresses these critical issues. The new system should incorporate complex mathematical
models, including multivariable calculus and differential equations, to capture intricate price-demand
relationships. It must leverage machine learning algorithms to predict demand patterns and optimal
prices based on a multitude of factors, while implementing Monte Carlo simulations to account for
market uncertainties. The solution should be capable of adjusting prices dynamically across all of
Amazon's sales channels, optimizing for both short-term revenue and long-term market position.

Crucially, this system must be computationally efficient, handling millions of pricing decisions
per hour without significant latency, and should include advanced visualization techniques to aid
business stakeholders in understanding and interacting with the pricing models. Your task is to
create a unified pricing framework that not only overcomes the current limitations but also positions
Amazon at the forefront of e-commerce pricing strategy for years to come.

---

Overview of my solution:

Everything from Calculus I-III, for 
differential equations, and machine learning. 

1. Data Handling and Preprocessing:
   - Pandas (pd) is used for loading, manipulating, and analyzing structured data.
   - NumPy (np) provides support for large, multi-dimensional arrays and matrices, along with a collection
     of mathematical functions to operate on these arrays efficiently.

2. Machine Learning Pipeline:
   - Scikit-learn's train_test_split splits the data into training and testing sets.
   - StandardScaler normalizes the features, ensuring that all variables contribute equally to the model.
   - RandomForestRegressor is an ensemble learning method that constructs multiple decision trees and
     merges them to get a more accurate and stable prediction.
   - Mean squared error and R-squared score are used to evaluate the model's performance.

3. Advanced Mathematical Modeling:
   - SciPy's stats module provides probability distributions and statistical functions for modeling
     demand uncertainty and performing hypothesis tests.
   - SciPy's optimize module offers various optimization algorithms for finding optimal prices.
   - SciPy's integrate module allows for numerical integration, useful in calculating consumer and
     producer surplus.
   - SymPy enables symbolic mathematics, allowing us to perform calculus operations analytically.

4. Visualization:
   - Matplotlib is used to create plots and visualizations of our mathematical models and results.

Solution: 
"""

# Symbolic mathematics setup
P, Q = sp.symbols('P Q')  # Define symbolic variables for Price and Quantity

# Define a symbolic demand function
# Q = a - b*P + c*P^2, where a, b, and c are constants
a, b, c = sp.symbols('a b c')
demand_function_symbolic = a - b*P + c*P**2

# Revenue function (R = P * Q)
revenue_function_symbolic = P * demand_function_symbolic

# Marginal Revenue function (dR/dP)
marginal_revenue_symbolic = sp.diff(revenue_function_symbolic, P)

# Profit function (assuming a linear cost function)
fixed_cost, variable_cost = sp.symbols('FC VC')
cost_function_symbolic = fixed_cost + variable_cost * demand_function_symbolic
profit_function_symbolic = revenue_function_symbolic - cost_function_symbolic

# Optimal price (where dProfit/dP = 0)
optimal_price_symbolic = sp.solve(sp.diff(profit_function_symbolic, P), P)

# Consumer Surplus function
consumer_surplus_symbolic = sp.integrate(demand_function_symbolic - P, (P, 0, P))

# Producer Surplus function
producer_surplus_symbolic = revenue_function_symbolic - variable_cost * demand_function_symbolic

# Convert symbolic expressions to Python functions
def demand_function(P, a_val, b_val, c_val):
    return float(demand_function_symbolic.subs({a: a_val, b: b_val, c: c_val, P: P}))

def revenue_function(P, a_val, b_val, c_val):
    return float(revenue_function_symbolic.subs({a: a_val, b: b_val, c: c_val, P: P}))

def marginal_revenue_function(P, a_val, b_val, c_val):
    return float(marginal_revenue_symbolic.subs({a: a_val, b: b_val, c: c_val, P: P}))

def profit_function(P, a_val, b_val, c_val, fc_val, vc_val):
    return float(profit_function_symbolic.subs({a: a_val, b: b_val, c: c_val, P: P, fixed_cost: fc_val, variable_cost: vc_val}))

def consumer_surplus_function(P, a_val, b_val, c_val):
    return float(consumer_surplus_symbolic.subs({a: a_val, b: b_val, c: c_val, P: P}))

def producer_surplus_function(P, a_val, b_val, c_val, vc_val):
    return float(producer_surplus_symbolic.subs({a: a_val, b: b_val, c: c_val, P: P, variable_cost: vc_val}))

# Advanced calculus concepts: Multivariable optimization
def demand_function_3d(P, M, a_val, b_val, c_val, d_val):
    """
    3D Demand function: Q = f(P, M)
    Incorporates marketing spend (M) into the demand function
    Q = a - b*P + c*P^2 + d*log(1+M)
    """
    return a_val - b_val*P + c_val*P**2 + d_val*np.log(1+M)

def revenue_function_3d(P, M, a_val, b_val, c_val, d_val):
    """
    3D Revenue function: R = P * Q = P * f(P, M)
    """
    return P * demand_function_3d(P, M, a_val, b_val, c_val, d_val)

def gradient_revenue_3d(P, M, a_val, b_val, c_val, d_val):
    """
    Gradient of Revenue: [∂R/∂P, ∂R/∂M]
    """
    dR_dP = a_val - 2*b_val*P + 3*c_val*P**2 + d_val*np.log(1+M)
    dR_dM = P * d_val / (1 + M)
    return np.array([dR_dP, dR_dM])

# Differential Equations: Price dynamics model
def price_dynamics(P, t, a_val, b_val, c_val, k):
    """
    Differential equation model for price dynamics
    dP/dt = k * (Q_d - Q_s)
    where Q_d is the demand and Q_s is the supply
    k is a constant representing the speed of price adjustment
    """
    Q_d = demand_function(P, a_val, b_val, c_val)
    Q_s = a_val  # Assuming constant supply for simplicity
    return k * (Q_d - Q_s)

# Monte Carlo simulation for demand uncertainty
def monte_carlo_demand(num_simulations, P, a_val, b_val, c_val, std_dev):
    """
    Perform a Monte Carlo simulation to estimate demand under uncertainty
    """
    base_demand = demand_function(P, a_val, b_val, c_val)
    demands = np.random.normal(base_demand, std_dev, num_simulations)
    return demands

# Machine Learning model for price prediction
def train_price_model(X, y):
    """
    Train a Random Forest model for price prediction
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_price(model, X):
    """
    Predict prices using the trained model
    """
    return model.predict(X)

# Main execution
if __name__ == "__main__":
    # Set parameter values
    a_val, b_val, c_val = 1000, 20, 0.1
    fc_val, vc_val = 1000, 10
    d_val = 50  # For 3D demand function
    k = 0.1  # For price dynamics model

    # Generate price range
    prices = np.linspace(0, 100, 1000)

    # Calculate demand, revenue, and profit
    demands = [demand_function(p, a_val, b_val, c_val) for p in prices]
    revenues = [revenue_function(p, a_val, b_val, c_val) for p in prices]
    profits = [profit_function(p, a_val, b_val, c_val, fc_val, vc_val) for p in prices]

    # Find optimal price numerically
    optimal_price = optimize.minimize_scalar(lambda p: -profit_function(p, a_val, b_val, c_val, fc_val, vc_val), 
                                             bounds=(0, 100), method='bounded')
    print(f"Optimal Price: ${optimal_price.x:.2f}")

    # Calculate and print key metrics at optimal price
    opt_p = optimal_price.x
    print(f"Demand at Optimal Price: {demand_function(opt_p, a_val, b_val, c_val):.2f}")
    print(f"Revenue at Optimal Price: ${revenue_function(opt_p, a_val, b_val, c_val):.2f}")
    print(f"Profit at Optimal Price: ${profit_function(opt_p, a_val, b_val, c_val, fc_val, vc_val):.2f}")
    print(f"Consumer Surplus at Optimal Price: ${consumer_surplus_function(opt_p, a_val, b_val, c_val):.2f}")
    print(f"Producer Surplus at Optimal Price: ${producer_surplus_function(opt_p, a_val, b_val, c_val, vc_val):.2f}")

    # Solve price dynamics differential equation
    t = np.linspace(0, 100, 1000)
    P0 = 50  # Initial price
    sol = integrate.odeint(price_dynamics, P0, t, args=(a_val, b_val, c_val, k))

    # Perform Monte Carlo simulation
    mc_demands = monte_carlo_demand(10000, opt_p, a_val, b_val, c_val, 50)
    expected_demand = np.mean(mc_demands)
    demand_ci = stats.t.interval(0.95, len(mc_demands)-1, loc=np.mean(mc_demands), scale=stats.sem(mc_demands))
    print(f"Expected Demand (Monte Carlo): {expected_demand:.2f}")
    print(f"95% Confidence Interval for Demand: {demand_ci}")

    # Train machine learning model
    X = np.column_stack((prices, demands))
    y = revenues
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = train_price_model(X_train_scaled, y_train)
    y_pred = predict_price(model, X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared Score: {r2:.2f}")

    # Visualizations
    plt.figure(figsize=(20, 15))

    # Demand Curve
    plt.subplot(2, 2, 1)
    plt.plot(prices, demands)
    plt.title("Demand Curve")
    plt.xlabel("Price")
    plt.ylabel("Quantity Demanded")

    # Revenue Curve
    plt.subplot(2, 2, 2)
    plt.plot(prices, revenues)
    plt.title("Revenue Curve")
    plt.xlabel("Price")
    plt.ylabel("Revenue")

    # Profit Curve
    plt.subplot(2, 2, 3)
    plt.plot(prices, profits)
    plt.title("Profit Curve")
    plt.xlabel("Price")
    plt.ylabel("Profit")
    plt.axvline(x=opt_p, color='r', linestyle='--', label='Optimal Price')
    plt.legend()

    # Price Dynamics
    plt.subplot(2, 2, 4)
    plt.plot(t, sol)
    plt.title("Price Dynamics")
    plt.xlabel("Time")
    plt.ylabel("Price")

    plt.tight_layout()
    plt.show()

    # 3D Revenue Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    P_range = np.linspace(0, 100, 100)
    M_range = np.linspace(0, 100, 100)
    P_mesh, M_mesh = np.meshgrid(P_range, M_range)
    R_mesh = revenue_function_3d(P_mesh, M_mesh, a_val, b_val, c_val, d_val)
    surf = ax.plot_surface(P_mesh, M_mesh, R_mesh, cmap='viridis')
    ax.set_xlabel('Price')
    ax.set_ylabel('Marketing Spend')
    ax.set_zlabel('Revenue')
    ax.set_title('3D Revenue Function')
    fig.colorbar(surf)
    plt.show()
