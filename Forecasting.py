import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import pymc3 as pm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Function to get market data
def get_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Returns'] = data['Adj Close'].pct_change().dropna()
    return data

# Function to fit Hidden Markov Model
def fit_hmm(data):
    model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000).fit(data[['Returns']])
    hidden_states = model.predict(data[['Returns']])
    data['Regime'] = hidden_states
    return model, data

# Function for Bayesian network
def bayesian_network(data):
    with pm.Model() as model:
        mu = pm.Normal('mu', mu=0, sigma=1)
        sigma = pm.HalfNormal('sigma', sigma=1)
        returns = pm.Normal('returns', mu=mu, sigma=sigma, observed=data['Returns'])
        trace = pm.sample(2000, return_inferencedata=False)
    return trace

# Function to calculate Value at Risk (VaR)
def calculate_var(returns, confidence_level=0.95):
    var = np.percentile(returns, (1-confidence_level)*100)
    return var

# Function to backtest a trading strategy
class Backtest:
    def __init__(self, data, strategy_func):
        self.data = data
        self.strategy_func = strategy_func

    def run(self):
        self.data['Signal'] = self.strategy_func(self.data)
        self.data['Strategy Returns'] = self.data['Signal'] * self.data['Returns']
        self.data['Cumulative Returns'] = (1 + self.data['Strategy Returns']).cumprod()
        return self.data

# Example trading strategy based on regimes
def example_strategy(data):
    data['Signal'] = np.where(data['Regime'] == 1, 1, 0)  # Buy in regime 1
    return data['Signal']

# Main execution
if __name__ == "__main__":
    # Fetch data
    ticker = 'AAPL'
    start_date = '2022-01-01'
    end_date = '2023-01-01'
    data = get_data(ticker, start_date, end_date)

    # Fit Hidden Markov Model
    hmm_model, data = fit_hmm(data)

    # Bayesian network analysis
    trace = bayesian_network(data)
    pm.plot_posterior(trace)
    plt.show()

    # Backtest the strategy
    backtest = Backtest(data, example_strategy)
    results = backtest.run()

    # Calculate Value at Risk (VaR)
    var = calculate_var(data['Returns'])
    print(f"Value at Risk (VaR) at 95% confidence level: {var}")

    # Plot cumulative returns
    results[['Cumulative Returns']].plot()
    plt.title('Cumulative Returns')
    plt.show()
