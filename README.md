# 📈 Financial Mathematics - Options Pricing

This repository implements option pricing models, ranging from classical models to newer machine learning models.

## Features

- 📉 **Black Scholes Model** (European Calls and Puts)
- 📊 **Greeks** (Delta, Gamma, Vega, Theta, Rho)
- 🎰 **Monte Carlo Pricing** with Geometric Brownian Motion
- 🖼️ **Path Visualisation** and Payoff Plotting
- 👨‍💻 **Data Gathering** for Stocks and their respective Options

## 🛣️ Roadmap
### Stage One
- ✅ Implement Black-Scholes model for European Call and Put Options
- ✅ Implement Greeks
- ✅ Implement Data Fetching using Yahoo Finance for Option Chain data
- ✅ Implement Monte Carlo approach with Geometric Brownian Motion (GBM)
- ✅ Add Stock Price visualisation and GBM visualisation

### Stage Two
- Expand Market Data:
  - ✅ Clean Bad Implied Volatilities (fall back to Historic Volatilities)
  -  Select near-the-money strikes
-  Compare model prices vs market quotes

### Stage Three
- Add Feature Engineering (log-moneyness, normalised T, volatility surface features) and their Visualisations
- Implement Longstaff-Schwartz Monte Carlo for pricing American Options
- Add Exotic Options
- Extend Monte Carlo approach to Bermudan and Asian Options

### Stage Four
- Implement Machine Learning Models:
    - Logistic Regression
    - Random Forest
    - Support Vector Machine
- Visualise predicted vs actual prices

### Stage Five
- Implement Deep Learning Models:
    - MLP
    - LSTM
    - CNN
- Explore and Experiment with Reinforcement Learning

## References

[1] Black, F., & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.* Journal of Political Economy, 81(3), 637–654.  
[2] Longstaff, F. A., & Schwartz, E. S. (2001). *Valuing American Options by Simulation: A Simple Least-Squares Approach.* Review of Financial Studies, 14(1), 113–147.  
[3] Hull, J. C. (2018). *Options, Futures, and Other Derivatives.* 10th Edition, Pearson.  
