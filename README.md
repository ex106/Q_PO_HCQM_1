# Quantum Portfolio Optimization

A portfolio optimization implementation using D-Wave's Hybrid CQM (Constrained Quadratic Model) solver, designed to find optimal asset allocation while balancing returns and risk.

## Overview

This project implements a portfolio optimization strategy using quantum computing techniques, specifically:
- D-Wave's Hybrid CQM solver for portfolio optimization
- Historical stock data analysis using yfinance
- Risk-return optimization with multiple constraints
- Real-time market data integration

## Features

- Modern Portfolio Theory (MPT) implementation
- Multiple constraint handling:
  - Portfolio weights sum to 1
  - Individual asset weight constraints
  - Risk (volatility) constraints
  - Skewness and kurtosis constraints
- Integration with real market data through yfinance
- D-Wave quantum computing optimization

## Requirements

```python
dwave-ocean-sdk
pandas
numpy
yfinance