#%%
##%% import necessary modules
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy.random as rn


##%% Q1 Statistics, 60pts

##%% Q1a, 15pts
rn.seed(0)
N = 16
m = 10**5

# Generate 100,000 experiments with 16 fair coin tosses each.
# Values greater than 0.5 are counted as heads.
nHeads = (rn.uniform(size=(m, N)) > 0.5).sum(axis=1)

# Plot the histogram and save the returned counts for later questions.
counts, bins, patches = plt.hist(nHeads, bins=range(18))
plt.xlabel('Number of Heads (k)')
plt.ylabel('Frequency')
plt.title('Q1a: Histogram of Heads in 16 Tosses')
plt.xticks(range(17))
plt.show()


##%% Q1b, 10pts
pmf_counts, pmf_bins, pmf_patches = plt.hist(
    nHeads,
    bins=range(18),
    density=True
)
plt.xlabel('Number of Heads (k)')
plt.ylabel('P(nHeads = k)')
plt.title('Q1b: Probability Mass Function')
plt.xticks(range(17))
plt.show()


##%% Q1c, 15pts
# Convert histogram counts to probabilities, then accumulate them.
empirical_pmf = counts / m
empirical_cdf = np.cumsum(empirical_pmf)
k = np.arange(17)

print('Empirical CDF:')
print(empirical_cdf)

plt.plot(k, empirical_cdf, marker='o')
plt.xlabel('k')
plt.ylabel('P(nHeads <= k)')
plt.title('Q1c: Cumulative Distribution Function')
plt.xticks(range(17))
plt.ylim(0, 1.05)
plt.grid(True)
plt.show()


##%% Q1d, 20pts
# Compute the theoretical CDF for a Binomial(n=16, p=0.5) distribution.
theoretical_cdf = stats.binom.cdf(k, N, 0.5)

print('x:', empirical_cdf)
print('y:', theoretical_cdf)

# Scatter plot: empirical CDF against theoretical CDF.
plt.scatter(empirical_cdf, theoretical_cdf)
plt.xlabel('Empirical CDF')
plt.ylabel('Theoretical CDF')
plt.title('Q1d: Scatter Plot')
plt.grid(True)
plt.show()

# Line plot: compare both CDFs for each value of k.
plt.plot(k, empirical_cdf, marker='o', label='Empirical CDF')
plt.plot(k, theoretical_cdf, marker='d', linestyle='--',
         label='Theoretical CDF')
plt.xlabel('k')
plt.ylabel('CDF')
plt.title('Q1d: Empirical and Theoretical CDFs')
plt.xticks(range(17))
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True)
plt.show()

# Log-log plot. Exclude the zero empirical probability because log(0)
# is undefined.
positive = empirical_cdf > 0
plt.loglog(
    empirical_cdf[positive],
    theoretical_cdf[positive],
    marker='o',
    linestyle='None'
)
plt.xlim(10**-5, 1)
plt.xlabel('Empirical CDF')
plt.ylabel('Theoretical CDF')
plt.title('Q1d: Log-Log Plot')
plt.grid(True, which='both')
plt.show()


##%% Q2 Linear Regression, 40pts

##%% Q2 i, 20pts
x_points = np.array([0.5, 2.0, 3.0])
y_points = np.array([1.0, 2.5, 3.0])

regression_models = {
    '(a) y = x + 0.5': (1.0, 0.5),
    '(b) y = x + 1': (1.0, 1.0),
    '(c) y = 0.8x + 0.3': (0.8, 0.3),
    '(d) y = 0.8x + 0.7': (0.8, 0.7)
}

sse_results = {}

for name, (slope, intercept) in regression_models.items():
    predicted_y = slope * x_points + intercept
    errors = y_points - predicted_y
    squared_errors = errors**2
    sse = np.sum(squared_errors)
    sse_results[name] = sse

    print('=' * 50)
    print(name)
    print('Actual y values:       ', y_points)
    print('Predicted y values:    ', predicted_y)
    print('Errors:                ', errors)
    print('Squared errors:        ', squared_errors)
    print('SSE:                   ', sse)

best_model = min(sse_results, key=sse_results.get)
print('\nBest regression model:', best_model)
print('Lowest SSE:', sse_results[best_model])


##%% Q2 ii.a, 15pts
years = np.array([2005, 2006, 2007, 2008, 2009], dtype=float)
sales = np.array([12, 19, 29, 37, 45], dtype=float)

x_mean = np.mean(years)
y_mean = np.mean(sales)

beta = np.sum((years - x_mean) * (sales - y_mean)) / np.sum(
    (years - x_mean)**2
)
alpha = y_mean - beta * x_mean

print('\nMean year:', x_mean)
print('Mean sales:', y_mean)
print('Slope (beta):', beta)
print('Intercept (alpha):', alpha)
print(f'Least-squares regression line: y = {beta:.2f}x + ({alpha:.2f})')


# Optional plot of the sales data and regression line.
predicted_sales = beta * years + alpha

plt.scatter(years, sales, label='Actual Sales')
plt.plot(years, predicted_sales, label='Regression Line')
plt.xlabel('Year')
plt.ylabel('Sales (Million Dollars)')
plt.title('Q2ii: Company Sales Regression')
plt.legend()
plt.grid(True)
plt.show()


##%% Q2 ii.b, 5pts
estimated_sales_2012 = beta * 2012 + alpha
print('Estimated sales in 2012 is', round(estimated_sales_2012, 2),
      'million dollars')
