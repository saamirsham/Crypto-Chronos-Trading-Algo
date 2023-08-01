# CryptoChronos: AI-Powered Trading Algorithm for Crypto Assets

Welcome to CryptoChronos, a cutting-edge, AI-powered trading algorithm designed specifically for crypto assets. Built with precision and accuracy, this project leverages the advanced power of artificial intelligence to initiate trades on a multitude of cryptocurrencies, based on their lag time to 5% pumps in Bitcoin's value. The objective of this algorithm is to anticipate and capitalize on crypto market dynamics, thereby offering optimized trade executions and potentially increasing returns. Whether you're an avid crypto trader or a budding enthusiast, CryptoChronos stands ready to bring a new level of sophistication to your trading experience.


## Table of Contents
1. [Project Overview](#project-overview)
2. [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
3. [Scripts Breakdown](#scripts-breakdown)
    * [Data Pulling and Organising Scripts](#data-pulling-and-organising-scripts)
        * [01_data_puller.ipynb](#01_data_puller.ipynb)
        * [02_db_builder.ipynb](#02_db_builder.ipynb)
        * [03_sql_query.ipynb](#03_sql_query.ipynb)
    * [Pre-processing Script](#pre-processing-script)
    * [Modeling Notebook](#modeling-notebook)
    * [Backtesting Notebook](#backtesting-notebook)
4. [Usage Instructions](#usage-instructions)
5. [Conclusion](#conclusion)
6. [Contributing](#contributing)
7. [License](#license)
8. [Contact](#contact)


## Project Overview

This project offers a comprehensive guide on how to apply machine learning techniques to cryptocurrency trading. The pipeline consists of data preparation, model development, and finally, backtesting of the trading strategy.

Cryptocurrencies have revolutionized the way we think about money, providing a decentralized solution for financial transactions. Their underlying technology, blockchain, offers transparency and security features that traditional banking systems often lack. However, due to their volatile nature, trading cryptocurrencies comes with substantial risk. In this project, we aim to mitigate this risk by developing a machine learning model that can accurately predict significant price changes in various coins based on 5% pumps of BTC on the one hour time frame.

## Getting Started

### Prerequisites

To run this project, you'll need the following:
- Python 3.x
- Libraries: Pandas, NumPy, Matplotlib, sklearn, statsmodels, TensorFlow

### Installation

1. Clone the repo
```
git clone https://github.com/your_username/repo_name.git
```
2. Install the required packages
```
pip install -r requirements.txt
```

## Scripts Breakdown

### Data Pulling and Organising Scripts 

##### *01_data_puller.ipynb*

Now that you have installed all the prerequisites, you can start using CryptoChronos. The main function in this script is `get_hourly_data`. It fetches hourly historical data for a given cryptocurrency symbol. Here's how to use it:

```python
symbol = "BTC"  # Replace with your cryptocurrency symbol
to_symbol = "USD"  # Replace with your desired currency symbol
limit = 100  # The number of data points to fetch
toTs = int(time.time())  # Replace with the timestamp up to which data is to be fetched
aggregate = 1  # The data aggregation level

df = get_hourly_data(symbol, to_symbol, limit, toTs, aggregate)
print(df)
```

This will print a pandas DataFrame with the historical data for the given cryptocurrency.

The `currency_data_puller` function allows for the retrieval and storage of extended historical data for a specified cryptocurrency. It fetches the hourly data for a given symbol from a specified date to 24/07/2023. It then stores this data in a CSV file. Here's an example of how you can use it:

```python
symbol = "BTC"  # Replace with your cryptocurrency symbol

currency_data_puller(symbol)
```

This will fetch the data and store it in a CSV file named `{symbol}_price_data.csv` (for example, 'BTC_price_data.csv' for Bitcoin) in a directory called `../data/raw/`. Please make sure to create this directory if it doesn't exist, or adjust the path to your preference. You should also adjust the date range if needed.

The script is equipped to fetch data for multiple cryptocurrencies in one run. In the list `currency_list`, you can add the symbols of all the cryptocurrencies you wish to pull data for. Here's an example:

```python
currency_list =  ["BTC", "ETH", "XRP", "LTC", "ADA", "DOT", "LINK", "BNB", "UNI", "DOGE"]

for i in currency_list:
    print(f'Starting to pull {i} data')
    currency_data_puller(i)
    print(f'Finished pulling {i} data')
    print('-'*20)
```

This will pull and store the data for each cryptocurrency listed in `currency_list`, informing you of the process as it runs.

##### *02_db_builder.ipynb*

The `02_db_builder` script is designed to create and populate a SQLite database from the CSV files created in the data pulling stage. 

Here is a brief summary of the steps involved:

1. **Database Creation:** The script first creates a SQLite database named `crypto_database.db` in the `../data/` directory using the `sqlite3` library.

2. **Data Import:** Then, it loads the Bitcoin CSV data into a pandas DataFrame and transfers it into a new table named 'BTC' in the database. 

3. **Table Creation and Data Transfer:** The script then goes through a list of other cryptocurrency symbols. For each symbol, it creates a new table in the database and fills it with data imported from the respective CSV file. 

Here's a quick breakdown of how to use the script:

- Install the required libraries: `sqlite3` and `pandas`.

- Run the script from the directory containing your CSV files.

- Check the `crypto_database.db` file in the `../data/` directory. Each cryptocurrency should have its own table filled with the data from the respective CSV file. 

##### *03_sql_query.ipynb*

The `03_sql_query` script performs the following steps:

1. **Database Connection:** It connects to the SQLite database `crypto_database.db`.

2. **SQL Query Execution:** It creates and executes an SQL query to merge data from the Bitcoin table with each other cryptocurrency table in the database based on the 'time' column. 

3. **Data Export:** The merged data is saved to new CSV files in the `../data/merged_data/` directory. One CSV file is created for each cryptocurrency. 

4. **Merging All Cryptocurrencies:** In the last part, it merges the data from all the cryptocurrencies and Bitcoin into a single DataFrame and saves it as a CSV file named `all_currencies.csv` in the `../data/merged_data/` directory. 

To use this script, ensure the SQLite database is in the right path and the necessary libraries are installed. Then, run the script to generate the CSV files with the merged data. 


### Pre-processing Script (01_pre-processing)

The `01_pre-processing` script performs the following steps:

1. **Data Loading & Processing:** It loads and processes CSV files from the `../data/merged_data/` directory.

2. **Calculate Percentage Changes:** It calculates the percentage changes for each cryptocurrency pair over a given number of hours.

3. **Clustering:** It performs clustering on the percentage changes using the KMeans algorithm.

4. **Granger Causality Tests:** It performs Granger causality tests to identify any potential cause-and-effect relationships.

5. **Find Windows After Rise:** It identifies time windows after Bitcoin value rises above a specified threshold.

6. **Find Lag Times:** It calculates the lag times between a rise in Bitcoin and a subsequent rise in another cryptocurrency.

7. **Save Processed Data:** It saves the final processed data to CSV files in the `../data/ready/` directory.

The `01_pre-processing` script involves several steps to process and analyze the merged cryptocurrency data. This script primarily focuses on calculating percentage changes in the cryptocurrency values, clustering the changes, conducting Granger causality tests, and extracting windows of data after specific thresholds. The script saves the processed data in CSV files in the `../data/ready/` directory.

Here's a breakdown of each function:

1. `load_and_process_data(file_path)`: This function reads the CSV file from the given `file_path`, sets the 'time' column as the index of the DataFrame, and returns the DataFrame.

2. `calculate_percentage_change(df, ticker, hours)`: This function calculates the percentage change in the opening and closing prices of Bitcoin and the specified cryptocurrency over a given number of hours. The new columns with percentage changes are added to the DataFrame, which is then returned.

3. `perform_clustering(df, ticker)`: This function performs K-means clustering on the percentage changes in the specified cryptocurrency and Bitcoin over a six-hour period. It returns the fitted KMeans object and the DataFrame used for clustering.

4. `visualize_clusters(km, X)`: This function visualizes the clusters obtained from the K-means clustering using a scatter plot.

5. `perform_granger_causality_tests(df, ticker)`: This function performs Granger causality tests on the percentage changes in Bitcoin and the specified cryptocurrency over one hour. It returns the test results.

6. `find_windows_after_rise(df, threshold)`: This function finds six-hour windows in the data following times when the percentage change in Bitcoin exceeded the given threshold. It returns a DataFrame containing these windows.

7. `find_lag_times(windows_after_rise, threshold, ticker)`: This function finds the lag time between a rise in Bitcoin value and a subsequent rise in the specified cryptocurrency value that exceeds a given threshold. It returns a DataFrame containing the lag times and whether the threshold was met.

Finally, the script loops over all the CSV files, applies these functions, and saves the resulting DataFrames in the `../data/ready/` directory. 

To use this script, you should have all necessary libraries installed (Pandas, NumPy, Matplotlib, sklearn, statsmodels) and CSV files for each cryptocurrency located in the `../data/merged_data/` directory. Then run the script to generate the preprocessed CSV files in the `../data/ready/` directory. 

### Modeling Notebook

The modeling notebook is divided into two parts. The first part is about classifying whether a price change meets a certain threshold within the next six hours, while the second part is predicting how long it will take to meet the threshold if it is met. 

There is a custom model for each of the crypto-currencies we are analysing. All models use neural nets in both setps of the modelling. the below explaination is for the 'ADA_model' notebook.

Here's a step-by-step summary:

1. It begins by importing necessary libraries. These include pandas, numpy, and various modules from TensorFlow, sklearn, and matplotlib.

2. The prepared data for Cardano (ADA) is loaded. The 'met_threshold' column, which is the binary target for whether the price increase threshold was met, is separated from the features.

3. The features are scaled using StandardScaler from sklearn. This helps to prevent features with larger scales from dominating those with smaller scales during the model fitting process.

4. The data is split into training and testing sets. 

5. A Sequential model from TensorFlow's Keras API is built for the classification task. The model consists of two dense layers with 64 neurons each, each followed by a dropout layer. The final layer is a single neuron with a sigmoid activation function, which is appropriate for binary classification tasks.

6. The model is compiled with Adam optimizer and binary cross-entropy loss function. It is then fitted using the training data, with the testing data used for validation.

7. The model's accuracy and loss progression is then plotted for both the training and testing sets.

8. The model's overall accuracy is computed using the sklearn's accuracy_score function. 

9. A confusion matrix is plotted using seaborn, which helps in understanding the model's performance in terms of true positives, false positives, true negatives, and false negatives.

10. The next step is building a regression model to predict the lag time for price increase. The same 'met_threshold' column is used, but now the target is 'lag_time' - the time it takes for the price to increase.

11. The Sequential model for this task consists of two dense layers with 64 neurons each, with dropout and L2 regularization applied. The final layer is a single neuron, which is standard for regression tasks. The model is compiled with Adam optimizer and mean squared error loss function.

12. The model is then fitted using the training data, with the testing data used for validation. Early stopping is also applied during training to prevent overfitting.

13. The model's performance is evaluated by plotting actual vs predicted values for the test set, and by plotting the progression of the model's loss.

14. Predictions are made on the full dataset and then added to the dataframe. 

15. The dataframe with predictions is saved to a new CSV file.

In the final line, the model's predictions are saved to a CSV file. 

This modeling process is repeated for each cryptocurrency in the study. 

This approach allows us to build a model that can predict both whether a significant price increase will occur within the next six hours, and if so, how long it will take to reach that increase.

### Backtesting Notebook

In this last notebook, you are running a backtest to evaluate the trading performance based on the model's predictions. The notebook does the following:

1. The prepared data and the predicted data for Cardano (ADA) are loaded, suffixed for differentiation, and then merged together.

2. A `backtest` function is defined. This function simulates trading based on the model's predictions with an initial capital of 1000 units. The function loops through each row in the dataframe. If the predicted 'met_threshold' is 0 (meaning the price change threshold is not expected to be met), or the predicted 'lag_time' is 0 or NaN, it skips the trade.

3. For trades that are not skipped, it checks if executing the trade would lead us beyond the range of the dataframe. If so, it skips the trade.

4. For each trade that is executed, it calculates the start and end prices, the opening and closing times, the percentage change in price, the profit or loss, and the remaining capital after the trade. It appends all these details to a `trade_history` list.

5. After looping through all rows, the function returns a dataframe constructed from the `trade_history` list, which shows the outcome of each executed trade.

6. This `backtest` function is then applied to the ADA data, and the output is displayed.

7. The notebook then defines a dictionary `csv_files` that maps each cryptocurrency ticker to its corresponding CSV file path. It extracts the list of tickers from the keys of this dictionary.

8. For each ticker in the list of tickers, it loads the prepared data and the predicted data, merges them together, and applies the `backtest` function. The result is saved to a new CSV file in the 'portfolio-performance' directory.

9. Finally, it loads and displays the portfolio performance of DOGE as an example. 

This backtesting process is essential in assessing the profitability and risk of the trading strategy suggested by the model's predictions. It's important to remember that while backtesting can provide valuable insights, past performance is not always indicative of future results due to the dynamic nature of financial markets.

### Conclusion

This project provides an in-depth exploration of how to leverage machine learning to create informed cryptocurrency trading strategies. Our pipeline seamlessly integrates the facets of data preparation, model development, and strategy backtesting to offer a holistic approach to trading strategy development.

Through our rigorous data pre-processing, we've aimed to ensure that the information feeding into our machine learning models is clean, relevant, and structured in a way that enables efficient learning. We've taken advantage of state-of-the-art algorithms and clustering techniques to dissect the intricacies of the crypto market dynamics, extracting hidden patterns and potential causal relationships.

Our custom-built models, one for each cryptocurrency in the study, were crafted with care, using neural networks to handle both classification and regression tasks. The models aim to predict whether a certain price threshold will be met in the near future and, if so, the anticipated time frame for such an event.

Lastly, our backtesting mechanism offers a simulated trading environment to validate the potential profitability and risk of our machine learning-informed trading strategies. This step is crucial in assessing the practical applicability of our models in real-world scenarios.

However, it is crucial to understand the inherent risks associated with cryptocurrency trading. The nature of financial markets is inherently unpredictable, with high volatility being a particular characteristic of the crypto market. Therefore, while our project provides a systematic way of developing trading strategies, it is by no means a guarantee of future profit. Our project is best used as a learning tool, showcasing how machine learning can be applied to financial data, rather than as a definitive guide to cryptocurrency trading. Always conduct your own research and consider consulting with financial advisors when dealing with investments.


## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request



## Contact

Your Name - shamsie@usc.edu

Project Link: https://git.generalassemb.ly/ssaamirs/Capstone

