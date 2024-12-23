import pandas as pd
from scipy.stats import pearsonr


class data_preprocess:

    def __init__(self, file_path):
        self.file_path = file_path
        self.seasonal_correlations = None
        self.drought_years = None
        self.std_dev_rainfall = None
        self.mean_rainfall = None
        self.extreme_rainfall_years = None
        self.seasonal_avg = None
        self.lowest_rainfall_month = None
        self.highest_rainfall_month = None
        self.monthly_avg = None
        self.data = None
        self.rows = None
        self.columns = None
        self.month_stats = None
        self.season_stats = None
        self.load_data()
        self.data_exploration()
        self.feature_extraction()

    def load_data(self):
        self.data = pd.read_csv(self.file_path)

    def data_exploration(self):
        self.rows = self.data.shape[0]
        self.columns = self.data.shape[1]
        self.month_stats = self.data.iloc[:, :12].describe()
        self.season_stats = self.data.iloc[:, 12:].describe()

    def feature_extraction(self):
        monthly_columns = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        self.monthly_avg = self.data[monthly_columns].mean()
        self.highest_rainfall_month = self.monthly_avg.idxmax()
        self.lowest_rainfall_month = self.monthly_avg.idxmin()
        seasonal_columns = ['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']
        self.seasonal_avg = self.data[seasonal_columns].mean()
        self.data['10-Year Rolling Avg'] = self.data['ANNUAL'].rolling(window=10).mean()
        self.data['15-Year Rolling Avg'] = self.data['ANNUAL'].rolling(window=10).mean()
        self.data['20-Year Rolling Avg'] = self.data['ANNUAL'].rolling(window=10).mean()
        self.mean_rainfall = self.data['ANNUAL'].mean()
        self.std_dev_rainfall = self.data['ANNUAL'].std()
        self.drought_years = self.data[self.data['ANNUAL'] < (self.mean_rainfall - 1.5 * self.std_dev_rainfall)]
        self.extreme_rainfall_years = self.data[self.data['ANNUAL'] > (
                self.mean_rainfall + 1.5 * self.std_dev_rainfall)]
        seasonal_columns = ['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec']
        self.seasonal_correlations = {
            season: pearsonr(self.data[season], self.data['ANNUAL'])[0] for season in seasonal_columns
        }
        self.data['10-Year Rolling Avg'] = self.data['ANNUAL'].rolling(window=10).mean()


if __name__ == '__main__':
    pass
