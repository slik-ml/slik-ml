"""
This module is responsible for:
1. Reads a data (path) and able to load multiple dataset.
2. Determines if a dataset is small, medium, or large.
    If multiple datasets are entered, then the total size of the combined dataset
    is used to determine the size of the dataset.
3. The data is framed as a pandas dataframe (for small data) or spark dataframe (for large data).
4. Provides a processing data (A smaller subset of larger dataset) to be used for data processing.
5. Provides functionality to check the Data Quality (the dqa method).

The LoadData class currently supports the following extensions
[.csv, .xls, .xlsx, .json, .parquet, .sav, .dta]

Current Limitations
1. When loading multiple paths, different extensions with different kwarg values can't be properly handled yet.
2. TODO Can't handle very large datasets...
    Functionalities for keeping large data out of memory or limiting it's size needs to be developed
3.
"""

# Importing necessary tools

from messages import log

import os
import re
import csv
from functools import partial

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

from IPython.display import display

from matplotlib import pyplot as plt

# Setting constant variables

# Over a terabyte
BIG_DATA = 999999999999
# Over a gigabyte
MEDIUM_DATA = 999999999


# Data Loading Class
class LoadData:
    """
    Class responsible for loading dataset, and breaking it down
    into components, such that it is easy to review:

    1. The data size and data content
    2. Numeric and categorical features
    3. Checking the data quality and visualizing the results
    4. Vis
    """

    def __init__(
            self, path_to_data=None, target_column=None, names=None,
            use_spark=False, random_state=None, log_message=True, **kwargs
    ):
        """
        Initializes the LoadData Class

        :param path_to_data: type = str/list, default=None, path to the
            location of the dataset. Locally or Remotely.
        :param target_column: type = str, default=None, target column
            for the dataset
        :param names: type = list, default=None, used when multiple dataset
            are entered, a name is assigned to each dataset and represented
            as a dictionary in `data_s`
        :param use_spark: type = bool, default is False, if True, then spark is
            forced upon the dataset. pyspark is enforced if the dataset
            is considered to be large.
        :param random_state: type = int, default is None, random state that
            should be used throughout the data decomposition
        :param kwargs: Other useful arguments that can be used internally
            within the function, especially when loading the data
        """

        def _log_message(*messages, code='normal', sep=' ', end='\n'):
            """Logs messages upon initialization if log_message is True"""

            if log_message:
                if path_to_data is not None:
                    log(*messages, code=code, sep=sep, end=end)

        # Initializing variables
        self.data = self.data_s = self.spark_data = self.spark_data_s = \
            self.processing_data = self.processing_data_s = None
        self.target_column = target_column
        self.random_state = random_state
        _log_message("Initialized variables", code='info')

        # Stores data into variable
        if type(path_to_data) == str:
            self.data = self.read_data_from_path_pandas(path_to_data, ext=None, **kwargs)
            self.data_size = os.path.getsize(path_to_data)
        elif type(path_to_data) == list:
            self.data_s = self.__load_paths(path_to_data)
            if (names is not None) and \
                    (len(names) == len(self.data_s)) and \
                    all([type(name) == str for name in names]):
                self.data_s = {name: data for name, data in zip(names, self.data_s)}
            else:
                raise ValueError("The names for the datasets provided couldn't be used. Check the "
                                 "length that it matches the dataset and ensure the names are all string type")
            self.data_size = sum([os.path.getsize(path) for path in path_to_data])
        elif path_to_data is None:
            self.data = None
            self.data_size = 0
        else:
            raise ValueError(f"The 'path_to_data' can either be a str/list type not {type(path_to_data)}")

        # Classifying the numeric and categorical variables
        if path_to_data is not None:
            self.num_attr, self.cat_attr = self.__get_attributes()
            if target_column is not None:
                if target_column in self.num_attr:
                    self.num_attr.remove(target_column)
                elif target_column in self.cat_attr:
                    self.cat_attr.remove(target_column)
                else:
                    raise ValueError("Target column is not in data columns")
            else:
                _log_message("Target column wasn't provided!", code='danger')

        # Determines data size
        self.data_class = 'BIG' if self.data_size > BIG_DATA else 'MEDIUM' \
            if self.data_size > MEDIUM_DATA else 'SMALL'

        # Applies pyspark to large dataset
        if use_spark or self.data_class == 'BIG':
            self.__to_pyspark_df()

        # Determines if processing data was made available
        self.processing_data_is_available = bool(self.__add_processing_data())

        _log_message("Data loaded successfully!", code='success')

    @staticmethod
    def read_data_from_path_pandas(
            path, ext=None, **kwargs
    ):
        """
        Reads the dataset into a pandas dataframe.

        :param path: type = str, default=None, path to the
            location of the dataset. Locally or Remotely.
        :param ext: type = str, default=None, extension
            of the dataset which should be one of [csv, xls,
            xlsx, json, parquet, sav, dta] if None, then the
            extension is determined automatically.
        :param kwargs: Useful arguments used when loading the data.

        :return: Pandas dataframe of the data.
        """

        extensions = [
            'csv', 'xls', 'xlsx', 'json', 'parquet', 'sav', 'dta'
        ]
        extensions_functions = [
            partial(pd.read_csv, **kwargs),
            *([partial(pd.read_excel, **kwargs)] * 2),
            partial(pd.read_json, **kwargs),
            partial(pd.read_parquet, **kwargs),
            partial(pd.read_spss, **kwargs),
            partial(pd.read_stata, **kwargs)
        ]
        path_ext_checkers = {
            _ext: [re.match(fr'.*{_ext}$', path), _funcs]
            for _ext, _funcs in zip(extensions, extensions_functions)
        }

        if ext is not None:
            return path_ext_checkers[ext][1](path)
        else:
            for pec in path_ext_checkers:
                is_ext, load_func = path_ext_checkers[pec]

                if is_ext:
                    return load_func(path)

    @staticmethod
    def split_csv_file(
            file_path=None, delimiter=',', row_limit=1000000,
            output_path='.', keep_headers=True, output_name_template='output_%s.csv'
    ):
        """
        Split large csv files to small csv files.
        Function splits large csv files into smaller files based on the row_limit
        specified. The files are stored in present working dir by default.

        :param file_path: str/file path. path to where data is stored.
        :param delimiter: str. Default is ','. separator in each row and column,
        :param row_limit: int. split each file by row count
        :param output_path: str. output path to store split files
        :param keep_headers: Bool. Default is True, make use of headers for all csv files
        :param output_name_template: str. Default is 'output_%s.csv'. template for naming the output files

        :return: split files are stored in output_path
        """

        reader = csv.reader(open(file_path, 'r'), delimiter=delimiter)
        current_piece = 1
        current_out_path = os.path.join(output_path, (output_name_template % current_piece))
        current_out_writer = csv.writer(open(current_out_path, 'w'), delimiter=delimiter)
        current_limit = row_limit
        if keep_headers:
            headers = next(reader)
            current_out_writer.writerow(headers)
        for i, row in enumerate(reader):
            if i + 1 > current_limit:
                current_piece += 1
                current_limit = row_limit * current_piece
                current_out_path = os.path.join(output_path, (output_name_template % current_piece))
                current_out_writer = csv.writer(open(current_out_path, 'w'), delimiter=delimiter)
                if keep_headers:
                    current_out_writer.writerow(headers)
            current_out_writer.writerow(row)

    @staticmethod
    def __plot_nan(data):
        """
        Plot the top values from a value count in a dataframe.

        :param data: DataFrame or name Series. Data set to perform plot operation on.
        :return: A bar plot. The bar plot of top n values.
        """

        plot = data.sort_values(ascending=False)[:30]
        fig, ax = plt.subplots(figsize=(16, 9))

        ax.barh(plot.index, plot.values)

        for s in ['top', 'bottom', 'left', 'right']:
            ax.spines[s].set_visible(False)

        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        ax.xaxis.set_tick_params(pad=5)
        ax.yaxis.set_tick_params(pad=10)

        ax.grid(b=True, color='grey',
                linestyle='-.', linewidth=0.5,
                alpha=0.2)

        ax.invert_yaxis()

        for i in ax.patches:
            plt.text(i.get_width() + 0.2, i.get_y() + 0.5,
                     str(round((i.get_width()), 2)),
                     fontsize=10, fontweight='bold',
                     color='grey')

        ax.set_title('Chart showing the top 30 missing values in the dataset',
                     loc='left', )

        fig.text(0.9, 0.15, 'Alvaro', fontsize=12,
                 color='grey', ha='right', va='bottom',
                 alpha=0.7)

        plt.show()

    @staticmethod
    def __summarise_results(results):
        """
        Computes a summary of results given a list object

        :param results: type = list, results to be reduced
        :return: Returns the reduced list
        """

        limit = 10
        results = list(results)

        f = limit // 2
        ff = f + limit % 2

        results = results[:ff] + ['...'] + results[-f:] if len(results) > limit else results

        return '[' + ', '.join(map(str, results)) + ']'

    def __load_paths(
            self, paths, **kwargs
    ):
        """
        Loads multiple dataset at ones

        :param paths: type = str, default=None, path to the
            location of the dataset. Locally or Remotely.
        :param kwargs: Useful arguments used when loading
            each data.

        :return: list of loaded data
        """

        # TODO: Paths with different extensions needs to be handled

        return [
            self.read_data_from_path_pandas(
                path, ext=None, **kwargs
            )
            for path in paths
        ]

    def dqa(
            self, dataframe=None, assessment_on='general_assessment', plot=False,
            display_inline=True, log_message=True
    ):
        """
        Performs data quality assessment based on the assessment type
        1. `general_assessment`: Gives a general assessment
        2. `duplicate_values`: Gives a duplicate assessment
        3. `outlier_values`: Gives an outlier assessment
        4. `missing_values`: Gives a missing values assessment

        :param dataframe: pandas DataFrame, if not None, then DQA is performed
            on the passed dataframe
        :param assessment_on: type = str, default to `general_assessment`
            used to select the kind of assessment to be used
        :param plot:  type = bool, default to False, Used to plot observations
            from report
        :param display_inline: type = bool, default to True, Used to present
            an Ipython (tabular) display of the result
        :param log_message:  type = bool, default to True, Displays log messages
        """

        if assessment_on == 'general_assessment':
            issue_checker = {
                'missing values': self.__missing_value_assessment,
                'duplicate variables': self.__duplicate_assessment,
                'outliers': self.__outlier_assessment
            }

            for issue in issue_checker:
                log(f"Checking for {issue}", end="\n\n", code='info')
                issue_checker[issue](
                    dataframe=dataframe, plot=plot,
                    display_inline=display_inline,
                    log_message=log_message
                )
                log(end="\n\n")

        elif assessment_on == 'duplicate_values':
            self.__duplicate_assessment(
                dataframe=dataframe, plot=plot,
                display_inline=display_inline,
                log_message=log_message
            )
        elif assessment_on == 'outlier_values':
            self.__outlier_assessment(
                dataframe=dataframe, plot=plot,
                display_inline=display_inline,
                log_message=log_message
            )
        elif assessment_on == 'missing_values':
            self.__missing_value_assessment(
                dataframe=dataframe, plot=plot,
                display_inline=display_inline,
                log_message=log_message
            )
        else:
            raise ValueError("Assessment Type Unknown!")

    def __add_processing_data(self):
        """Creates a subset of the data that can be used in the preprocessing stage"""

        if self.data_class != 'SMALL':
            if self.data is not None:
                if self.data_class == 'MEDIUM':
                    n_sample = int(len(self.data) * .1)
                else:
                    n_sample = int(len(self.data) * .01)

                self.processing_data = self.data.sample(
                    n=n_sample, random_state=self.random_state
                )

            if self.data_s is not None:
                if type(self.data_s) == dict:
                    self.spark_data_s = {
                        name: self.data_s[name].sample(
                            n=int(len(self.data_s[name]) * .1),
                            random_state=self.random_state
                        )
                        for name in self.data_s
                    }
                else:
                    self.spark_data_s = [
                        data.sample(
                            n=int(len(data) * .1),
                            random_state=self.random_state
                        )
                        for data in self.data_s
                    ]

            log("Data too large! Processing data has been made available for processing!", code='info')
        else:
            if self.data is not None:
                self.processing_data = self.data

            if self.data_s is not None:
                self.processing_data_s = self.data_s

        return self.data_class != 'SMALL'

    def __duplicate_assessment(
            self, dataframe=None, plot=False, display_inline=True, log_message=True
    ):
        """
        Performs duplicate assessment

        :param dataframe: pandas DataFrame, if not None, then DQA is performed
            on the passed dataframe
        :param plot:  type = bool, default to False, Used to plot observations
            from report
        :param display_inline: type = bool, default to True, Used to present
            an Ipython (tabular) display of the result
        :param log_message:  type = bool, default to True, Displays log messages
        """

        if dataframe is not None:
            df = dataframe.copy()
            t_df = df.T.copy()
        else:
            df = self.data.copy()
            t_df = self.data.T.copy()

        duplicated_columns = t_df[t_df.duplicated()]
        duplicated_rows = df[df.duplicated()]

        if len(duplicated_columns):
            self.duplicated_columns = list(duplicated_columns.index)

            if plot:
                # TODO: Add plotting functions for duplicated columns
                pass

            if log_message:
                log(
                    f"Dataframe contains duplicate columns that you should address. "
                    f"\n\ncolumns={self.__summarise_results(list(self.duplicated_columns))}\n",
                    code='warning'
                )

            if display_inline:
                display(duplicated_columns.T)

        if len(duplicated_rows):
            self.duplicated_rows = list(duplicated_rows.index)

            if plot:
                # TODO: Add plotting functions for duplicated rows
                pass

            if log_message:
                log(
                    f"Dataframe contains duplicate rows that you should address. "
                    f"\n\nrows={self.__summarise_results(list(self.duplicated_rows))}\n",
                    code='warning'
                )

            if display_inline:
                display(duplicated_rows)

        if not len(duplicated_rows) and not len(duplicated_columns):
            log("No duplicate values in both rows and columns!!!", code='success')

    def __get_attributes(self, dataframe=None):
        """
        Gets the numerical and categorical attributes from the data

        :return: the categorical features and Numerical features(in a pandas dataframe) as a list.
        """

        if dataframe is not None:
            num_attributes = dataframe.select_dtypes(include=np.number).columns.tolist()
            cat_attributes = [x for x in dataframe.columns if x not in num_attributes]
        else:
            num_attributes = self.data.select_dtypes(include=np.number).columns.tolist()
            cat_attributes = [x for x in self.data.columns if x not in num_attributes]

        return num_attributes, cat_attributes

    def __missing_value_assessment(
            self, dataframe=None, plot=False, display_inline=True, log_message=True
    ):
        """
        Performs missing value assessment

        :param dataframe: pandas DataFrame, if not None, then DQA is performed
            on the passed dataframe
        :param plot:  type = bool, default to False, Used to plot observations
            from report
        :param display_inline: type = bool, default to True, Used to present
            an Ipython (tabular) display of the result
        :param log_message:  type = bool, default to True, Displays log messages
        """

        if dataframe is not None:
            df = dataframe.isna().sum().reset_index()
            df.columns = ['features', 'missing_counts']
            df['missing_percent'] = round((df['missing_counts'] / (dataframe.shape[0]) * 100), 1)
        else:
            df = self.data.isna().sum().reset_index()
            df.columns = ['features', 'missing_counts']
            df['missing_percent'] = round((df['missing_counts'] / (self.data.shape[0]) * 100), 1)

        if len(df):
            self.missing_value_assessment = df

        if log_message:
            if len(df):
                log(
                    f"Dataframe contains missing values that you should address. "
                    f"\n\ncolumns={self.__summarise_results(list(df.index))}\n",
                    code='warning'
                )
            else:
                log("No missing values!!!", code='success')

        if plot:
            self.__plot_nan(df.set_index('features')['missing_percent'])

        if display_inline:
            display(df)

    def __outlier_assessment(
            self, dataframe=None, plot=False, display_inline=True, log_message=True
    ):
        """
        Performs outlier assessment

        :param dataframe: pandas DataFrame, if not None, then DQA is performed
            on the passed dataframe
        :param plot:  type = bool, default to False, Used to plot observations
            from report
        :param display_inline: type = bool, default to True, Used to present
            an Ipython (tabular) display of the result
        :param log_message:  type = bool, default to True, Displays log messages
        """

        if dataframe is not None:
            df = dataframe.copy()
        else:
            df = self.data.copy()

        num_attr, _ = self.__get_attributes(df)

        def contains_outliers(column):
            """Checks if the given column contains outliers"""

            df.loc[:, column] = abs(df[column])

            q25 = np.percentile(df[column].dropna(), 25)
            q75 = np.percentile(df[column].dropna(), 75)

            outlier_cut_off = ((q75 - q25) * 1.5)
            lower_bound, upper_bound = (q25 - outlier_cut_off), (q75 + outlier_cut_off)

            outlier_list_col = df[column][(df[column] < lower_bound) | (df[column] > upper_bound)].index

            return bool(len(outlier_list_col))

        outliers = [column for column in num_attr if contains_outliers(column)]

        if len(outliers):
            self.outliers = outliers

            if plot:
                # TODO: Create a plot for the missing outlier
                pass

            if log_message:
                log(
                    f"Ignore target column, if target column is considered an outlier\n",
                    code="info"
                )
                log(
                    f"Dataframe contains outliers that you should address. "
                    f"\n\ncolumns={self.__summarise_results(contains_outliers)}\n",
                    code='warning'
                )

            if display_inline:
                display(df[contains_outliers].head())
        else:
            if log_message:
                log("No outliers in dataset!!!", code='success')

    def __to_pyspark_df(self):
        """Converts pandas dataframe to pyspark dataframe"""

        self.use_spark = True

        try:
            spark = SparkSession.builder.getOrCreate()

            if self.data is not None:
                self.spark_data = spark.createDataFrame(self.data)

            if self.data_s is not None:
                if type(self.data_s) == dict:
                    self.spark_data_s = {
                        name: spark.createDataFrame(self.data_s[name])
                        for name in self.data_s
                    }
                else:
                    self.spark_data_s = [
                        spark.createDataFrame(data)
                        for data in self.data_s
                    ]

            log("Applied pyspark to large data!", code='info')
        except RuntimeError:
            log(
                "Runtime Error! Do you have java installed?",
                "Check https://stackoverflow.com/questions/31841509/pyspark-exception-java-gateway-process-exited-"
                "before-sending-the-driver-its-po",
                sep='\n', code='danger'
            )
