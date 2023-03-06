from data_loading import LoadData


def read_file(file_path, **kwargs):
    """
    Load a file path into a dataframe.

    This function takes in a file path - csv, excel, json, spss, stata data,
    or parquet and reads the data based on the input columns specified.
    Can only load  one file at a time.

    :param file_path: path to the dataset
    :param kwargs: pandas parameters for adjusting the dataframe

    :return: pandas DataFrame
    """

    return LoadData.read_data_from_path_pandas(file_path, **kwargs)


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

    LoadData.split_csv_file(
        file_path=file_path, delimiter=delimiter, row_limit=row_limit,
        output_path=output_path, keep_headers=keep_headers,
        output_name_template=output_name_template
    )
