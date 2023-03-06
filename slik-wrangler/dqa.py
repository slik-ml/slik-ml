from data_loading import LoadData


def missing_value_assessment(dataframe, display_findings=True):
    """
    Checks the missing values from the given dataset and generates
    a report of its findings.

    :param dataframe: pandas Dataframe. Data set to perform assessment on.
    :param display_findings: boolean, Default True. Whether to display a dataframe highlighting
        the missing values count and percentage.
    """

    LoadData().dqa(
        dataframe=dataframe,
        assessment_on='missing_values',
        log_message=display_findings
    )


def duplicate_assessment(dataframe, display_findings=True):
    """
    Checks the duplicate values from the given dataset and generates
    a report of its findings. It does this assessment for both rows
    and feature columns.

    :param dataframe: pandas Dataframe. Data set to perform assessment on.
    :param display_findings: boolean, Default True. Whether to display a dataframe highlighting
        the missing values count and percentage.
    """

    LoadData().dqa(
        dataframe=dataframe,
        assessment_on='duplicate_values',
        log_message=display_findings
    )


def outliers_assessment(dataframe, display_findings=True):
    """
    Checks for outliers in the given dataset and generates
    a report of its findings.

    :param dataframe: pandas Dataframe. Data set to perform assessment on.
    :param display_findings: boolean, Default True. Whether to display a dataframe highlighting
        the missing values count and percentage.
    """

    LoadData().dqa(
        dataframe=dataframe,
        assessment_on='outlier_values',
        log_message=display_findings
    )


def data_cleanness_assessment(dataframe, display_findings=True):
    """
    Checks for the overall cleanness of the dataframe:

    1. Checks if there are missing values in the dataset
    2. Checks if the dataset set contains any duplicates
    3. checks if there are outliers in the feature columns

    :param dataframe: pandas Dataframe. Data set to perform assessment on.
    :param display_findings: boolean, Default True. Whether to display a dataframe highlighting
        the missing values count and percentage.
    """

    LoadData().dqa(
        dataframe=dataframe,
        assessment_on='general_assessment',
        log_message=display_findings
    )
