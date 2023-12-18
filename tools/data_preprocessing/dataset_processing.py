import re
import os
import pandas as pd


def is_valid_sequence(sequence):
    pattern = re.compile('[^ARNDCQEGHILKMFPSTWYV]')
    return not pattern.search(sequence)


def load_and_validate_dataset(dataset):
    # Check if the file exists
    if not os.path.exists(dataset):
        raise FileNotFoundError(f"The file '{dataset}' does not exist at the specified path.")

    # Read the CSV file and load the data into a DataFrame
    data = pd.read_csv(dataset)

    # Verify that the columns are 'id', 'sequence', 'activity', and 'partition'
    expected_columns = ['id', 'sequence', 'activity', 'partition']
    if not all(column in data.columns for column in expected_columns):
        raise ValueError("The CSV file must contain columns 'id', 'sequence', 'activity', and 'partition'.")

    # Eliminate synthetic sequences
    valid_mask = data['sequence'].apply(is_valid_sequence)
    data = data[valid_mask]

    # Verify that the database not is empty
    if data.shape[0] < 1:
        raise ValueError("The database is empty. There are no data rows in the CSV file.")

    # Remove spaces from the 'id' and 'sequence' columns
    data['id'] = data['id'].str.replace(' ', '')
    data['sequence'] = data['sequence'].str.replace(' ', '')

    # Verify that the 'activity' column only contains values 0 or 1
    if not set(data['activity']).issubset({0, 1}):
        raise ValueError("The 'activity' column must contain only values 0 or 1.")

    # Verify that the 'partition' column only contains values 1, 2, or 3
    if not set(data['partition']).issubset({1, 2, 3}):
        raise ValueError("The 'partition' column must contain only values 1, 2, or 3.")

    # Filters out instances with sequence lengths between 5 and 30 amino acids.
    #data = data[data['sequence'].str.len().between(5, 30)]

    return data