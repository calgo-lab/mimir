import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List


def process_column_names(string_list):
    """
    RENUVER expects column names that contain no special characters
    and no numberes. This function transforms existing column
    names accordingly.
    
    Args:
        string_list (list): List of strings to process
        
    Returns:
        list: Processed list of strings
    """
    # Dictionary for number to word conversion
    number_to_word = {
        '0': 'zero',
        '1': 'one',
        '2': 'two',
        '3': 'three',
        '4': 'four',
        '5': 'five',
        '6': 'six',
        '7': 'seven',
        '8': 'eight',
        '9': 'nine'
    }

    def process_string(s):
        # Convert to lowercase
        s = s.lower()
        
        # Replace numbers with words
        for num, word in number_to_word.items():
            s = s.replace(num, word)
            
        # Replace special characters with 'x'
        processed = ''
        for char in s:
            if char.isalnum() or char.isspace():
                processed += char
            else:
                processed += 'x'
                
        return processed

    return [process_string(s) for s in string_list]


class ExportDataset:
    """
    Helper to make exporting datasets easier.
    """
    def __init__(self, export_path: str, initial_tuples_path: str|None = None):
        self.export_path = Path(export_path)
        if initial_tuples_path is not None:
            self.initial_tuples_path = Path(initial_tuples_path)

    def export_table(self, name: str, path: str, n_rows=None, na_values: List[str]|None = None):
        df = pd.read_csv(path, dtype=str, na_values=na_values)
        if n_rows is not None:
            df = df.iloc[:n_rows, :]

        # Replace missing values with 'empty'
        df.fillna('empty', inplace=True)

        # Remove commas from fields because this breaks domino
        df = df.apply(lambda x: x.str.replace(',', ''))

        # Ensure column names only contain letters
        df.columns = process_column_names(df.columns)

        df.to_csv(self.export_path/f"{name}.csv", index=False, sep=',')

    def transform_export_table(self, name: str, path_clean: str, path_dirty: str, n_rows=None, na_values: List[str]|None = None):
        """
        Helper to export erroneous datasets and inject the char "?" that 
        RENUVER expects to indicate errors.
        """
        df_clean = pd.read_csv(path_clean, dtype=str, na_values=na_values)
        df_dirty = pd.read_csv(path_dirty, dtype=str, na_values=na_values)
        
        if n_rows is not None:
            df_clean, df_dirty = df_clean.iloc[:n_rows, :], df_dirty.iloc[:n_rows, :]

        # Replace missing values with 'empty'
        df_clean.fillna('empty', inplace=True), df_dirty.fillna('empty', inplace=True)

        # Remove commas from fields because this breaks domino
        df_clean = df_clean.apply(lambda x: x.str.replace(',', ''))
        df_dirty = df_dirty.apply(lambda x: x.str.replace(',', ''))

        # Ensure column names only contain letters
        df_clean.columns = process_column_names(df_clean.columns)
        df_dirty.columns = process_column_names(df_dirty.columns)

        error_mask = df_clean != df_dirty
        
        # Set missing value char ? that renuver expects and save .csv
        df_dirty[error_mask] = '?'
        df_dirty.to_csv(self.export_path/f"{name}.csv", index=False, sep=',')

        # Also generate file for InitialTuples/ that renuver expects to store results
        rows, cols = np.where(error_mask)
        df_initial_tuples = pd.DataFrame({
            'row': rows + 1,  # + 1 since we want 1-based indexing, otherwise renuver throws errors
            'attribute': df_dirty.columns[cols],
            'value': df_dirty.values[rows, cols]
        })
        df_initial_tuples.to_csv(self.initial_tuples_path/f"{name}.csv", index=False, header=False, sep=';')

        df_clean_tuples = pd.DataFrame({
            'row': rows + 1,
            'attribute': df_clean.columns[cols],
            'value': df_clean.values[rows, cols]
        })
        df_clean_tuples.to_csv(self.export_path/f"{name}_clean_tuples.csv", index=False, header=False, sep=';')


def export_datasets():
    """
    Export datasets the datasets's clean and dirty version to export_path
    as .csv files.
    """
    export_domino = ExportDataset('domino/datasets/')
    export_renuver = ExportDataset('renuver/Dataset/', 'renuver/InitialTuples/')
    baran_dataset_ids = ["beers", "flights", "hospital", "rayyan", "tax", "food"]
    renuver_dataset_ids = ["bridges", "cars", "glass", "restaurant"]
    openml_dataset_ids = ["6", "137", "151", "184", "1481", "43572"]

    for dataset_name in renuver_dataset_ids:
        na_values = None
        path_clean = f"../../datasets/renuver/{dataset_name}/clean.csv"
        if dataset_name == 'bridges':
            na_values = ['?']
        export_domino.export_table(dataset_name, path_clean, None, na_values)
        export_renuver.export_table(dataset_name, path_clean, None, na_values)
        for version in [1,2,3]:
            for error_fraction in [1,3]:
                agg_name = f'{dataset_name}_{error_fraction}_{version}'
                path_dirty = f"../../datasets/renuver/{dataset_name}/{dataset_name}_{error_fraction}_{version}.csv"
                export_renuver.transform_export_table(agg_name, path_clean, path_dirty, None, na_values)
    print(f'Exported RENUVER datasets to {export_domino.export_path} and {export_renuver.export_path}')

    for dataset_name in baran_dataset_ids:
        path_clean = f"../../datasets/{dataset_name}/clean.csv"
        export_domino.export_table(dataset_name, path_clean)
        export_renuver.export_table(dataset_name, path_clean)

        agg_name = f'{dataset_name}_0_0'  # renuver needs two underscores in filename
        path_dirty = f"../../datasets/{dataset_name}/dirty.csv"
        export_renuver.transform_export_table(agg_name, path_clean, path_dirty)

    print(f'Exported Baran datasets to {export_domino.export_path} and {export_renuver.export_path}')

    for dataset_name in openml_dataset_ids:
        path_clean = f"../../datasets/openml/{dataset_name}/clean.csv"
        export_domino.export_table(dataset_name, path_clean, 1000)
        export_renuver.export_table(dataset_name, path_clean, 1000)
        
        error_class = 'imputer_simple_mcar'
        error_fraction = 5
        agg_name = f'{dataset_name}_{error_fraction}_0'
        path_dirty = f"../../datasets/openml/{dataset_name}/{error_class}_{error_fraction}.csv"
        export_renuver.transform_export_table(agg_name, path_clean, path_dirty, 1000)
    print(f'Exported OpenML datasets to {export_domino.export_path} and {export_renuver.export_path}')

if __name__ == '__main__':
    export_datasets()

