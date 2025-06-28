import os
import data_load as dl
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import joblib

def delete_columns(df, columns):
    """
    Deletes specified columns from the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame from which to delete columns.
        columns (list): A list of column names to delete.
    """
    df.drop(columns=columns, inplace=True, errors='ignore')
    return df

if __name__ == "__main__":

    data_version = sorted(os.listdir('./train/X/'))[-1].split('_')[0]
    df_X = pd.read_csv('./train/X/' + data_version +'_pump_it_up_TRAIN_X.csv')
    df_y = pd.read_csv('./train/y/pump_it_up_TRAIN_y.csv')
    df_TEST_X = pd.read_csv('./test/X/' + data_version +'_pump_it_up_TEST_X.csv')
    version = int(data_version)

    print(f"df_X has {df_X.shape[0]} rows and {df_X.shape[1]} columns")
    print(f"df_TEST_X has {df_TEST_X.shape[0]} rows and {df_TEST_X.shape[1]} columns")
    print(f"df_y has {df_y.shape[0]} rows and {df_y.shape[1]} columns")

    if version < 1:
        redundant_columns = ['recorded_by', 'quantity_group', 'quality_group', 'source_type', 'source_class', 'lga',
                             'waterpoint_type_group', 'extraction_type', 'extraction_type_class', 'management_group']
        correlated_columns = ['ward', 'region_code']

        df_X = delete_columns(df_X, redundant_columns)
        df_TEST_X = delete_columns(df_TEST_X, redundant_columns)
        df_X = delete_columns(df_X, correlated_columns)
        df_TEST_X = delete_columns(df_TEST_X, correlated_columns)

        # df_X.to_csv('train/X/1_pump_it_up_TRAIN_X.csv', index=False)
        # df_TEST_X.to_csv('test/X/1_pump_it_up_TEST_X.csv', index=False)

    if version < 2:
        columns_to_delete = ['id']
        df_X = delete_columns(df_X, columns_to_delete)
        df_TEST_X = delete_columns(df_TEST_X, columns_to_delete)

        # df_X.to_csv('train/X/2_pump_it_up_TRAIN_X.csv', index=False)
        # df_TEST_X.to_csv('test/X/2_pump_it_up_TEST_X.csv', index=False)

    if version < 3:
        categorical_columns = df_X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df_X[col] = df_X[col].fillna('unknown')
            df_TEST_X[col] = df_TEST_X[col].fillna('unknown')

        # df_X.to_csv('train/X/3_pump_it_up_TRAIN_X.csv', index=False)
        # df_TEST_X.to_csv('test/X/3_pump_it_up_TEST_X.csv', index=False)

    if version < 4:
        categorical_columns = df_X.select_dtypes(include=['object']).columns
        columns_to_delete = []
        for col in categorical_columns:
            unknown_count = df_X[col].value_counts().get('unknown', 0)
            if unknown_count / len(df_X) > 0.5:
                columns_to_delete.append(col)

        print("Columns to delete due to high unknown values: ", columns_to_delete)
        df_X.drop(columns=columns_to_delete, inplace=True, errors='ignore')
        df_TEST_X.drop(columns=columns_to_delete, inplace=True, errors='ignore')

        # df_X.to_csv('train/X/4_pump_it_up_TRAIN_X.csv', index=False)
        # df_TEST_X.to_csv('test/X/4_pump_it_up_TEST_X.csv', index=False)

    if version < 5:
        df_X['funder'] = df_X['funder'].replace('Not known', 'unknown')
        df_TEST_X['funder'] = df_TEST_X['funder'].replace('Not known', 'unknown')

        # df_X.to_csv('train/X/5_pump_it_up_TRAIN_X.csv', index=False)
        # df_TEST_X.to_csv('test/X/5_pump_it_up_TEST_X.csv', index=False)

    print(f"df_X has {df_X.shape[0]} rows and {df_X.shape[1]} columns")
    print(f"df_TEST_X has {df_TEST_X.shape[0]} rows and {df_TEST_X.shape[1]} columns")
    print(f"df_y has {df_y.shape[0]} rows and {df_y.shape[1]} columns")

    if version < 6:
        duplicates_train = df_X.duplicated(keep='first')

        df_X = df_X[~duplicates_train]
        df_y = df_y[~duplicates_train]
        df_y.reset_index(drop=True, inplace=True)
        df_X.reset_index(drop=True, inplace=True)

        df_y.to_csv('train/y/pump_it_up_TRAIN_y_new.csv', index=False)
        df_X.to_csv('train/X/6_pump_it_up_TRAIN_X.csv', index=False)
        df_TEST_X.to_csv('test/X/6_pump_it_up_TEST_X.csv', index=False)

    print(f"df_X has {df_X.shape[0]} rows and {df_X.shape[1]} columns")
    print(f"df_TEST_X has {df_TEST_X.shape[0]} rows and {df_TEST_X.shape[1]} columns")
    print(f"df_y has {df_y.shape[0]} rows and {df_y.shape[1]} columns")

    if version < 7:
        # Convert construction_year to age in years (float)
        newest_date_X = pd.to_datetime(df_X['construction_year'].max())
        newest_date_TEST_X = pd.to_datetime(df_TEST_X['construction_year'].max())
        newest_date = max(newest_date_X, newest_date_TEST_X)
        df_X['construction_year'] = pd.to_datetime(df_X['construction_year'])
        df_TEST_X['construction_year'] = pd.to_datetime(df_TEST_X['construction_year'])
        df_X['construction_year'] = (newest_date - df_X['construction_year']).dt.days / 365.25
        df_TEST_X['construction_year'] = (newest_date - df_TEST_X['construction_year']).dt.days / 365.25

        repeat_threshold = 100
        other_label = "unknown"

        # Group rare categorical values into 'unknown'
        for col in df_X.select_dtypes(include=['object']).columns:
            counts = df_X[col].value_counts()
            mask = counts < repeat_threshold
            df_X[col] = df_X[col].where(~df_X[col].isin(counts[mask].index), other_label)
            df_TEST_X[col] = df_TEST_X[col].where(~df_TEST_X[col].isin(counts[mask].index), other_label)

        df_X.to_csv('train/X/7_pump_it_up_TRAIN_X.csv', index=False)
        df_TEST_X.to_csv('test/X/7_pump_it_up_TEST_X.csv', index=False)

    if version < 8:
        # Apply PCA to reduce dimensionality, retaining 95% of variance
        pca = PCA(n_components=0.95)
        preprocessor = joblib.load('./preprocessors/7_features_preprocessor.joblib')
        X_preprocessed = preprocessor.transform(df_X)
        X_pca = pca.fit_transform(X_preprocessed)
        TEST_X_preprocessed = preprocessor.transform(df_TEST_X)
        TEST_X_pca = pca.transform(TEST_X_preprocessed)

        df_X_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
        df_TEST_X_pca = pd.DataFrame(TEST_X_pca, columns=[f'PC{i+1}' for i in range(TEST_X_pca.shape[1])])
        df_X_pca.to_csv('train/X/8_pump_it_up_TRAIN_X.csv', index=False)
        df_TEST_X_pca.to_csv('test/X/8_pump_it_up_TEST_X.csv', index=False)