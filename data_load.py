import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import chi2_contingency
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import stats
from sklearn.decomposition import PCA
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


FILE_ROOT = './'

def load_pump_data(data_version='', random_state=42, val_size=0.2, file_root=FILE_ROOT, verbose=False, resample=False):
    """
    Loads, preprocesses, and splits the pump data into training and validation sets.

    This function handles loading data, identifying feature types, creating or loading
    a scikit-learn preprocessor pipeline (imputation and scaling/encoding),
    splitting the data, and optionally applying resampling techniques to the training set.

    Args:
        data_version (str): The version of the data to load. If empty, the latest version is used.
        random_state (int): Seed for reproducibility in train/test split and resampling.
        val_size (float): The proportion of the dataset to allocate to the validation set.
        file_root (str): The root directory where data files are located.
        verbose (bool): If True, prints detailed information about the process.
        resample (bool): If True, applies resampling (ADASYN) to the training data.

    Returns:
        tuple: A tuple containing the transformed data splits:
            (X_train_transformed, y_train, X_val_transformed, y_val, X_test_transformed)
    """
    if data_version == '':
        data_version = sorted(os.listdir(file_root + 'train/X/'))[-1].split('_')[0]

    df_X, df_y, df_TEST_X = load_unprocessed_data(data_version, file_root)

    y = df_y['status_group']
    if verbose:
        print(f"\nTarget variable 'status_group' extracted. Value counts:\n{y.value_counts(normalize=True)}")

    if 'date_recorded' in df_X.columns:
        df_X = df_X.drop('date_recorded', axis=1)
        if verbose:
            print("Dropped 'date_recorded' column.")

    if 'date_recorded' in df_TEST_X.columns:
        df_TEST_X = df_TEST_X.drop('date_recorded', axis=1)
        if verbose:
            print("Dropped 'date_recorded' column from test set.")

    numerical_features = df_X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df_X.select_dtypes(include=['object', 'bool']).columns.tolist()
    if verbose:
        print(f"\nIdentified {len(numerical_features)} numerical features.")
        print(f"Identified {len(categorical_features)} categorical features.")

    try:
        preprocessor = joblib.load(file_root + 'preprocessors/' + data_version + '_features_preprocessor.joblib')
        if verbose:
            print("\nPreprocessor loaded.")
    except FileNotFoundError:
        if verbose:
            print("Preprocessor not found. Creating a new one.")
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        if verbose:
            print("\nPreprocessor defined.")
        preprocessor.fit(df_X)
        if verbose:
            print("Preprocessor fitted on training data.")
        joblib.dump(preprocessor, file_root + 'preprocessors/' + data_version + '_features_preprocessor.joblib')
        if verbose:
            print(f"Preprocessor saved as {data_version}_features_preprocessor.joblib.")

    if verbose:
        print(f"\nSplitting data into train/validation sets (Test size: {val_size})...")
    X_train, X_val, y_train, y_val = train_test_split(
        df_X, y,
        test_size=val_size,
        random_state=random_state,
        stratify=y
    )
    if verbose:
        print(f"Raw X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"Raw X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    if verbose:
        print("Transforming training data...")
    X_train_transformed = preprocessor.transform(X_train)
    if verbose:
        print("Transforming validation data...")
    X_val_transformed = preprocessor.transform(X_val)
    if verbose:
        print("Transforming test data...")
    X_test_transformed = preprocessor.transform(df_TEST_X)
    if verbose:
        print(f"Transformed training data shape: {X_train_transformed.shape}")
        print(f"Transformed validation data shape: {X_val_transformed.shape}")
    print(f"Transformed test data shape: {X_test_transformed.shape}")

    if resample:
        if verbose:
            print("Applying resampling with ADASYN.")
        resampling_pipeline = ImbPipeline([
            ('adasyn', ADASYN(sampling_strategy={
                'functional needs repair': 10000,
            }, random_state=random_state)),
        ])
        X_train_transformed, y_train = resampling_pipeline.fit_resample(X_train_transformed, y_train)
        if verbose:
            print(f"After resampling: X_train shape: {X_train_transformed.shape}, y_train distribution:\n{pd.Series(y_train).value_counts()}")

    y_train_encoded = y_train.copy()
    y_val_encoded = y_val.copy()

    return X_train_transformed, y_train_encoded, X_val_transformed, y_val_encoded, X_test_transformed

def load_unprocessed_data(data_version='', file_root=FILE_ROOT):
    """
    Loads raw, unprocessed pump data from CSV files.

    Args:
        data_version (str): The version of the data to load. If empty, the latest version is used.
        file_root (str): The root directory containing the data folders.

    Returns:
        tuple: A tuple of three pandas DataFrames: (df_X, df_y, df_TEST_X).
    """
    if data_version == '':
        data_version = sorted(os.listdir(file_root + 'train/X/'))[-1].split('_')[0]

    df_X = pd.read_csv(file_root + 'train/X/' + data_version +'_pump_it_up_TRAIN_X.csv')
    df_y = pd.read_csv(file_root + 'train/y/pump_it_up_TRAIN_y.csv')
    df_TEST_X = pd.read_csv(file_root + 'test/X/' + data_version +'_pump_it_up_TEST_X.csv')

    return df_X, df_y, df_TEST_X

def encode_target(y_raw, file_root=FILE_ROOT, verbose=False):
    """
    Encodes target variable labels from strings to integers.

    It loads a pre-fitted LabelEncoder if available; otherwise, it fits a new one
    and saves it to ensure consistent encoding across runs.

    Args:
        y_raw: The raw data of string labels.
        file_root (str): The root directory for saving/loading the preprocessor.
        verbose (bool): If True, prints status messages.
    Returns:
        np.ndarray: The encoded integer labels.
    """
    encoder_path = os.path.join(file_root, 'preprocessors', 'label_encoder.joblib')
    os.makedirs(os.path.dirname(encoder_path), exist_ok=True)

    try:
        label_encoder = joblib.load(encoder_path)
        if verbose:
            print("\nLabelEncoder loaded from file.")
    except FileNotFoundError:
        if verbose:
            print("\nLabelEncoder not found. Creating, fitting, and saving a new one.")
        label_encoder = LabelEncoder()
        label_encoder.fit(y_raw)
        joblib.dump(label_encoder, encoder_path)
        if verbose:
            print(f"LabelEncoder saved to {encoder_path}")

    y_encoded = label_encoder.transform(y_raw)

    if verbose:
        mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        print(f"Label mapping: {mapping}")

    return y_encoded

def decode_target(y_encoded):
    """
    Decodes target variable labels from integers back to their original string representation.

    Args:
        y_encoded (np.ndarray): The array of encoded integer labels.

    Returns:
        np.ndarray: The array of decoded string labels.
    """
    try:
        label_encoder = joblib.load(os.path.join(FILE_ROOT, 'preprocessors', 'label_encoder.joblib'))
    except FileNotFoundError:
        raise ValueError("LabelEncoder not found. Please encode the target variable first using encode_target().")

    return label_encoder.inverse_transform(y_encoded)

def basic_categorical_eda(df):
    """Performs a basic EDA on all categorical columns in a DataFrame."""
    cat_cols = df.select_dtypes(include=['object', 'bool']).columns
    for col in cat_cols:
        print(f"--- {col} ---")
        print(f"Type: {df[col].dtype}")
        print(f"Nulls: {df[col].isnull().sum()}")
        print(f"Unique: {df[col].nunique()}")
        print(df[col].value_counts(dropna=False).head(10))
        print()

def basic_numerical_eda(df):
    """Performs a basic EDA on all numerical columns in a DataFrame."""
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        print(f"--- {col} ---")
        print(f"Nulls: {df[col].isnull().sum()}")
        print(f"Unique: {df[col].nunique()}")
        print(df[col].describe())
        print()
        print(f"Unique values: {df[col].unique()}")

def categorical_display_vs_numeric(df_cat, df_num, cat_col, num_col):
    """
    Displays a boxplot of a numeric variable across categories of a categorical variable.

    Args:
        df_cat (pd.DataFrame): DataFrame containing the categorical column.
        df_num (pd.DataFrame): DataFrame containing the numeric column.
        cat_col (str): The name of the categorical column.
        num_col (str): The name of the numeric column.
    """
    if cat_col not in df_cat.columns:
        raise ValueError(f"Categorical column '{cat_col}' does not exist in the DataFrame.")
    if num_col not in df_num.columns:
        raise ValueError(f"Numeric column '{num_col}' does not exist in the DataFrame.")

    plt.figure(figsize=(10, 6))
    sns.boxplot(x=cat_col, y=num_col, data=df_num)
    plt.title(f'Distribution of {num_col} by {cat_col}')
    plt.xticks(rotation=45)
    plt.show()

def categorical_display_vs_coordinates(df_cat, df_coord, cat_col, lat_col='latitude', lon_col='longitude'):
    """
    Displays a scatter plot of a categorical variable on a map of coordinates.

    Args:
        df_cat (pd.DataFrame): DataFrame containing the categorical column.
        df_coord (pd.DataFrame): DataFrame containing the coordinate columns.
        cat_col (str): The name of the categorical column.
        lat_col (str): The name of the latitude column.
        lon_col (str): The name of the longitude column.
    """
    if cat_col not in df_cat.columns:
        raise ValueError(f"Categorical column '{cat_col}' does not exist in the DataFrame.")
    if lat_col not in df_coord.columns or lon_col not in df_coord.columns:
        raise ValueError(f"Latitude or Longitude column does not exist in the DataFrame.")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=lon_col, y=lat_col, hue=cat_col, data=df_coord, alpha=0.7)
    plt.title(f'{cat_col} Distribution by Coordinates')
    plt.show()


def categorical_display_vs_coordinates_rasterized(df_cat, df_coord, cat_col, lat_col='latitude', lon_col='longitude'):
    """
    Displays the relationship using a rasterized scatter plot for performance.

    Args:
        df_cat (pd.DataFrame): DataFrame containing the categorical column.
        df_coord (pd.DataFrame): DataFrame containing the coordinate columns.
        cat_col (str): The name of the categorical column.
        lat_col (str): The name of the latitude column.
        lon_col (str): The name of the longitude column.
    """
    if cat_col not in df_cat.columns:
        raise ValueError(f"Categorical column '{cat_col}' does not exist in the DataFrame.")
    if lat_col not in df_coord.columns or lon_col not in df_coord.columns:
        raise ValueError(f"Latitude '{lat_col}' or Longitude '{lon_col}' column does not exist.")

    point_size = 5
    alpha_val = 0.3

    plt.figure(figsize=(12, 8))
    # Avoid coordinates (0,0) if they are not valid
    zero_mask = (df_cat[lat_col] != 0) & (df_cat[lon_col] != 0)
    df_coord = df_coord[zero_mask]
    df_cat = df_cat[zero_mask]

    # Avoid NaN values
    na_mask = df_coord[lat_col].isna() | df_coord[lon_col].isna()
    df_coord = df_coord[~na_mask]
    df_cat = df_cat[~na_mask]

    if df_cat.empty or df_coord.empty:
        print(f"No valid data available for {cat_col} vs coordinates.")
        return

    sns.scatterplot(
        x=lon_col,
        y=lat_col,
        hue=cat_col,
        data=df_coord,
        s=point_size,
        alpha=alpha_val,
        linewidth=0,
        rasterized=True
    )
    plt.title(f'{cat_col} Distribution by Coordinates (Rasterized)')
    plt.xlabel(lon_col)
    plt.ylabel(lat_col)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def categorical_display_vs_categorical(df_col_1, df_col_2, cat_col_1, cat_col_2):
    """
    Displays the relationship between two categorical variables using a count plot.

    Args:
        df_col_1 (pd.DataFrame): DataFrame containing the first categorical column.
        df_col_2 (pd.DataFrame): DataFrame containing the second categorical column.
        cat_col_1 (str): Name of the first categorical column.
        cat_col_2 (str): Name of the second categorical column.
    """
    if cat_col_1 not in df_col_1.columns:
        raise ValueError(f"Categorical column '{cat_col_1}' does not exist in the DataFrame.")
    if cat_col_2 not in df_col_2.columns:
        raise ValueError(f"Categorical column '{cat_col_2}' does not exist in the DataFrame.")

    df_data = df_col_1[[cat_col_1]].copy()
    df_data[cat_col_2] = df_col_2[cat_col_2].copy()

    plt.figure(figsize=(10, 6))
    sns.countplot(x=cat_col_1, hue=cat_col_2, data=df_data)
    plt.title(f'Distribution of {cat_col_1} by {cat_col_2}')
    plt.xticks(rotation=45)
    plt.show()

def row_null_count(df):
    """
    Counts the number of rows with at least one null value in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to check for null values.
    Returns:
        int: The number of rows with at least one null value.
    """
    return df.isnull().any(axis=1).sum()

def check_normality(data):
    """
    Checks the normality of a numeric array.

    Uses Shapiro-Wilk for N <= 5000 and D'Agostino-Pearson for N > 5000.
    Returns p-value or None if the data is too small.
    """
    n = len(data)
    if n < 3:
        return None
    elif n <= 5000:
        return stats.shapiro(data).pvalue
    else:
        return stats.normaltest(data).pvalue

def cramers_v(df, col1, col2):
    """
    Calculates the correlation between two categorical variables using Cramér's V.

    Args:
        df (pd.DataFrame): The pandas DataFrame.
        col1 (str): Name of the first categorical variable.
        col2 (str): Name of the second categorical variable.

    Returns:
        float: Cramér's V value between 0 and 1.
    """
    confusion_matrix = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = chi2_contingency(confusion_matrix)

    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    # Bias correction recommended by Bergsma (2013)
    phi2corr = max(0, phi2 - ((k-1)*(r-1)) / (n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)

    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

def apply_PCA(df, n_components=None):
    """
    Applies PCA to the DataFrame and returns the transformed data.

    Args:
        df (pd.DataFrame): The pandas DataFrame.
        n_components (int): Number of principal components to keep.

    Returns:
        pd.DataFrame: Transformed DataFrame with PCA applied.
    """
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(df)
    return pd.DataFrame(transformed_data, columns=[f"PC{i+1}" for i in range(n_components)])


if __name__ == "__main__":
    # Testing
    df_X, df_y, df_TEST_X = load_unprocessed_data()
    print("Loaded unprocessed data:")
    print("df_X:")
    print(df_X.head())
    print("df_y:")
    print(df_y.head())
    print("df_TEST_X:")
    print(df_TEST_X.head())