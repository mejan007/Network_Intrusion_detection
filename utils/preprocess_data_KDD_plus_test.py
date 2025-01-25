import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os


attack_groups = {
'DoS': ['neptune', 'smurf', 'teardrop', 'apache2', 'land', 'back', 'mailbomb', 'pod', 'httptunnel'],
'Probe': ['nmap', 'ipsweep', 'portsweep', 'saint', 'satan', 'snmpgetattack', 'snmpguess'],
'R2L': ['guess_passwd', 'warezmaster', 'snmpguess', 'saint', 'sendmail', 'ftp_write', 'imap', 'xterm', 'phf'],
}
    

def convert_data_to_csv(input_file_path, output_file_path):
    # Read the text file into a DataFrame without headers
    df = pd.read_csv(input_file_path, header=None)

    # Define the column names
    columns = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_root", 
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
        "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", 
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "level"
    ]

    # Assign the column names to the DataFrame
    df.columns = columns

    # List of columns to keep
    keep_columns = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "count", "srv_count", "serror_rate",
        "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate", "attack"
    ]

    # Drop columns not in the keep_columns list
    df = df[keep_columns]

    # Convert 'attack' column to binary classification
    # df["attack"] = df["attack"].apply(lambda x: "normal" if x == "normal" else "anomaly")
    df['attack'] = df['attack'].apply(group_attack)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Save the processed data to a CSV file
    df.to_csv(output_file_path, index=False)

    print(f"Data has been successfully saved to {output_file_path}.")

    

def split_data(
    file_path, train_file_path, test_file_path, test_size=0.2, random_state=42
):
    # Read the text file into a DataFrame without headers
    df = pd.read_csv(file_path, header=None)

    # Define the column names
    columns = [
        "duration",
        "protocol_type",
        "service",
        "flag",
        "src_bytes",
        "dst_bytes",
        "land",
        "wrong_fragment",
        "urgent",
        "hot",
        "num_failed_logins",
        "logged_in",
        "num_compromised",
        "root_shell",
        "su_attempted",
        "num_root",
        "num_file_creations",
        "num_shells",
        "num_access_files",
        "num_outbound_cmds",
        "is_host_login",
        "is_guest_login",
        "count",
        "srv_count",
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
        "attack",
        "level",
    ]

    # Assign the column names to the DataFrame
    df.columns = columns

    # List of columns to keep
    keep_columns = [
        "duration",
        "protocol_type",
        "service",
        "flag",
        "src_bytes",
        "dst_bytes",
        "land",
        "wrong_fragment",
        "urgent",
        "count",
        "srv_count",
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
        "attack",
    ]

    # Drop columns not in the keep_columns list
    df = df[keep_columns]

    # Convert 'attack' column to binary classification
    # df["attack"] = df["attack"].apply(
    #     lambda x: "normal" if x == "normal" else "anomaly"
    # )
    df['attack'] = df['attack'].apply(group_attack)
    df.drop_duplicates(inplace=True)

    # Separate features and target
    X = df.drop("attack", axis=1)
    y = df["attack"]

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Combine features with target
    train_data = pd.concat(
        [X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1
    )
    test_data = pd.concat(
        [X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1
    )

    # Save the processed data to files
    train_data.to_csv(train_file_path, index=False)
    test_data.to_csv(test_file_path, index=False)

    print(
        f"Train and test data have been saved to {train_file_path} and {test_file_path}, respectively."
    )


def create_pipeline_and_save_models(
    train_file_path,
    save_folder="preprocessing_pipeline",
    random_state=42,
    n_components=0.95,
):
    # Load the training data
    df = pd.read_csv(train_file_path)

    # Define the categorical and numerical features
    categorical_features = ["protocol_type", "service", "flag"]
    numerical_features = df.columns.difference(categorical_features + ["attack"])

    # Define the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False),
                categorical_features,
            ),
            ("num", StandardScaler(), numerical_features),
        ]
    )

    # Define the pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("pca", PCA(n_components=n_components, random_state=random_state)),
        ]
    )

    # Separate features and target
    X = df.drop(columns=["attack"])
    y = df["attack"]

    # Encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Fit the pipeline on the training data
    X_transformed = pipeline.fit_transform(X)

    # Save the models
    joblib.dump(pipeline, os.path.join(save_folder, "pipeline.pkl"))
    joblib.dump(label_encoder, os.path.join(save_folder, "attack_label_encoder.pkl"))

    print(
        "Pipeline and models have been saved as pipeline.pkl and attack_label_encoder.pkl."
    )


def preprocess_data(
    train_file_path,
    val_file_path,
    test_file_path,
    preprocessing_models_folder="preprocessing_pipeline",
):
    # Load the training and testing datasets
    df_train = pd.read_csv(train_file_path)
    df_val = pd.read_csv(val_file_path)
    df_test = pd.read_csv(test_file_path)

    # Load the saved pipeline and label encoder
    pipeline = joblib.load(os.path.join(preprocessing_models_folder, "pipeline.pkl"))
    label_encoder = joblib.load(
        os.path.join(preprocessing_models_folder, "attack_label_encoder.pkl")
    )

    # Separate features and target in training data
    X_train = df_train.drop(columns=["attack"])
    y_train = df_train["attack"]

    # Apply the pipeline to transform the training data
    X_train_transformed = pipeline.transform(X_train)

    # Encode the target variable in training data
    y_train_encoded = label_encoder.transform(y_train)

    # Apply same pipeline to validation data
    X_val = df_val.drop(columns = ['attack'])
    y_val = df_val["attack"]

    X_val_transformed = pipeline.transform(X_val)
    y_val_encoded = label_encoder.transform(y_val)

    # Apply the same pipeline to the testing data
    X_test = df_test.drop(columns=["attack"])
    y_test = df_test["attack"]

    # Transform the testing data
    X_test_transformed = pipeline.transform(X_test)

    # Encode the target variable in testing data
    y_test_encoded = label_encoder.transform(y_test)

    return X_train_transformed, X_val_transformed, X_test_transformed, y_train_encoded, y_val_encoded, y_test_encoded

def group_attack(attack):

    if attack == 'normal':
        return 'normal'
    for group, attacks in attack_groups.items():
        if attack in attacks:
            return group
    return 'Other' 