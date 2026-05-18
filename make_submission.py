import pandas as pd
import numpy as np
from pathlib import Path

from src.data.categorical_encoding import CategoricalEncoder
from src.features.temporal_features import TemporalFeatureEngineer
from src.models.lightgbm_model import LightGBMFraudModel


def load_data(transaction_path: Path, identity_path: Path) -> pd.DataFrame:
    df = pd.read_csv(transaction_path)
    if identity_path.exists():
        identity_df = pd.read_csv(identity_path)
        df = df.merge(identity_df, on='TransactionID', how='left')
    return df


def prepare_features(df: pd.DataFrame, drop_cols: list, label_col: str = 'isFraud'):
    feature_cols = [
        col
        for col in df.columns
        if col not in drop_cols + [label_col]
        and df[col].dtype != object
    ]
    X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y = df[label_col].to_numpy(dtype=np.int8, copy=False)
    return X, y, feature_cols


def main():
    root = Path(__file__).resolve().parent
    raw_dir = root / 'data' / 'raw'

    train_transaction = raw_dir / 'train_transaction.csv'
    train_identity = raw_dir / 'train_identity.csv'
    test_transaction = raw_dir / 'test_transaction.csv'
    test_identity = raw_dir / 'test_identity.csv'
    sample_submission = raw_dir / 'sample_submission.csv'
    output_file = root / 'submission.csv'

    print('Loading raw data...')
    train_df = load_data(train_transaction, train_identity)
    test_df = load_data(test_transaction, test_identity)

    print('Preserving test order from sample_submission.csv...')
    sample_order = pd.read_csv(sample_submission)['TransactionID']

    # Keep original test order for final submission
    test_df = test_df.copy()
    test_df['__sample_order__'] = np.arange(len(test_df))

    # Merge train/test for temporal feature engineering
    full_df = pd.concat([train_df, test_df], axis=0)
    full_df = full_df.sort_values('TransactionDT')

    print('Building temporal features on full timeline...')
    engineer = TemporalFeatureEngineer()
    full_df = engineer.engineer_all_features(full_df)

    print('Encoding categorical features using train-only encodings...')
    encoder = CategoricalEncoder()
    cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    reserved_cols = {'DeviceInfo', 'addr1', 'card1'}
    cat_cols = [col for col in cat_cols if col not in reserved_cols]

    train_rows = full_df.loc[train_df.index].copy()
    if cat_cols:
        encoder.fit(train_rows, cat_cols)
        full_df = encoder.transform(full_df)

    train_rows = full_df.loc[train_df.index].copy()
    test_rows = full_df.loc[test_df.index].copy()

    # Ensure feature alignment and split
    drop_cols = ['TransactionID', 'TransactionDT']
    X_all, y_all, feature_cols = prepare_features(train_rows, drop_cols)

    # Use the last 10% of train as validation for LightGBM early stopping
    train_size = int(len(train_rows) * 0.9)
    if train_size < 1:
        raise ValueError('Training data is too small to create an internal validation split.')

    train_part = train_rows.iloc[:train_size].copy()
    valid_part = train_rows.iloc[train_size:].copy()

    X_train, y_train, _ = prepare_features(train_part, drop_cols)
    X_val, y_val, _ = prepare_features(valid_part, drop_cols)

    print('Training LightGBM model...')
    model = LightGBMFraudModel()
    model.train(X_train, y_train, X_val, y_val, feature_cols)

    print('Predicting test probabilities...')
    X_test = test_rows[feature_cols].reindex(columns=feature_cols, fill_value=0).to_numpy(dtype=np.float32, copy=False)
    test_probs = model.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({
        'TransactionID': test_rows['TransactionID'].astype(np.int64),
        'isFraud': test_probs.astype(np.float32)
    })

    submission = submission.set_index('TransactionID').reindex(sample_order).reset_index()

    print(f'Writing submission to {output_file}')
    submission.to_csv(output_file, index=False)
    print('Done.')


if __name__ == '__main__':
    main()
