import pandas as pd
from BankClassify import BankClassify


def test_mintReader_returns_date_description_ammount():
    bc = BankClassify()
    df = bc._read_mint_csv('transactions.csv')

    columns = df.columns.values.tolist()
    assert 'date' in columns
    assert 'desc' in columns
    assert 'amount' in columns


def test_onlyTreeColumns():
    bc = BankClassify()
    df = bc._read_mint_csv('transactions.csv')

    assert len(df.columns.values.tolist()) == 3


def test_debitIsNegative_creditIsPositive():
    df = pd.read_csv('transactions.csv', skiprows=0)

    """Rename columns """
    # df.columns = ['date', 'desc', 'amount']
    df.rename(
        columns={
            "Date": 'date',
            "Original Description": 'desc',
            "Amount": 'amount',
            "Transaction Type": 'type'
        },
        inplace=True
    )

    bc = BankClassify()
    df_dut = bc._read_mint_csv('transactions.csv')

    baseline = df['type'] == 'debit'

    assert (df_dut.loc[baseline, 'amount'] < 0).all()
    assert (df_dut.loc[~baseline, 'amount'] >= 0).all()


def test_reload_trainingset():
    bc = BankClassify(data='')
    assert len(bc.classifier.train_set) == 0

    dataset = pd.read_csv('test/test_training_set.csv')
    bc.retrain_classifier(dataset)
    assert len(bc.classifier.train_set) == 13
