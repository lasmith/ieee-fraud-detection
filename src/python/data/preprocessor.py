import datetime
import numpy as np
import pandas as pd
from pandas import DataFrame

START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')

emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}
us_emails = ['gmail', 'net', 'edu']


def preprocess(data_dir:str):
    train_identity = pd.read_csv(data_dir + '/train_identity.csv')
    train_transaction = pd.read_csv(data_dir + '/train_transaction.csv')
    test_identity = pd.read_csv(data_dir + '/test_identity.csv')
    test_transaction = pd.read_csv(data_dir + '/test_transaction.csv')

    df_train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
    df_test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')

    engineer_features(df_train)
    engineer_features(df_test)

    # Some fixes to remove NaNs
    df_train = df_train.replace(np.nan, '', regex=True)
    df_test = df_test.replace(np.nan, '', regex=True)
    return df_train, df_test


def add_means(df:DataFrame, in_col, col_to_aggregate='TransactionAmt'):
    df[col_to_aggregate+'_to_mean_'+in_col] = df[col_to_aggregate] - df.groupby([in_col])[col_to_aggregate].transform('mean')
    df[col_to_aggregate+'_to_std_'+in_col] = df[col_to_aggregate+'_to_mean_'+in_col] / df.groupby([in_col])[col_to_aggregate].transform('std')

def engineer_features(df:DataFrame, engineer_identity_features=True):
    """
    Engineer some new features
    :param df: The dataframe to amend
    :param engineer_identity_features: If the identity data frame has been included
    :return:
    """
    # First some date fields relative to an estimated start date
    df['Date'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
    df['DT_M'] = (df['Date'].dt.year-2017)*12 + df['Date'].dt.month
    df['DT_W'] = df['Date'].dt.dayofweek
    df['DT_H'] = df['Date'].dt.hour
    df['DT_D'] = df['Date'].dt.day

    # Log transform the TX amount so its normally distributed. Oddly keeping both cols gives better performance..
    df['TransactionAmt'] = np.log(df['TransactionAmt'])

    # Bin the emails
    for c in ['P_emaildomain', 'R_emaildomain']:
        df[c + '_bin'] = df[c].map(emails)
        df[c + '_suffix'] = df[c].map(lambda x: str(x).split('.')[-1])
        df[c + '_suffix'] = df[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

    add_means(df, 'card1')
    add_means(df, 'card2')
    add_means(df, 'card3')
    add_means(df, 'card4')

    add_means(df, 'card1', col_to_aggregate='id_02')
    add_means(df, 'card4', col_to_aggregate='id_02')

    add_means(df, 'card1', col_to_aggregate='D15')
    add_means(df, 'card4', col_to_aggregate='D15')

    add_means(df, 'addr1', col_to_aggregate='D15')
    add_means(df, 'addr2', col_to_aggregate='D15')

    # Drop some columns. These came from EDA analysis
    useful_features = ['TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1',
                       'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13',
                       'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M2', 'M3',
                       'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V17',
                       'V19', 'V20', 'V29', 'V30', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V40', 'V44', 'V45', 'V46', 'V47', 'V48',
                       'V49', 'V51', 'V52', 'V53', 'V54', 'V56', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V69', 'V70', 'V71',
                       'V72', 'V73', 'V74', 'V75', 'V76', 'V78', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V87', 'V90', 'V91', 'V92',
                       'V93', 'V94', 'V95', 'V96', 'V97', 'V99', 'V100', 'V126', 'V127', 'V128', 'V130', 'V131', 'V138', 'V139', 'V140',
                       'V143', 'V145', 'V146', 'V147', 'V149', 'V150', 'V151', 'V152', 'V154', 'V156', 'V158', 'V159', 'V160', 'V161',
                       'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V169', 'V170', 'V171', 'V172', 'V173', 'V175', 'V176', 'V177',
                       'V178', 'V180', 'V182', 'V184', 'V187', 'V188', 'V189', 'V195', 'V197', 'V200', 'V201', 'V202', 'V203', 'V204',
                       'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V212', 'V213', 'V214', 'V215', 'V216', 'V217', 'V219', 'V220',
                       'V221', 'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229', 'V231', 'V233', 'V234', 'V238', 'V239',
                       'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V249', 'V251', 'V253', 'V256', 'V257', 'V258', 'V259', 'V261',
                       'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276',
                       'V277', 'V278', 'V279', 'V280', 'V282', 'V283', 'V285', 'V287', 'V288', 'V289', 'V291', 'V292', 'V294', 'V303',
                       'V304', 'V306', 'V307', 'V308', 'V310', 'V312', 'V313', 'V314', 'V315', 'V317', 'V322', 'V323', 'V324', 'V326',
                       'V329', 'V331', 'V332', 'V333', 'V335', 'V336', 'V338', 'id_01', 'id_02', 'id_03', 'id_05', 'id_06', 'id_09',
                       'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_17', 'id_19', 'id_20', 'id_30', 'id_31', 'id_32', 'id_33',
                       'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']
    cols_to_drop = [col for col in df.columns if col not in useful_features]
    cols_to_drop.remove('isFraud')
    cols_to_drop.remove('TransactionID')
    df.drop(cols_to_drop, axis=1)
    #df.drop('Date',axis=1, inplace=True)

    if engineer_identity_features:
        for x in range(1, 12):
            df['id_'+str(x).zfill(2)] = np.log(df['id_'+str(x).zfill(2)])
        add_means(df, 'card1', col_to_aggregate='id_02')
        add_means(df, 'card4', col_to_aggregate='id_02')

    return df

def get_categorical_cols(df:DataFrame):
    cols_to_find = [
        # TX features
        'ProductCD',
        'card1',
        'card2',
        'card3',
        'card4',
        'card5',
        'card6',
        'addr1',
        'addr2',
        'P_emaildomain',
        'R_emaildomain',
        'M1',
        'M2',
        'M3',
        'M4',
        'M5',
        'M6',
        'M7',
        'M8',
        'M9',
        # Identity features
        'DeviceType',
        'DeviceInfo',
        'id_12',  'id_13',  'id_14', 'id_15',  'id_16',  'id_17',  'id_18',  'id_19',
        'id_20',  'id_21',  'id_22',  'id_23',  'id_24',  'id_25',  'id_26',  'id_27',  'id_28',  'id_29',
        'id_30',  'id_31',  'id_32',  'id_33',  'id_34',  'id_35',  'id_36',  'id_37',  'id_38',

        # Engineered features
        'P_emaildomain_bin',
        'P_emaildomain_suffix',
        'R_emaildomain_bin',
        'R_emaildomain_suffix'
    ]

    return [df.columns.get_loc(col) for col in cols_to_find]
