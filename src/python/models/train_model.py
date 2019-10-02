# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import numpy as np

from catboost import CatBoostClassifier
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sklearn.metrics import accuracy_score,roc_auc_score, f1_score, matthews_corrcoef

from data.KFoldTargetEncoderTrain import KFOLD_TARGET_ENC_COL_POSTFIX
from data.preprocessor import preprocess, get_categorical_cols
from data.utils import get_lbo_pools, gen_seeds, get_catboost_pools

# These are taken from HP
params = {
    'loss_function': 'Logloss',
    'iterations': 200,
    'learning_rate': 0.3668666368559461,
    'l2_leaf_reg': 2,
    'custom_metric': ['Accuracy', 'Recall', 'F1','MCC'],
    'eval_metric': 'AUC',
    #'eval_metric': 'F1',
    'random_seed': 42,
    'logging_level': 'Silent',
    'use_best_model': True,
    'od_type': 'Iter',
    'od_wait': 50,
    #'class_weights': [1,2],
    'depth': 7
}

@click.command()
@click.argument('train_file', type=click.Path())
@click.argument('model_file', type=click.Path())
def main(train_file, model_file):
    """ Runs training
    """
    logger = logging.getLogger(__name__)
    logger.info('Starting training...')
    logger.debug('Loading data...')
    df_train = pd.read_csv(train_file)
    #df_train.drop('card1'+KFOLD_TARGET_ENC_COL_POSTFIX, axis=1, inplace=True)
    df_train = df_train.replace(np.nan, '', regex=True)
    logger.debug('Data loaded. About to create LBO data set..')
    X, y, X_valid, y_valid = get_lbo_pools(df_train)
    train_pool, validate_pool = get_catboost_pools(X, y, X_valid, y_valid)

    SEED = 42
    gen_seeds(SEED)
    model = CatBoostClassifier(**params)
    logger.debug('Data created, about to fit model..')
    model.fit(
        train_pool,
        eval_set=validate_pool,
        logging_level='Info' #='Verbose',
        #plot=False
    )
    logger.debug('Model fitted. About to check..')
    preds_proba = model.predict_proba(X_valid)[:,1]
    preds = model.predict(X_valid)

    logger.info('Accuracy: ' + str(accuracy_score(y_valid, preds)) )
    logger.info('AUC score: ' + str(roc_auc_score(y_valid, preds_proba)) )
    logger.info('F1 score: ' + str(f1_score(y_valid, preds_proba.round())) )
    logger.info('MCC score: ' + str(matthews_corrcoef(y_valid, preds_proba.round())) )

    logger.debug('About to save model to file: '+model_file)
    model.save_model(model_file)
    logger.info('Completed training')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
