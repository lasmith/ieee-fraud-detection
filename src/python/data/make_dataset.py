# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd

from data.preprocessor import preprocess


@click.command()
@click.argument('input_dir', type=click.Path( dir_okay=True))
@click.argument('output_dir', type=click.Path())
def main(input_dir, output_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data...')
    df_train, df_test = preprocess(input_dir)
    df_train.to_csv(output_dir + '/train2.csv', index=False, header=True)
    df_test.to_csv(output_dir + '/test2.csv', index=False, header=True)
    logger.info('Completed final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
