import zipfile
from urllib.request import urlretrieve
import os

DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00372/HTRU2.zip'


def download_dataset(save_path):
    if os.path.isfile(save_path):
        print("File exist! skip download")
        return

    print("Downloading HTRU2 dataset")
    urlretrieve(DATASET_URL, save_path)


def unzip_dataset(zipfile_path, extract_directory):
    print("Unzipping dataset")
    zip_file = zipfile.ZipFile(zipfile_path, 'r')
    zip_file.extractall(extract_directory)
    zip_file.close()


def prepare_dataset(data_directory='data'):
    filename = 'HTRU2.zip'
    download_dataset(filename)
    unzip_dataset(filename, data_directory)


if __name__ == '__main__':
    prepare_dataset('data')
