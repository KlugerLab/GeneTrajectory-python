import logging
from urllib.request import urlretrieve
import hashlib
import os

logger = logging.getLogger()


def download_file_if_missing(
        file: str,
        url: str,
        create_target_folder_if_missing: bool = False,
        md5_hash: str = None,
):
    """
    Checks if a file exists and downloads it if missing

    :param file: where the file should be saved
    :param url: the url to download from
    :param create_target_folder_if_missing: creates the folder where the target file should be saved if not existing
    :param md5_hash: the md5 hash of the file
    """
    outdir = os.path.dirname(file)
    if outdir and not os.path.isdir(outdir):
        if create_target_folder_if_missing:
            os.makedirs(outdir, exist_ok=True)
        else:
            raise ValueError(f'Directory does not exist: {outdir}')

    if not os.path.exists(file):
        print(f'\nDownloading {url} to {file}...')
        urlretrieve(url, file)
        print(f'File saved on {file}')
    if md5_hash:
        check_md5_hash(file, md5_hash)


def check_md5_hash(file: str, md5_hash: str):
    with open(file, "rb") as f:
        hexdigest = hashlib.file_digest(f, "md5").hexdigest() # noqa
    if hexdigest != md5_hash:
        raise ValueError(f'The md5 hash of {file} is {hexdigest} and does not match {md5_hash}')
