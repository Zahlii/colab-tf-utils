_res = get_ipython().run_cell("""
!pip install tqdm
!pip install keras
!rm tboard.py
!wget https://raw.githubusercontent.com/mixuala/colab_utils/master/tboard.py
!rm -rf log/
""")


import os
import tboard
from tqdm import tqdm
# set paths
ROOT = os.path.abspath('.')
LOG_DIR = os.path.join(ROOT, 'log')

# will install `ngrok`, if necessary
# will create `log_dir` if path does not exist
tboard.launch_tensorboard( bin_dir=ROOT, log_dir=LOG_DIR )

from collections import namedtuple
import keras
from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from tqdm import tqdm


# Represents a Folder or File in your Google Drive
GDriveItem = namedtuple('GDriveItem', ['name', 'fid'])

# Represents Epoch information as returned from keras
EpochData = namedtuple('EpochData', ['epoch', 'losses'])


class GDriveSync:
    """
    Simple up/downloading functionality to move local files into the cloud and vice versa.
    Provides progress bars for both up- and download.
    """

    def __init__(self):
        auth.authenticate_user()
        # prompt the user to access his Google Drive via the API

        self.drive_service = build('drive', 'v3')
        self.default_folder = self.find_items('Colab Notebooks')[0]

    def find_items(self, name):
        """
        Find folders or files based on their name. This always searches the full Google Drive tree!
        :param name: Term to be searched. All files containing this search term are returned.
        :return:
        """
        folder_list = self.drive_service.files().list(q='name contains "%s"' % name).execute()
        folders = []
        for folder in folder_list['files']:
            folders.append(GDriveItem(folder['name'], folder['id']))

        return folders

    def upload_file_to_folder(self, local_file, folder = None):
        """
        Upload a local file, optionally to a specific folder in Google Drive
        :param local_file: Path to the local file
        :param folder: (Option) GDriveItem which should be the parent.
        :return:
        """
        if folder is not None:
            assert type(folder)==GDriveItem	

        file_metadata = {
            'title': local_file,
            'name': local_file
        }

        if folder is not None:
            file_metadata['parents'] = [folder.fid]

        media = MediaFileUpload(local_file, resumable=True)
        created = self.drive_service.files().create(body=file_metadata,
                                                    media_body=media,
                                                    fields='id')

        response = None
        last_progress = 0

        if folder is not None:
            d = 'Uploading file %s to folder %s' % (local_file, folder.name)
        else:
            d = 'Uploading file %s' % local_file

        pbar = tqdm(total=100, desc=d)
        while response is None:
            status, response = created.next_chunk()
            if status:
                p = status.progress() * 100
                dp = p - last_progress
                pbar.update(dp)
                last_progress = p

        pbar.update(100 - last_progress)

    def download_file_to_folder(self, remote_file, path):
        """
        Download a GDriveItem to a local folder
        :param remote_file:
        :param path:
        :return:
        """
        assert type(remote_file)==GDriveItem
        request = self.drive_service.files().get_media(fileId=remote_file.fid)

        last_progress = 0

        pbar = tqdm(total=100, desc='Downloading file %s to %s' % (remote_file.name, path))

        with open(path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                if status:
                    p = status.progress() * 100
                    dp = p - last_progress
                    pbar.update(dp)
                    last_progress = p

        pbar.update(100 - last_progress)

    def delete_file(self, file):
        """
        Delete a remote GDriveItem
        :param file:
        :return:
        """
        assert file==GDriveItem
        request = self.drive_service.files().delete(fileId=file.fid)
        request.execute()


class GDriveCheckpointer(keras.callbacks.Callback):
    """
    Keras Callback that automatically saves models into your Google Drive.
    Outdated checkpoints are automatically deleted remotely to prevent GDrive from filling up.

    Checkpointing is controlled by two functions:
        compare_fn(best_epoch: EpochData, current_epoch: EpochData) -> bool
        - If this function returns true, the current_epoch is assumed to have better performance than the older best_epoch.
        - e.g. return best_epoch.losses['val_acc'] < current_epoch.losses['val_acc']

        filepath_fn(epoch: EpochData) -> Union[String, None]
        - If this function returns None, the checkpoint is skipped. This can be used to skip backing up early epochs.
          If it returns a String path, the model is uploaded into the default GDrive folder with the given file name.
    """
    def __init__(self, compare_fn, filepath_fn):
        assert compare_fn is not None, 'Need a compare function which gets all the losses and evaluation data of two epochs and which needs to return True if the second one is better.'
        assert filepath_fn is not None, 'Need a function that derives a file path based on a dictionary of losses and metrics.'

        super(GDriveCheckpointer, self).__init__()

        self.saver = GDriveSync()

        self.compare_fn = compare_fn
        self.filepath_fn = filepath_fn
        self.best_epoch = None
        self.best_filename = None

    def on_epoch_end(self, epoch, logs={}):
        l = dict(logs)
        d = EpochData(epoch, l)

        if self.best_epoch is None or self.compare_fn(self.best_epoch, d):
            self.best_epoch = d
            fn = self.filepath_fn(d)
            if fn is not None and fn:
                if self.best_filename:
                    os.remove(self.best_filename)
                    old_file = self.saver.find_items(self.best_filename)[0]
                    print('Removing old cloud file %s' % old_file.name)
                    self.saver.delete_file(old_file)
                self.best_filename = fn
                self._save_checkpoint()
            else:
                print('Skipping upload because path function returned no path.')
        else:
            print('No improvement.')

    def _save_checkpoint(self):
        self.model.save(self.best_filename)
        self.saver.upload_file_to_folder(self.best_filename, self.saver.default_folder)
