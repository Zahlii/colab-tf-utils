# colab-tf-utils
Simple GDrive-Based model checkpointing from within Google's Colab service


## Usage:
### Creating checkpoint callback for keras
    !wget https://raw.githubusercontent.com/Zahlii/colab-tf-utils/master/utils.py
    import utils
    import os
    import keras

    def compare(best, new):
      return best.losses['val_acc'] < new.losses['val_acc']

    def path(new):
      if new.losses['val_acc'] > 0.8:
        return 'VGG16_%s.h5' % new.losses['val_acc']

    callbacks = cb = [
          utils.GDriveCheckpointer(compare,path),
          keras.callbacks.TensorBoard(log_dir=os.path.join(utils.LOG_DIR,'VGG16'))
    ]
### Downloading a file from google drive
    from utils import GDriveSync
    downloader=GDriveSync()
    filename='VGG16_0.81.h5'
    drive_file_path=downloader.find_items(filename)[0]
    downloader.download_file_to_folder(drive_file_path,filename)
