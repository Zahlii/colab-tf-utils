# colab-tf-utils
Simple GDrive-Based model checkpointing from within Google's Colab service


Usage:

    !wget https://raw.githubusercontent.com/Zahlii/colab-tf-utils/master/utils.py
    import utils

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

