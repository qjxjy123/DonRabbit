class _const:
    class ConstError(TypeError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("Can't change const.%s" % name)
        if not name.isupper():
            raise self.ConstCaseError('Const name "%s" is not all uppercase' % name)
        self.__dict__[name] = value


const = _const()
const.HEIGHT = const.WIDTH = 500   # 500 for alex and 1000 for mosaic
const.MODE = 'alex'     # alex or mosaic
const.MASK_COLOR_MODE = 'rgba'
const.DATASET_PATH = 'datasets/COADREAD/%sx%s' % (const.HEIGHT, const.WIDTH)
const.PREDICT_NUM = 11       # alex = 11 x 5 = 55  ; mosaic = 11 x 3x3= 99
const.DATASET_NAME = 'COADREAD_%sx%s' % (const.HEIGHT, const.WIDTH)
const.NB_CLASSES = 4
# const.GENERATOR_PARAMS = dict(
#     horizontal_flip=True,
#     vertical_flip=True,
#     fill_mode='reflect',
#     rotation_range=20,
#     width_shift_range=0.05,
#     height_shift_range=0.05,
#     shear_range=0.05,
#     zoom_range=0.05,
# )

const.GENERATOR_PARAMS = dict(
    fill_mode='reflect'
)