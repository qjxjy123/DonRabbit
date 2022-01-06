from option import Options
import os
import keras
from utils.utils_process.common_utils import time_calculate
import tensorflow as tf
import glob
class Trainer():
    def __init__(self,args):
        self.args = args
        # data_transforms
        generator_params = dict(
            fill_mode='reflect'
        )
        train_data_gen_args = generator_params
        valid_data_gen_args = dict()

        # dataset path
        dataset_path = os.path.join(args.data_folder, args.dataset, '{}x{}'.format(args.base_size,args.base_size))
        self.train_path = os.path.join(dataset_path,'train')
        self.valid_path = os.path.join(dataset_path,'valid')
        print(self.train_path)
        # dataloader
        if args.multi_inputs:
            from experiments.data_process_multi_inputs import trainGenerator
        else:
            from experiments.data_process_single_input import trainGenerator
        if args.nbclasses == 3:
            mask_color_mode = 'rgb'
        elif args.nbclasses == 4:
            mask_color_mode = 'rgba'
        self.myGeneTrain = trainGenerator(args.batch_size, self.train_path, 'image', 'label',
                                     train_data_gen_args, target_size=(args.base_size, args.base_size),
                                     mask_color_mode=mask_color_mode)  # ,save_to_dir = r'data/cell/train/aug/'
        self.myGeneValid = trainGenerator(args.batch_size, self.valid_path, 'image', 'label', valid_data_gen_args,
                                          target_size=(args.base_size, args.base_size),
                                     mask_color_mode=mask_color_mode)
        # model
        if args.model == 'unet_vanilla':
            from models.unet_vanilla import unet
            self.model = unet(loss=args.loss, nb_classes=args.nbclasses,times=args.loss_nr_times,beta=args.loss_nr_beta
                              , lambd=args.loss_nr_lambda)
        if args.model == 'FCRN_A':
            from models.FCRN_A import FCRN_A
            self.model = FCRN_A(img_dim=args.crop_size, batch_size=args.batch_size).inference(nb_classes=args.nbclasses,
                                                    LOSS_INFO=args.loss,times=args.loss_nr_times,beta=args.loss_nr_beta
                                                    , lambd=args.loss_nr_lambda)
        if args.model == 'FCRN_B':
            from models.FCRN_B import FCRN_B
            self.model = FCRN_B(img_dim=args.crop_size, batch_size=args.batch_size).inference(nb_classes=args.nbclasses,
                                                    LOSS_INFO=args.loss,times=args.loss_nr_times,beta=args.loss_nr_beta
                                                    , lambd=args.loss_nr_lambda)
        if args.model == 'FRCN':
            from models.FRCN import unet
            self.model = unet(loss=args.loss, nb_classes=args.nbclasses,times=args.loss_nr_times,beta=args.loss_nr_beta
                              , lambd=args.loss_nr_lambda)
        if args.model == 'FCN8s':
            from models.FCN8s import FCN8s
            self.model = FCN8s().fcn8s(loss=args.loss, nb_classes=args.nbclasses,times=args.loss_nr_times,beta=args.loss_nr_beta
                              , lambd=args.loss_nr_lambda)
        if args.model == 'MVC':
            from models.MVC import MVC
            self.model = MVC(img_dim=args.crop_size, batch_size=args.batch_size,nb_classes=args.nbclasses,times=args.loss_nr_times).build_MVC(loss=args.loss, nb_classes=args.nbclasses,times=args.loss_nr_times,beta=args.loss_nr_beta
                              , lambd=args.loss_nr_lambda)
        if args.model == 'SFCN_OPI':
            from models.SFCN_OPI import SFCNnetwork
            self.model = SFCNnetwork(input_shape=args.crop_size).joint_branch(loss=args.loss, nb_classes=args.nbclasses,times=args.loss_nr_times,beta=args.loss_nr_beta
                              , lambd=args.loss_nr_lambda)
        if args.model == 'AWMF_CNN':
            from models.AWMF_CNN import AWMF
            self.model = AWMF(img_dim=args.crop_size, batch_size=args.batch_size).build_AWMF(loss=args.loss, nb_classes=args.nbclasses,times=args.loss_nr_times,beta=args.loss_nr_beta
                              , lambd=args.loss_nr_lambda)
        # log
        weights_name = os.path.join('weights',args.checkname)
        moniter = 'val_loss'
        self.model_checkpoint = keras.callbacks.ModelCheckpoint(weights_name, monitor=moniter, verbose=1, save_best_only=True,
                                                           save_weights_only=True)
        self.model_earlystopping = keras.callbacks.EarlyStopping(monitor=moniter, patience=25, restore_best_weights=True)
        self.tbcallbacks = keras.callbacks.TensorBoard(log_dir=args.tensorboard_dir, histogram_freq=0, write_graph=True,
                                          write_images=True)


    def training(self):
        nums_orig_imgs_train = len(glob.glob(os.path.join(self.train_path,'image/*.png')))
        nums_orig_imgs_valid = len(glob.glob(os.path.join(self.valid_path,'image/*.png')))
        nums_crop_per_img = 8
        batch_num_train = nums_orig_imgs_train * nums_crop_per_img // self.args.batch_size
        batch_num_valid = nums_orig_imgs_valid * nums_crop_per_img // self.args.batch_size
        print('batch_num : ', batch_num_train,batch_num_valid)
        history = self.model.fit_generator(self.myGeneTrain, steps_per_epoch=batch_num_train, epochs=self.args.epochs,
                                        validation_data=self.myGeneValid,
                                        validation_steps=batch_num_valid,
                                        callbacks=[self.model_checkpoint, self.tbcallbacks, self.model_earlystopping])

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    args = Options().parse()
    trainer = Trainer(args)
    print('Model info: ',args.checkname)
    time_calculate(trainer.training)