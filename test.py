import shutil

from option import Options
import os
import keras
from utils.utils_process.common_utils import time_calculate
class Tester():
    def __init__(self,args):
        self.args = args

        # dataset path
        dataset_path = os.path.join(args.data_folder, args.dataset, '{}x{}'.format(args.base_size,args.base_size))
        test_path = os.path.join(dataset_path,'test')
        print(test_path)
        # dataloader
        if args.multi_inputs:
            from experiments.data_process_multi_inputs import testGenerator, testGenerator_sequence, saveResult
        else:
            from experiments.data_process_single_input import testGenerator, testGenerator_sequence, saveResult
        if args.nbclasses == 3:
            mask_color_mode = 'rgb'
        elif args.nbclasses == 4:
            mask_color_mode = 'rgba'

        self.testGene = testGenerator(test_path, crop_mode=args.test_eval_mode, target_size=args.base_size,
                                 mask_color_mode=mask_color_mode)
        self.predict_names = testGenerator_sequence(test_path, args.test_eval_mode, predict_num=args.test_num,
                                                    predict_base_num=args.test_base_num,
                                               target_size=args.base_size, mask_color_mode=mask_color_mode)
        # model
        if args.model == 'unet_vanilla':
            from models.unet_vanilla import unet
            self.model = unet(loss=args.loss, nb_classes=args.nbclasses)
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

        weights_name = os.path.join('weights', args.checkname)
        self.model.load_weights(weights_name)

        self.saveResult = saveResult
        # log path
        self.predict_save_path = os.path.join(test_path,'predict')
        self.visualize_path = os.path.join('visualization', args.dataset, args.model + '_' + args.loss)
        self.logs_path = os.path.join('logs', args.dataset, args.model + '_' + args.loss)


    def evaluate(self):
        if os.path.isdir(self.predict_save_path):
            shutil.rmtree(self.predict_save_path, True)

        results = self.model.predict_generator(self.testGene, self.args.test_num, verbose=1)
        # results = None
        time_calculate(self.saveResult,save_path=self.predict_save_path, mode=self.args.test_eval_mode, npyfile=results,
                   drawCircle_path=self.visualize_path,times=self.args.loss_nr_times,
                   metrics_path=self.logs_path, predict_names=self.predict_names, useGaussian=self.args.Gaussian)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    args = Options().parse()
    tester = Tester(args)
    print('Model info: ',args.checkname)
    time_calculate(tester.evaluate)