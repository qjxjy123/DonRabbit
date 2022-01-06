import os
import argparse

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='Keras \
            cell recognition')
        # model and dataset
        parser.add_argument('--model', type=str, default='unet_vanilla',
                            help='model name (default: unet_vanilla, select list: \
                            unet_vanilla, FCN8s, FRCN, FCRN_A,FCRN_B, AWMF_CNN, SFCN_OPI, MapDe, MVC)')

        parser.add_argument('--dataset', type=str, default='pan-NETs',
                            help='dataset name (default: pan-NETs, select list:\
                            pan-NETs, COADREAD)')
        parser.add_argument('--data-folder', type=str,
                            default='datasets',
                            help='training dataset folder (default: \
                            $(HOME)/datasets)')

        parser.add_argument('--base-size', type=int, default=None, metavar='BS',
                            help='base image size (default: auto)')
        parser.add_argument('--crop-size', type=int, default=256,
                            help='crop image size')


        # loss-nr
        parser.add_argument('--loss', type=str,default='mapping_loss',help='nucleus regression loss(default: \
                                               mapping_loss, select list: mse, mapping loss, reverseMapping loss)')
        parser.add_argument('--final-loss-nr-weight', type=float, default=1.0, help='final loss-nr weight \
                                                                                            (default: 1.0)')
        parser.add_argument('--aux-loss-nr-weight',type=float,default=0.5,help='aux loss-nr weight(default: 0.5)')

        parser.add_argument('--loss-nr-times',type=float,default=5.0,help='result map values expansion multiple \
                                                                              default:5.0)')
        parser.add_argument('--loss-nr-beta', type=float, default=5.0, help='loss-nr gamma(default:0.2)')
        parser.add_argument('--loss-nr-lambda', type=float, default=5.0, help='loss-nr lambda(default:1.0)')
        # loss-lc
        parser.add_argument('--aux', action='store_true', default= False,
                            help='Auxilary Loss')
        parser.add_argument('--aux-weight', type=float, default=0.2,
                            help='Auxilary loss weight (default: 0.2)')
        parser.add_argument('--loss-lc', action='store_true', default= False,
                            help='Semantic Encoding Loss SE-loss')
        parser.add_argument('--loss-lc-weight', type=float, default=0.2,
                            help='SE-loss weight (default: 0.2)')
        # training hyper params
        parser.add_argument('--multi-inputs', type=bool, default=None,
                            help='number of epochs to train (default: auto)')
        parser.add_argument('--epochs', type=int, default=500, metavar='N',
                            help='number of epochs to train (default: 500)')
        parser.add_argument('--start_epoch', type=int, default=0,
                            metavar='N', help='start epochs (default:0)')
        parser.add_argument('--batch-size', type=int, default=4,
                            metavar='N', help='input batch size for \
                            training (default: 4)')
        parser.add_argument('--test-batch-size', type=int, default=1,
                            metavar='N', help='input batch size for \
                            testing (default: 1)')
        parser.add_argument('--nbclasses',type=int,default=None,
                            help='number of classes( default: auto')
        # optimizer params
        parser.add_argument('--optimizer' ,type=str, default='adam',
                            help='optimizer (default: adam)')
        parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                            help='learning rate (default: 1e-4)')

        # evaluation
        parser.add_argument('--no-val', action='store_true', default=False,
                            help='skip validation during training')
        parser.add_argument('--test-eval-mode', type=str, default='mosaic',
                            help='evaluate image mode (default:mosaic,select list: mosaic, alex')
        parser.add_argument('--test-base-num', type=int, default=41,
                            help='evaluate whole image number (default:41 ')
        parser.add_argument('--test-num', type=int, default=None,
                            help='evaluate image number (default:auto ')
        # pro-process
        parser.add_argument('--Gaussian', action='store_true', default=False,
                            help='use Gaussian filter before peak local maximal(default:False)')

        parser.add_argument('--denoise',action='store_true', default=True,
                            help='use denoise(default: True)')
        parser.add_argument('--denoise-radius',type=int, default=4,
                            help='denoise radius(default: 4)')
        # denoise method: Mitosis Detection in Breast Cancer Histology Images with Deep Neural Networks

        # visualize
        parser.add_argument('--visualize-result', action='store_true', default=False)



        # logging
        parser.add_argument('--tensorboard-dir', type=str, default=None)
        # checking point
        parser.add_argument('--checkname', type=str, default=None,
                            help='set the checkpoint name (default: auto)')

        # test option
        parser.add_argument('--test-folder', type=str, default=None,
                            help='path to test image folder')
        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        # default settings for base_size, checkname

        if args.base_size is None:
            base_sizes = {
                'pan-NETs': 1000,
                'COADREAD': 500,
                'BCData': 640,
            }
            args.base_size = base_sizes[args.dataset]

        if args.checkname is None:
            args.checkname = args.dataset + '_' + args.model + '_' + args.loss

        if args.multi_inputs is None:
            models_input = {
                'AWMF_CNN': True,
                'MVC': True,
                'FRCN': False,
                'FCRN_A': False,
                'FCN8s': False,
                'FCRN_B': False,
                'SFCN_OPI': False,
                'MapDe': False,
                'unet_vanilla': False,
            }
            args.multi_inputs = models_input[args.model]
        if args.nbclasses is None:
            nbs = {
                'bcdata': 3,
                'pan-nets': 3,
                'coadread': 4,
            }
            args.nbclasses = nbs[args.dataset.lower()]

        if args.tensorboard_dir is None:
            args.tensorboard_dir = os.path.join('tensorboard',args.checkname)

        if args.test_num is None:
            test_nums = {
                'alex_coadread' : args.test_base_num * 5,
                'mosaic_coadread' : args.test_base_num * 9,
                'alex_pan-nets' : args.test_base_num * 5,
                'mosaic_pan-nets': args.test_base_num * 25,
                'mosaic_bcdata': args.test_base_num * 9,
            }
            args.test_num = test_nums[args.test_eval_mode+'_'+args.dataset.lower()]
        print(args)
        return args

