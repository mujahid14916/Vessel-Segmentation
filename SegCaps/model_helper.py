'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This is a helper file for choosing which model to create.
'''
import tensorflow as tf

def create_model(net, input_shape):
    if net == 'segcapsr1':
        from SegCaps.capsnet import CapsNetR1
        model_list = CapsNetR1(input_shape)
        return model_list
    elif net == 'segcapsr3':
        from SegCaps.capsnet import CapsNetR3
        model_list = CapsNetR3(input_shape)
        return model_list
    elif net == 'capsbasic':
        from SegCaps.capsnet import CapsNetBasic
        model_list = CapsNetBasic(input_shape)
        return model_list
    else:
        raise Exception('Unknown network type specified: {}'.format(args.net))
