# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class PQDCNN(torch.nn.Module):
    def __init__(self):
        super(PQDCNN, self).__init__()
        self.module_0 = py_nndct.nn.Input() #PQDCNN::input_0
        self.module_1 = py_nndct.nn.Conv1d(in_channels=1, out_channels=16, kernel_size=[5], stride=[1], padding=2, dilation=[1], groups=1, bias=True) #PQDCNN::PQDCNN/Sequential[conv_layers]/Conv1d[0]/input.3
        self.module_2 = py_nndct.nn.ReLU(inplace=False) #PQDCNN::PQDCNN/Sequential[conv_layers]/ReLU[1]/313
        self.module_3 = py_nndct.nn.MaxPool1d(kernel_size=[2], stride=[2], padding=0, dilation=[1], ceil_mode=False) #PQDCNN::PQDCNN/Sequential[conv_layers]/MaxPool1d[2]/input.5
        self.module_4 = py_nndct.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=[5], stride=[1], padding=2, dilation=[1], groups=1, bias=True) #PQDCNN::PQDCNN/Sequential[conv_layers]/Conv1d[3]/input.7
        self.module_5 = py_nndct.nn.ReLU(inplace=False) #PQDCNN::PQDCNN/Sequential[conv_layers]/ReLU[4]/339
        self.module_6 = py_nndct.nn.MaxPool1d(kernel_size=[2], stride=[2], padding=0, dilation=[1], ceil_mode=False) #PQDCNN::PQDCNN/Sequential[conv_layers]/MaxPool1d[5]/input.9
        self.module_7 = py_nndct.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=[5], stride=[1], padding=2, dilation=[1], groups=1, bias=True) #PQDCNN::PQDCNN/Sequential[conv_layers]/Conv1d[6]/input.11
        self.module_8 = py_nndct.nn.ReLU(inplace=False) #PQDCNN::PQDCNN/Sequential[conv_layers]/ReLU[7]/365
        self.module_9 = py_nndct.nn.MaxPool1d(kernel_size=[2], stride=[2], padding=0, dilation=[1], ceil_mode=False) #PQDCNN::PQDCNN/Sequential[conv_layers]/MaxPool1d[8]/375
        self.module_10 = py_nndct.nn.Module('shape') #PQDCNN::PQDCNN/377
        self.module_11 = py_nndct.nn.Module('reshape') #PQDCNN::PQDCNN/input.13
        self.module_12 = py_nndct.nn.Linear(in_features=2048, out_features=128, bias=True) #PQDCNN::PQDCNN/Sequential[fc_layers]/Linear[0]/input.15
        self.module_13 = py_nndct.nn.ReLU(inplace=False) #PQDCNN::PQDCNN/Sequential[fc_layers]/ReLU[1]/input
        self.module_14 = py_nndct.nn.Linear(in_features=128, out_features=6, bias=True) #PQDCNN::PQDCNN/Sequential[fc_layers]/Linear[2]/385

    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_0 = self.module_5(output_module_0)
        output_module_0 = self.module_6(output_module_0)
        output_module_0 = self.module_7(output_module_0)
        output_module_0 = self.module_8(output_module_0)
        output_module_0 = self.module_9(output_module_0)
        output_module_10 = self.module_10(input=output_module_0, dim=0)
        output_module_11 = self.module_11(input=output_module_0, shape=[output_module_10,-1])
        output_module_11 = self.module_12(output_module_11)
        output_module_11 = self.module_13(output_module_11)
        output_module_11 = self.module_14(output_module_11)
        return output_module_11
