import os
import torch
from uda_digit import *
from collections import OrderedDict

# load model
# model = torch.load('ckps_digits/seed2020/m2u/merge_model/source/Merged_source_MNIST.pt')
model = torch.load('ckps_digits/seed2020/m2u_half/merge_model/student/mnist-lenet5_half-t.pt')


lenet_half_parmas = ['conv', 'fc1', 'fc2', 'bn', 'wn']

print('all params: ', model.keys())
"""
A_params = OrderedDict()
B_params = OrderedDict()
C_params = OrderedDict()

# split model
# distinguish model by its name
for par in model.keys():
    par_name = par.split('.')[0]
    if par_name.startswith(lenet_half_parmas[0]):
        A_params[par] = model[par]
    
    # fc layer - remove number
    elif par_name == lenet_half_parmas[1]:
        ch_li = [ch for ch in par if not ch.isdigit()]
        k_par = ''.join(ch_li)
        B_params[k_par] = model[par]
    elif par_name == lenet_half_parmas[2]:
        ch_li = [ch for ch in par if not ch.isdigit()]
        k_par = ''.join(ch_li)
        C_params[k_par] = model[par]

    elif par_name == lenet_half_parmas[3]:
        B_params[par] = model[par]

    elif par_name == lenet_half_parmas[4]:
        C_params[par] = model[par]


print(A_params.keys())
print(B_params.keys())
print(C_params.keys())


# save model
save_path = 'ckps_digits/seed2020/m2u_half/split_model/student/'
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)
torch.save(A_params, save_path+'source_F.pt')
torch.save(B_params, save_path+'source_B.pt')
torch.save(C_params, save_path+'source_C.pt')
"""
