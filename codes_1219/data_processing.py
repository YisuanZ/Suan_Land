'''
1. 先获取数据
2. 数据预处理
3. 训练模型
'''
import os
import csv
import h5py
import math
import torch
import numpy as np
from tqdm import tqdm
from aux import seq2feature, feature2seq, TorchStandardScaler

model_conditions = 'Glu'

######################
##Data Preprocessing##
######################



with open(os.path.join('training_data_'+model_conditions+'.txt')) as f:
    reader = csv.reader(f, delimiter="\t")
    d = list(reader)

sequences = [di[0] for di in d]
sequences = sequences[0:2000000]

for i in tqdm(range(0,len(sequences))) :  # len(sequences)
    if (len(sequences[i]) > 110) :
        sequences[i] = sequences[i][-110:]
    if (len(sequences[i]) < 110) : 
        while (len(sequences[i]) < 110) :
            sequences[i] = 'N'+sequences[i]
            
# (31349363, 110, 4)
seqdata_transformed = seq2feature(sequences)
# print(seqdata_transformed.shape)


with h5py.File(os.path.join(model_conditions ,'onehot_sequences_bool.h5'), 'w') as hf:
    hf.create_dataset("onehot_sequences_bool",  data=seqdata_transformed)
# print(type(seqdata_transformed[0][0][0]))

## Now , Create The Data class label vectors and Store in the same h5py file
expressions = [di[1] for di in d]

expressions = expressions[0:2000000]

expdata = np.asarray(expressions)
expdata = expdata.astype('float')  
with h5py.File(os.path.join(model_conditions ,'expression.h5'), 'w') as hf:
    hf.create_dataset("expression",  data=expdata)


########################
##Dataset Construction##
########################


with h5py.File(os.path.join(model_conditions ,'expression.h5'), 'r') as hf:
        expressions = hf['expression'][:]

with h5py.File(os.path.join(model_conditions ,'onehot_sequences_bool.h5'), 'r') as hf:
    onehot_sequences = hf['onehot_sequences_bool'][:]
expdata = expressions
seqdata_transformed = onehot_sequences

randomize  =  np.random.permutation(len(onehot_sequences))
onehot_sequences = onehot_sequences[randomize,:]
expressions = expressions[randomize]

N = len(expressions)

r1 = math.ceil(N*0.7)
r2 = math.ceil(N*0.9)

_trX = onehot_sequences[0:r1]
_trY = expressions[0:r1]
_vaX = onehot_sequences[r1:r2]
_vaY = expressions[r1:r2]
_teX = onehot_sequences[r2:]
_teY = expressions[r2:]

with h5py.File(os.path.join(model_conditions ,'_trX.h5'), 'w') as hf:
    hf.create_dataset("_trX",  data=_trX)  
    
with h5py.File(os.path.join(model_conditions ,'_trY.h5'), 'w') as hf:
    hf.create_dataset("_trY",  data=_trY)  

with h5py.File(os.path.join(model_conditions ,'_vaX.h5'), 'w') as hf:
    hf.create_dataset("_vaX",  data=_vaX)  

with h5py.File(os.path.join(model_conditions ,'_vaY.h5'), 'w') as hf:
    hf.create_dataset("_vaY",  data=_vaY)  

with h5py.File(os.path.join(model_conditions ,'_teX.h5'), 'w') as hf:
    hf.create_dataset("_teX",  data=_teX)  

with h5py.File(os.path.join(model_conditions ,'_teY.h5'), 'w') as hf:
    hf.create_dataset("_teY",  data=_teY) 


########################
###Scaler Construction##
########################

'''
sklearn的StandardScaler转换为pytorch版本
https://www.jianshu.com/p/55e261ce9b3e
https://discuss.pytorch.org/t/pytorch-tensor-scaling/38576/7

将数据集(y值)按照我们预先指定的标准进行正态归一化
'''




synthesized_seqs_filepath = 'synthesized_sequences_results.txt'

def read_synthesized_sequences(filename) :
    with open(filename) as f:
        reader = csv.reader(f, delimiter="\t")
        d = list(reader)
    scura_exp = [di[3] for di in d]
    glu_exp = [di[4] for di in d]
    scura_exp = scura_exp[1:]
    glu_exp = glu_exp[1:]
    return glu_exp,scura_exp

def clean_exp(Y) :
    exp_NA = [(a=='NA') for a in Y]
    exp_NA = np.array(exp_NA)
    Y = np.array(Y)
    clean_exp = Y[~exp_NA]
    clean_exp = [float(a) for a in clean_exp ]
    return clean_exp

glu_exp, _ = read_synthesized_sequences(synthesized_seqs_filepath)
clean_glu_exp = np.array(clean_exp(glu_exp)).reshape(-1, 1)
# clean_glu_exp = torch.tensor(clean_glu_exp)

glu_scaler = TorchStandardScaler()
glu_scaler.fit(clean_glu_exp)

x,y = glu_scaler.getvalue()
res_dict = {"mean":x, "std": y}

scaler_save_path = os.path.join(model_conditions ,"scaler_save.npy")
np.save(scaler_save_path, res_dict)

print(res_dict)






