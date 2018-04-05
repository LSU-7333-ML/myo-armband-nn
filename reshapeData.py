import numpy as np

def reshapeData(x, y, size = 1): #origin size is 4, 16*4, default resize to 16*1
    x1 = x.reshape(-1, size*16)
    print('reshaped x: ', x1.shape)
    if x1.shape[0] <= y.shape[0]:
        y1 = y[0:x1.shape[0]]
        print('reshaped y: ', y1.shape)
    else:
        lable = y[0:1]
        lable1 = lable.repeat(x1.shape[0] - y.shape[0], axis = 0)
        y1 = np.concatenate((y, lable1), axis = 0)
        print('reshaped y: ', y1.shape)
    return x1, y1

def resample(x, y, opt = 0): # opt=0 choose first signal, opt = 1 choose last one
    length = x.shape[1]
    if opt == 0:
        L = [x for x in range(length) if x % 2 == 0]
        x1 = x.T[L].T
    elif opt == 1:
        L = [x for x in range(length + 1) if x % 2 != 0]
        x1 = x.T[L].T
    else:
        print("please input correct 'opt' value")
    return x1, y

filepath = 'deng_0'
savepath = filepath + '_1'
npzfile = np.load(filepath + '.npz')
x = npzfile['x']
y = npzfile['y']
print('Original shape: ', x.shape, y.shape)
x1, y1 = reshapeData(x, y, size = 1)
#x1, y1 = resample(x, y, 1)
#np.savez(savepath + '.npz', x = x1, y = y1)
