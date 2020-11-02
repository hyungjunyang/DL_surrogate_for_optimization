# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 14:45:57 2019

@author: kjy
"""

"""
Neural Network, Multi-Layer Perceptron
"""

import numpy as np
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
import keras
from keras.datasets import cifar10
from keras.utils import np_utils, plot_model
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Activation, MaxPooling2D, AveragePooling2D, Dropout, concatenate, Conv2DTranspose, BatchNormalization, Flatten, merge, normalization, LeakyReLU
from keras.models import Model, load_model
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from matplotlib import pyplot
import os

from scipy.io import loadmat, savemat
from sklearn.preprocessing import scale, robust_scale, minmax_scale, maxabs_scale, normalize, StandardScaler
from sklearn.metrics import r2_score
from scipy import stats

def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def root_mean_squared_error(y_true, y_pred):
    from keras import backend as K
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

def round_up(x, place):
    return round(x + 5 * 10**(-1 * (place + 1)), place)

def round_down(x, place):
    return round(x - 5 * 10**(-1 * (place + 1)), place)

## load training data
data = '_c1_6(4)(C)'
train = loadmat('train%s.mat' %data)
validation = loadmat('validation%s.mat' %data)
test = loadmat('test%s.mat' %data)

X_train = train['X_train']
Y_train = train['Y_train']

X_validation = validation['X_validation']
Y_validation = validation['Y_validation']

X_test = test['X_test']
Y_test = test['Y_test']

## reshape input data for training
#X_train = np.append(X_train, X_validation, axis=3)
#Y_train = np.append(Y_train, Y_validation, axis=0)
tmp1 = np.zeros((X_train.shape[3], X_train.shape[0], X_train.shape[1], X_train.shape[2]))
for  i in range(0,X_train.shape[3]):
    for j in range(0,X_train.shape[2]):
        tmp1[i,:,:,j] = X_train[:,:,j,i]
tmp2 = np.zeros((X_validation.shape[3], X_validation.shape[0], X_validation.shape[1], X_validation.shape[2]))        
for  i in range(0,X_validation.shape[3]):
    for j in range(0,X_validation.shape[2]):
        tmp2[i,:,:,j] = X_validation[:,:,j,i]        
tmp3 = np.zeros((X_test.shape[3], X_test.shape[0], X_test.shape[1], X_test.shape[2]))        
for  i in range(0,X_test.shape[3]):
    for j in range(0,X_test.shape[2]):
        tmp3[i,:,:,j] = X_test[:,:,j,i]
X_train = tmp1
X_validation = tmp2
X_test = tmp3        
        
ny = X_train.shape[1]
nx = X_train.shape[2]
nchannel = X_train.shape[-1]

## input nomalization
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean) / std
X_train[np.isnan(X_train)] = 0

mean = np.mean(X_validation, axis=0, keepdims=True)
std = np.std(X_validation, axis=0, keepdims=True)
X_validation = (X_validation - mean) / std
X_validation[np.isnan(X_validation)] = 0

mean = np.mean(X_test, axis=0, keepdims=True)
std = np.std(X_test, axis=0, keepdims=True)
X_test = (X_test - mean) / std
X_test[np.isnan(X_test)] = 0
            
### output nomalization 
temp = np.append(Y_train, Y_validation, axis = 0)
mean0 = np.mean(temp, axis=0)
std0 = np.std(temp, axis=0)
temp = (temp-mean0)/std0
Y_Train = temp[0:Y_train.shape[0]]
Y_Validation = temp[Y_train.shape[0]:]
Y_Test = (Y_test-np.mean(Y_test, axis=0))/np.std(Y_test, axis=0)

### training
model = Sequential()
model.add(Conv2D(128, kernel_size=(3,3), padding='same', 
                 kernel_initializer = 'glorot_uniform', activation = None, 
                 input_shape=(ny,nx,nchannel)))
model.add(BatchNormalization(epsilon=1e-5))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation = None,
                 kernel_initializer = 'glorot_uniform'))
model.add(BatchNormalization(epsilon=1e-5))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation = None,
                 kernel_initializer = 'glorot_uniform'))
model.add(BatchNormalization(epsilon=1e-5))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation = None,
                 kernel_initializer = 'glorot_uniform')) 
model.add(BatchNormalization(epsilon=1e-5))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2), strides=2))

model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation = None,
                 kernel_initializer = 'glorot_uniform')) 
model.add(BatchNormalization(epsilon=1e-5))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2,2), strides=2))

model.add(Flatten())

model.add(Dense(128, kernel_initializer = 'glorot_uniform'))
model.add(BatchNormalization(epsilon=1e-5))
model.add(Activation('relu'))

model.add(Dropout(0.3))

model.add(Dense(1, kernel_initializer='glorot_uniform'))
model.add(BatchNormalization(epsilon=1e-5))
model.add(Activation('linear'))

adam = optimizers.adam(lr=0.001)
model.compile(optimizer=adam, loss=root_mean_squared_error, metrics=[root_mean_squared_error, 'mean_squared_error'])

#early_stopping = EarlyStopping(monitor='val_loss', mode='min', baseline=0.01, patience=0)
#os.remove('best_model.h5')
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
history = model.fit([X_train],Y_Train, validation_data=(X_validation,Y_Validation),
                    epochs=100, batch_size=80, verbose=2, callbacks=[model_checkpoint])

saved_model = load_model('best_model.h5', custom_objects={'root_mean_squared_error': root_mean_squared_error})

Y_prediction = saved_model.predict(X_train)*std0+mean0
Y_prediction2 = saved_model.predict(X_validation)*std0+mean0
Y_prediction3 = saved_model.predict(X_test)*std0+mean0

### plot
slope, intercept, r_value, p_value, std_err = stats.linregress(Y_train.reshape(1,Y_prediction.shape[0]), Y_prediction.reshape(1,Y_prediction.shape[0]))
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(Y_validation.reshape(1,Y_prediction2.shape[0]), Y_prediction2.reshape(1,Y_prediction2.shape[0]))
slope3, intercept3, r_value3, p_value3, std_err3 = stats.linregress(Y_test.reshape(1,Y_prediction3.shape[0]), Y_prediction3.reshape(1,Y_prediction3.shape[0]))

r_value = r2_score(Y_train.reshape(Y_prediction.shape[0],), Y_prediction.reshape(Y_prediction.shape[0],))
r_value2 = r2_score(Y_validation.reshape(Y_prediction2.shape[0],), Y_prediction2.reshape(Y_prediction2.shape[0],))
r_value3 = r2_score(Y_test.reshape(Y_prediction3.shape[0],), Y_prediction3.reshape(Y_prediction3.shape[0],))

pyplot.figure(figsize=(4,4)) 
pyplot.plot(history.history['val_mean_squared_error'])
pyplot.title('val_rmse = %1.3f' %min(history.history['val_mean_squared_error']),fontsize=16)
pyplot.savefig('val_rmse.png', dpi = 300, bbox_inches='tight')

idx = history.history['val_mean_squared_error'].index(min(history.history['val_mean_squared_error']))
pyplot.figure(figsize=(4,4)) 
pyplot.plot(history.history['mean_squared_error'])
pyplot.title('rmse = %1.3f' %history.history['mean_squared_error'][idx],fontsize=16)
pyplot.savefig('rmse.png', dpi = 300, bbox_inches='tight')

MAX = round_up(np.max(np.append(np.append(Y_test, Y_validation, axis = 0), Y_test, axis = 0)), -7)
MIN = round_down(np.min(np.append(np.append(Y_test, Y_validation, axis = 0), Y_test, axis = 0)), -7)
pyplot.figure(figsize=(4,4)) 
pyplot.plot(Y_train, Y_prediction, '.')
C, S = np.arange(MIN, MAX, 0.01e8), slope*np.arange(MIN, MAX,0.01e8)+intercept
pyplot.plot(np.arange(MIN, MAX, 0.01e8), C, label='y=x')
pyplot.plot(np.arange(MIN, MAX, 0.01e8), S, label='y=%1.3fx+%1.2e' %(slope, intercept) )
pyplot.legend(loc=2)
pyplot.axis('square')
#pyplot.axis([-4, 4, -4, 4])
pyplot.axis([MIN, MAX, MIN, MAX])
pyplot.xticks(np.arange(MIN, MAX, step = 0.2e8))
pyplot.yticks(np.arange(MIN, MAX, step = 0.2e8))
#pyplot.title('Training data ($R^2$ = %1.3f) \n' %r_value ,fontsize=16)
pyplot.title('$R^2$ = %1.3f' %r_value ,fontsize=16)
pyplot.xlabel('True NPV ($)' %r_value ,fontsize=16, fontname='Times New Roman')
pyplot.ylabel('Predicted NPV ($)' %r_value ,fontsize=16, fontname='Times New Roman')
pyplot.savefig('training.png', dpi = 300, bbox_inches='tight')

pyplot.figure(figsize=(4,4)) 
pyplot.plot( Y_validation, Y_prediction2, '.')
C, S = np.arange(MIN, MAX,0.01e8), slope2*np.arange(MIN, MAX,0.01e8)+intercept2
pyplot.plot(np.arange(MIN, MAX,0.01e8), C, label='y=x')
pyplot.plot(np.arange(MIN, MAX,0.01e8), S, label='y=%1.3fx+%1.2e' %(slope2, intercept2) )
pyplot.legend(loc=2)
pyplot.axis('square')
pyplot.axis([MIN, MAX, MIN, MAX])
pyplot.xticks(np.arange(MIN, MAX, step = 0.2e8))
pyplot.yticks(np.arange(MIN, MAX, step = 0.2e8))
pyplot.title('$R^2$ = %1.3f' %r_value2 ,fontsize=16)
pyplot.xlabel('True NPV ($)' %r_value ,fontsize=16, fontname='Times New Roman')
pyplot.ylabel('Predicted NPV ($)' %r_value ,fontsize=16, fontname='Times New Roman')
pyplot.savefig('validation.png', dpi = 300, bbox_inches='tight')

pyplot.figure(figsize=(4,4)) 
pyplot.plot(Y_test, Y_prediction3,  '.')
C, S = np.arange(MIN, MAX,0.01e8), slope3*np.arange(MIN, MAX,0.01e8)+intercept3
pyplot.plot(np.arange(MIN, MAX,0.01e8), C, label='y=x')
pyplot.plot(np.arange(MIN, MAX,0.01e8), S, label='y=%1.3fx+%1.2e' %(slope3, intercept3) )
pyplot.legend(loc=2)
pyplot.axis('square')
pyplot.axis([MIN, MAX, MIN, MAX])
pyplot.xticks(np.arange(MIN, MAX, step = 0.2e8))
pyplot.yticks(np.arange(MIN, MAX, step = 0.2e8))
pyplot.title('$R^2$ = %1.3f' %r_value3 ,fontsize=16)
pyplot.xlabel('True NPV ($)' %r_value ,fontsize=16, fontname='Times New Roman')
pyplot.ylabel('Predicted NPV ($)' %r_value ,fontsize=16, fontname='Times New Roman')
pyplot.savefig('test.png', dpi = 300, bbox_inches='tight')


import pickle
with open('variables.pickle', 'wb') as f:
    pickle.dump([Y_prediction, Y_prediction2, Y_prediction3, Y_train, Y_validation, Y_test] , f)

#import pickle  
#with open('variables.pickle', 'rb') as f:
#    Y_prediction, Y_prediction2, Y_prediction3, Y_train, Y_validation, Y_test = pickle.load(f)

savemat('data.mat', {'y_train': Y_train, 'y_validation': Y_validation, 'y_test': Y_test, 
                              'y_train_p': Y_prediction, 'y_validation_p': Y_prediction2, 'y_test_p': Y_prediction3})
savemat('History.mat', {'History': history.history})
    
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
plot_model(saved_model, to_file='model_plot.pdf', show_shapes=True, show_layer_names=True)


