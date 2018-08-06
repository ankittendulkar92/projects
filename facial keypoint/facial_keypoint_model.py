import os
import matplotlib.pyplot as plt
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import time

FTRAIN=os.getcwd()+'/data/training.csv'
FTEST=os.getcwd()+'/data/test.csv'
FIdLookup=os.getcwd()+'/data/IdlookupTable.csv'


# setting up gpu usage limit and using single gpu
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
print(tf.__version__)
config=tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.95
config.gpu_options.visible_device_list='0'
set_session(tf.Session(config=config))

def plot_sample(x,y,axs):
    #SHow Image and scatter plot 
    #rescale y 1 to -1
    axs.imshow(x.reshape(96,96),cmap="gray")
    axs.scatter(48*y[0::2]+48,48*y[1::2]+48)

# define the load function to get the input data for testing and training 
# which can be chose by the test=true or false 
def load(test=False,cols=None):
    """cols: list containing landmark label names
    if this is specified only a subset is extracted
    ex:[left_eye_center_x,lef_eye_center_y]
    
    return:
        x:2-d numpy array(nsample,ncol*nrow)
        y:2-d numpy array(nsample,nlandmarks*2)"""
        
    fname=FTEST if test else FTRAIN
    df=read_csv(os.path.expanduser(fname))
    
    #convert image column data seperated by , in excel to an array with the values 
    df['Image']=df['Image'].apply(lambda im:np.fromstring(im,sep=' '))
    
    if cols:
        df=df[list(cols)+['Image']]
    myprint=df.count()
    myprint=myprint.reset_index()
    print(myprint)
    
    #remove rows with atleast one n/a
    df=df.dropna()
    
    x=np.vstack(df['Image'].values)/255 # change values to 0 to 1 and stack in one after another in a column
    x=x.astype(np.float32)
    
    if not test:
        y=df[df.columns[:-1]].values #choose all columns except the last image data column
        y=(y-48)/48 # values between -1 and 1
        x,y=shuffle(x,y,random_state=42) # shuffle data randomly together
        y=y.astype(np.float32)
    else:
        y=None
        
    return x,y


#load data as (nosamples,norows,nocolumns,1) from the excel data so we have images stored as data
def load2d(test=False,cols=None):
    re=load(test,cols)
    
    # keeps the number of rows same and splits columns into 96,96
    x=re[0].reshape(-1,96,96,1)
    y=re[1]
    
    return x,y


def plot_loss(hist,name,plt,RMSE_TF=False):
    '''
    RMSE_TFTrue then the rmse is plotted with the original scale
    '''
    loss=hist['loss']
    val_loss=hist['val_loss']
    if RMSE_TF:
        loss=np.sqrt(np.array(loss))*48
        val_loss=np.sqrt(np.array(val_loss))*48
        
    plt.plot(loss,"--",linewidth=3,label='train:'+name)
    plt.plot(val_loss,linewidth=3,label='val:'+name)


#LOAD THE DATA
x,y=load()




## SINGLE LAYER NEETWORK

from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import SGD

# create single layer with 100 neurons and 30 outut as y has 30 columns for locations
model=Sequential()
model.add(Dense(100,input_dim=x.shape[1]))
model.add(Activation('relu'))
model.add(Dense(30))
start=time.time()
sgd=SGD(lr=0.01,momentum=0.9,nesterov=True)
model.compile(loss='mean_squared_error',optimizer=sgd)
hist=model.fit(x,y,epochs=100,validation_split=0.2,verbose=False)
end=time.time()
print(end-start)

#generate plot
plot_loss(hist.history,"model 1",plt)
plt.legend()
plt.grid()
plt.yscale("log")
plt.xlabel("epoch")
plt.ylabel("log loss")
plt.show()

#evaluate model
x_test,_=load(test=True)
y_test=model.predict(x_test)

#plot some examples
fig=plt.figure(figsize=(7,7))
fig.subplots_adjust(hspace=0.13,wspace=0.0001,left=0,right=1,bottom=0,top=1) #adjusr spaces on sides and between pics

npic=9
count=1
#create subplots and choose random pics and corresponding coordinates form y_test to show points
for irow in range(npic):
    ipic=np.random.choice(x_test.shape[0])
    ax=fig.add_subplot(npic/3,3,count,xticks=[],yticks=[])
    plot_sample(x_test[ipic],y_test[ipic],ax) #plot sample uses image and scatters points across it based on y_test coordinates
    ax.set_title('picture'+str(ipic))
    count+=1
plt.show()
 
    
#(optinal) SAVE MODEL WEIGHTS
'''
from keras.models import model_from_json

def save_model(model,name):
    
    json_string=model.to_json()
    open(name+'_setup.json','w').write(json_string)
    model.save_weights(name+'_weights.h5')

def load_model(name):
    model=model_from_json(open(name+'_setup.json').read())
    model.load_weights(name+'_weights.h5')
    return(model)

save_model(model,'model1')
model=load_model('model1')

'''

