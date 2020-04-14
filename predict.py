from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense


XnormalDataset = loadtxt('ddosdata.csv',delimiter=',') 
XattackDataset = loadtxt('attackdata.csv',delimiter=',')
normalDataset = XnormalDataset[:,0:37]
attackDataset = XattackDataset[:,0:37]
print(normalDataset.shape)
print(attackDataset.shape)
model = Sequential()
shape = normalDataset[0].shape
model.add(Dense(12,input_shape=shape, activation = 'relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(37,activation='sigmoid'))


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(normalDataset,attackDataset,epochs = 300, batch_size=10)
