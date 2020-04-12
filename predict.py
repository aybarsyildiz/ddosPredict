import math
import tensorflow as tf
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split

#veriler pandas kütüphanesi ile çekilmesi
normaldata = pd.read_csv("ddosdata.csv") #saldiri olmayan veriler
attackdata = pd.read_csv("attackdata.csv") #saldiri veriler

Normal_train, Normal_test, Attack_train, Attack_test = train_test_split(
    normaldata,attackdata,test_size = 0.33, random_state=300
)
#parametreler
learning_rate = 0.1
num_steps = 500

#sinirağı parametreleri
n_hidden_1 = 256
n_hidden_2 = 256
num_input = 11100
num_classes = 10

weights = {
    'h1': tf.Variable(tf.random.normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random.normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random.normal([n_hidden_1])),
    'b2': tf.Variable(tf.random.normal([n_hidden_2])),
    'out': tf.Variable(tf.random.normal([num_classes]))
}

# sinirağ modeli oluşturulması
def neural_net(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
  
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer