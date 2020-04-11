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
