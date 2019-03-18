#Copy all of this code to jupyter notebook

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from random import randint

#Variables
jmlh = 10000
panjang = 28
train_iteration = 20
EPOCH = 3
BASIZE = 16

def string_biner(num):
    hasil = str(bin(num)[2:])
    hasil = hasil.rjust(panjang, '0')
    return hasil

def list_int_string_biner(string):
    temp = list(string)
    return list(map(float, temp))

def stringto_int_biner(lmit):
    lmit = list(map(round, lmit))
    lmit = list(map(int, lmit))
    return ''.join(list(map(str, lmit)))

def int_biner(string):
    return int(string, 2)

def MyPrediction(ang):
    angmud = list_int_string_biner(string_biner(ang))
    temp = np.array([angmud])
    tempno = md.predict(temp)[0]
    return int_biner(stringto_int_biner(tempno)), tempno, stringto_int_biner(tempno)

#data
angka = [list_int_string_biner(string_biner(i)) for i in range(jmlh)]
hasil = [list_int_string_biner(string_biner(i**2)) for i in range(jmlh)]
angka = np.array(angka)
hasil = np.array(hasil)

#Make the model
md = Sequential()

md.add(Dense(panjang, input_dim = panjang, activation = 'relu'))
md.add(Dense(panjang*4, activation = 'relu'))
md.add(Dense(panjang*4, activation = 'relu'))
md.add(Dense(panjang, activation = 'sigmoid'))
md.compile(optimizer='adam',
           loss = 'binary_crossentropy',
           metrics=['accuracy'])
md.summary()
print(md.input_shape, md.output_shape)
print(angka.shape, hasil.shape)

#train
for i in range(train_iteration):
    print ("iteration :", i)
    md.fit(angka, hasil, epochs = EPOCH, batch_size = BASIZE)
    
    sementara = [list_int_string_biner(string_biner(randint(1, 100))) for _ in range(10)]
    sementara = np.array(sementara)
    u = 0
    for j in md.predict(sementara):
        print(int_biner(stringto_int_biner(sementara[u])), '-> Real Result ->',
        (int_biner(stringto_int_biner(sementara[u])))**2, 
              '-> AI Result ->', int_biner(stringto_int_biner(j)), end = '')
        if (int_biner(stringto_int_biner(sementara[u]))**2 == int_biner(stringto_int_biner(j))):
            print(" -> AC")
        else:
            print(" -> WA")
        u += 1