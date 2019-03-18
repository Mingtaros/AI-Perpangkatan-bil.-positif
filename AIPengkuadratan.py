#Copy all of this code (per segment) to jupyter notebook

from keras.models import Sequential, model_from_json
from keras.layers import Dense
import numpy as np
from random import randint
import matplotlib.pyplot as plt

#Variables
jmlh = 10000
panjang = 28
train_iteration = 200
EPOCH = 3
BASIZE = 16

#functions to convert input
def string_biner(num):
    hasil = str(bin(num)[2:])
    hasil = hasil.rjust(panjang, '0') #string padding
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

#function to show prediction result
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
md.add(Dense(panjang*8, activation = 'relu'))
md.add(Dense(panjang*4, activation = 'relu'))
md.add(Dense(panjang, activation = 'sigmoid'))
md.compile(optimizer='adam',
           loss = 'binary_crossentropy',
           metrics=['accuracy'])
md.summary()

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

#show training result
vis_angka = [int_biner(stringto_int_biner(i)) for i in angka]
vis_hasil = [int_biner(stringto_int_biner(j)) for j in hasil]

plt.subplot(1,2,1)
plt.title("Hasil asli")
plt.plot(vis_hasil)

ai_hasil = []
for k in md.predict(angka):
    ai_hasil.append(int_biner(stringto_int_biner(k)))

plt.subplot(1,2,2)
plt.title("Hasil AI")
plt.plot(ai_hasil, 'g')
plt.show()


# saving AI to .json
md_json = md.to_json()
with open("MyAIgen1.json", "w") as json_file:
    json_file.write(md_json)
# saving AI weights to .h5
md.save_weights("MyAIgen1.h5")
print("Saved model to disk")

#load json model
json_file = open("MyAIgen1.json", 'r')
md_load_from_json = json_file.read()
json_file.close()
md = model_from_json(md_load_from_json)
# load weights into the loaded model
md.load_weights("MyAIgen1.h5")
print("Loaded model from disk")

md.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])