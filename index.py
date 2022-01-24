from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
import json
app = Flask(__name__)

#Import Data
df = pd.read_csv('data_train.csv', ';',header=0 ,names=['Nama Pegawai', 'Masa Kerja(thn)', 'Usia', 'Nilai Pelatihan', 'Hasil', 'Evaluasi'])
    
#Preprocessing
df['Evaluasi_INT'] = df['Evaluasi'].map({'PROMOSI':1 ,'MUTASI':2, 'PHK':3})

df = df[['Masa Kerja(thn)', 'Usia', 'Nilai Pelatihan', 'Hasil', 'Evaluasi_INT']]

#Normalisasi Min Max Scaller
scaller = MinMaxScaler()
df[['Masa Kerja(thn)', 'Usia', 'Nilai Pelatihan', 'Hasil']] = scaller.fit_transform(df[['Masa Kerja(thn)', 'Usia', 'Nilai Pelatihan', 'Hasil']])
X = np.array(df[['Masa Kerja(thn)', 'Usia', 'Nilai Pelatihan', 'Hasil']])
y = np.array(df['Evaluasi_INT'])

def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))

#Train
# def lvq_fit(train, target, learning_rate, b, max_epoch):
#     label, train_idx = np.unique(target, return_index=True)
#     weight = train[train_idx].astype(np.float64)
#     bobot_awal = train[train_idx].astype(np.float64)
#     train = np.array([e for i, e in enumerate(zip(train, target)) if i not in train_idx])
#     train, target = train[:, 0], train[:, 1]
#     epoch = 0
#     print("HASIL PROSES TRAINING")
#     print("======================")
#     print("Bobot Awal:", bobot_awal)
#     print("")
#     data = [{
#         "Bobot Awal" : bobot_awal
#     }]
#     kata_training = []
#     while epoch < max_epoch:
#         kata = ("Epoch ke-"+ str(epoch+1))
#         kata_training.append(kata)
#         for i, x in enumerate(train):
#             print("Iterasi ke-"+ str(i+1))
#             distance = [sum((w - x) ** 2) for w in weight]
#             # print(distance)
#             min = np.argmin(distance)
#             print("Update bobot kelas ke-", min+1)
#             sign = 1 if target[i] == label[min] else -1
#             weight[min] += sign * learning_rate * (x - weight[min])
#             print(weight)
#             print("")
#             kata_perulangan = ("Iterasi ke-"+ str(i+1))
#             kata_training.append(kata_perulangan)
#             kata_bobot = ("Bobot :", weight)
#             kata_training.append(kata_bobot)
#         learning_rate *= b
#         epoch +=1 
#     data.append({
#         "Training" : kata_training
#     })
#     data.append({
#         "Bobot Akhir" : weight
#     })       
#     data = json.dumps(data, default=default)
#     return data

def lvq_train(train, target, learning_rate, b, max_epoch):
    label, train_idx = np.unique(target, return_index=True)
    weight = train[train_idx].astype(np.float64)
    train = np.array([e for i, e in enumerate(zip(train, target)) if i not in train_idx])
    train, target = train[:, 0], train[:, 1]
    epoch = 0
    print("HASIL PROSES TRAINING")
    print("======================")
    print("Bobot Awal:", weight)
    print("")

    while epoch < max_epoch:
        print("Epoch ke-", epoch+1)
        for i, x in enumerate(train):
            print("Iterasi ke-"+ str(i+1))
            distance = [sum((w - x) ** 2) for w in weight]
            min = np.argmin(distance)
            print("Update bobot kelas ke-", min+1)
            sign = 1 if target[i] == label[min] else -1
            weight[min] += sign * learning_rate * (x - weight[min])
            print(weight)
            print("")

        learning_rate *= b
        epoch +=1 

    return weight, label

def lvq_test(x, W):
    
    W, c = W
    d = [math.sqrt(sum((w - x) ** 2)) for w in W]

    return c[np.argmin(d)]

@app.route('/train')
def train():
    data = lvq_train(X, y, .05, .5, 1)  
    data = json.dumps(data, default=default)  
    return data

@app.route('/test', methods=['POST'])
def testing():
    data = []
    namaPegawai = request.form.get('Nama Pegawai')
    masaKerja = int(request.form.get('Masa Kerja(thn)'))
    usia = int(request.form.get('Usia'))
    nilaiPelatihan = int(request.form.get('Nilai Pelatihan'))
    nilaiKinerja = int(request.form.get('Nilai Kinerja'))
    
    bobot = lvq_train(X, y, .05, .5, 1)
    test = [[masaKerja, usia, nilaiPelatihan, nilaiKinerja]]
    test = np.array(scaller.transform(test))
    result = lvq_test(test[0], bobot)
    if result == 1:
        hasil = 'Promosi'
    elif result == 2:
        hasil = 'Mutasi'
    elif result == 3:
        hasil = 'PHK'
    data.append({
        "nama": namaPegawai,
        "hasil" : hasil
    })    
    data = json.dumps(data, default=default)
    print(data)
    return data

if __name__ == '__main__':
  app.run(debug=True)   