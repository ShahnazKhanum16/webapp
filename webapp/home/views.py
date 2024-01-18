from django.shortcuts import render
from .models import *
import pandas as pd
from machine_learning.preprocessing import preprocess
from machine_learning.train import DT, RF, NB, knn, adb
from sklearn.model_selection import train_test_split
import numpy as np

# Create your views here.


def index(request):
    return render(request, 'index.html')

# Uploading the dataset.
# global path

def upload_dataset(request):
    if request.method == 'POST':
        file = request.FILES['file']
        # d = dataset(file=file)
        # d.save()
        f = open('data.csv', 'wb')
        f.write(file.read())
        f.close()
        fn = 'data.csv'
        print("fn",fn)
        global data, path
        path = fn
        print(path)
        data = pd.read_csv(fn)
        datas = data.iloc[:100, :]
        table = datas.to_html()
        return render(request, 'upload.html', {'table': table})
    return render(request, 'upload.html')
# Spliting the dataset.


def splits_dataset(request):
    global data, path, pca
    global sdata
    if request.method == 'POST':
        test_size = int(request.POST['test'])/100
        pca, data = preprocess(test_size, path)
        return render(request, 'split.html', {'res': 'The Data was split successfully'})
    return render(request, 'split.html')

# Training the dataset.


def train(request):
    global data, enc, sdata, clf

    if request.method == 'POST':
        algo = int(request.POST['algo'])
        x_train, x_test, y_train, y_test = data

        if algo == 1:
            clf, acc = DT(x_train, y_train, x_test, y_test)
            return render(request, 'train.html', {'algo': "DT", 'acc': acc})
        elif algo == 2:
            clf, acc = RF(x_train, y_train, x_test, y_test)
            return render(request, 'train.html', {'algo': "RF", 'acc': acc})
        elif algo == 3:
            clf, acc = NB(x_train, y_train, x_test, y_test)
            # Set clf1 to None here or initialize it with some default value
            return render(request, 'train.html', {'algo': "NB", 'acc': acc})
        elif algo == 4:
            clf, acc = knn(x_train, y_train, x_test, y_test)
            # Set clf1 to None here or initialize it with some default value
            return render(request, 'train.html', {'algo': "knn", 'acc': acc})
        elif algo == 5:
            clf, acc = adb(x_train, y_train, x_test, y_test)
            return render(request, 'train.html', {'algo': "adb", 'acc': acc})

    return render(request, 'train.html')

# PREDICTING.


def predictions(request):

    datas = pd.read_csv(path)

    if request.method == 'POST':
        # d = dict(request.POST)
        # del d['csrfmiddlewaretoken']
        # dx = []
        # for x in d.keys():
        #     print(d[x][0])
        #     dx.append(float(d[x][0]))

        # f1 = float(request.POST['fluid_overload'])
        # f2 = float(request.POST['painful_walking'])
        # f3 = float(request.POST['blackheads'])
        # f4 = float(request.POST['small_dents_in_nails'])
        # f5 = float(request.POST['red_sore_around_nose'])
        # f6 = float(request.POST['continuous_sneezing'])

        # list = [dx]
        # print(list)
        itching= str(request.POST['itching'])
        skin_rash= str(request.POST['skin_rash'])
        nodal_skin_eruptions= str(request.POST['nodal_skin_eruptions'])
        continuous_sneezing= str(request.POST['continuous_sneezing'])
        shivering= str(request.POST['shivering'])
        chills= str(request.POST['chills'])
        joint_pain= str(request.POST['joint_pain'])
        stomach_pain= str(request.POST['stomach_pain'])
        acidity= str(request.POST['acidity'])
        ulcers_on_tongue= str(request.POST['ulcers_on_tongue'])
        muscle_wasting= str(request.POST['muscle_wasting'])
        vomiting= str(request.POST['vomiting'])
        burning_micturition= str(request.POST['burning_micturition'])
        spotting_urination= str(request.POST['spotting_urination'])
        fatigue= str(request.POST['fatigue'])
        weight_gain= str(request.POST['weight_gain'])
        anxiety= str(request.POST['anxiety'])
        cold_hands_and_feets= str(request.POST['cold_hands_and_feets'])
                     
        list=[[itching,skin_rash,nodal_skin_eruptions,continuous_sneezing,shivering,
              chills,joint_pain,stomach_pain,acidity,ulcers_on_tongue,muscle_wasting,
              vomiting,burning_micturition,spotting_urination,fatigue,weight_gain,
              anxiety,cold_hands_and_feets]]
        
        res = clf.predict(list)
        print(res)
        return render(request, 'predictions.html', {'res': res, })

    return render(request, 'predictions.html')



# views.py

# from django.shortcuts import render
# import pandas as pd
# from machine_learning.preprocessing import preprocess
# from machine_learning.train import DT, RF, NB, knn, adb
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split

# # Initialize global variables
# clf = None
# clf1 = None
# data = None
# path = None

# def index(request):
#     return render(request, 'index.html')

# def upload_dataset(request):
#     if request.method == 'POST':
#         file = request.FILES['file']
#         f = open('data.csv', 'wb')
#         f.write(file.read())
#         f.close()
#         fn = 'data.csv'
#         global data, path
#         path = fn
#         data = pd.read_csv(fn)
#         datas = data.iloc[:100, :]
#         table = datas.to_html()
#         return render(request, 'upload.html', {'table': table})
#     return render(request, 'upload.html')

# def splits_dataset(request):
#     global data, path
#     if request.method == 'POST':
#         test_size = int(request.POST['test']) / 100
#         pca, data = preprocess(test_size, path)
#         return render(request, 'split.html', {'res': 'The Data was split successfully'})
#     return render(request, 'split.html')

# def train(request):
#     global data, clf, clf1

#     if request.method == 'POST':
#         algo = int(request.POST['algo'])
#         x_train, x_test, y_train, y_test = data

#         if algo == 1:
#             clf, acc = DT(x_train, y_train, x_test, y_test)
#             return render(request, 'train.html', {'algo': "DT", 'acc': acc})
#         elif algo == 2:
#             clf1, acc = RF(x_train, y_train, x_test, y_test)
#             return render(request, 'train.html', {'algo': "RF", 'acc': acc})
#         elif algo == 3:
#             clf, acc = NB(x_train, y_train, x_test, y_test)
#             return render(request, 'train.html', {'algo': "NB", 'acc': acc})
#         elif algo == 4:
#             clf, acc = knn(x_train, y_train, x_test, y_test)
#             return render(request, 'train.html', {'algo': "knn", 'acc': acc})
#         elif algo == 5:
#             clf, acc = adb(x_train, y_train, x_test, y_test)
#             return render(request, 'train.html', {'algo': "adb", 'acc': acc})

#     return render(request, 'train.html')

# def predictions(request):
#     global path, clf1

#     datas = pd.read_csv(path)

#     if request.method == 'POST':
#         # ... (your input processing code)
#         itching= str(request.POST['itching'])
#         skin_rash= str(request.POST['skin_rash'])
#         nodal_skin_eruptions= str(request.POST['nodal_skin_eruptions'])
#         continuous_sneezing= str(request.POST['continuous_sneezing'])
#         shivering= str(request.POST['shivering'])
#         chills= str(request.POST['chills'])
#         joint_pain= str(request.POST['joint_pain'])
#         stomach_pain= str(request.POST['stomach_pain'])
#         acidity= str(request.POST['acidity'])
#         ulcers_on_tongue= str(request.POST['ulcers_on_tongue'])
#         muscle_wasting= str(request.POST['muscle_wasting'])
#         vomiting= str(request.POST['vomiting'])
#         burning_micturition= str(request.POST['burning_micturition'])
#         spotting_urination= str(request.POST['spotting_urination'])
#         fatigue= str(request.POST['fatigue'])
#         weight_gain= str(request.POST['weight_gain'])
#         anxiety= str(request.POST['anxiety'])
#         cold_hands_and_feets= str(request.POST['cold_hands_and_feets'])
#         list = [[itching, skin_rash, nodal_skin_eruptions, continuous_sneezing, shivering,
#                  chills, joint_pain, stomach_pain, acidity, ulcers_on_tongue, muscle_wasting,
#                  vomiting, burning_micturition, spotting_urination, fatigue, weight_gain,
#                  anxiety, cold_hands_and_feets]]

#         if clf1 is not None:
#             res = clf1.predict(list)
#             print(res)
#             return render(request, 'predictions.html', {'res': res})
#         else:
#             return render(request, 'predictions.html', {'error': 'Model not trained'})

#     return render(request, 'predictions.html')
