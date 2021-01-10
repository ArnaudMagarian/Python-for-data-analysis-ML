from django.shortcuts import render

def home(request, *args, **kwargs):
	return render(request, "home.html", {})

def dataset_presentation(request, *args, **kwargs):
	return render(request, "dataset_presentation/dataset_presentation.html",{})

def introduction(request, *args, **kwargs):
	return render(request, "dataset_presentation/introduction.html",{})

def features(request, *args, **kwargs):
	return render(request, "dataset_presentation/features.html",{})

def label(request, *args, **kwargs):
	return render(request, "dataset_presentation/label.html",{})

def analysis(request, *args, **kwargs):
	return render(request, "analysis/analysis.html",{})



#importing the necessary librairies

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib. patches as mpatches
import seaborn as sns

#importing the databases from sqlite3, the db by default used with django.
from .models import dataset
from .models import DataSignification
#to import the data, I have used DB Browser for SQLite and have imported using the CSV tool implemented in the app.

qs = dataset.objects.all().values()
data = pd.DataFrame(qs)

data = data.drop(['id'], axis=1)

qs = DataSignification.objects.all().values()
DataSignification = pd.DataFrame(qs)

myBlueColors1 = ["#98E3FE",
                 "#57D1FF",
                 "#4CB7DF",
                 "#419DBF",
                 "#36839F",
                 "#2B6980",
                 "#214E60",
                 "#163440",
                 "#0B1A20"
                ]

myColor1 = "#36839F"
myColor2 = "#FFA3D2"
myColor3 = "#9F366C"

myPinkColors = ["#FFA3D2",
                "#FF57AD",
                "#DF4C97",
                "#BF4182",
                "#9F366C",
                "#802B56",
                "#602141",
                "#40162B",
                "#200B16"
                 ]

myBlueColors2 = ["#C3EEFF",
                 "#8EDFFF",
                 "#61D2FF",
                 "#37C6FF",
                 "#00B6FF",
                 "#0096D2",
                 "#0078A8"
                ]

def head(request, *args, **kwargs):

	context = {
		'df' : data.head(100).to_html()
	}
	return render(request, "analysis/head.html",context)

def describe(request, *args, **kwargs):

	context = {
		'df' : data.describe().to_html()
	}
	return render(request, "analysis/describe.html",context)

def data_s(request, *args, **kwargs):

	context = {
		'ds' : DataSignification.to_html()
	}
	return render(request, "analysis/data_s.html",context)


def visualization(request, *args, **kwargs):
    return render(request, "analysis/visualization.html",{})

obesity = data
import base64
from io import BytesIO

def get_graph():
	buffer = BytesIO()
	plt.savefig(buffer, format='png')
	buffer.seek(0)
	image_png = buffer.getvalue()
	graph = base64.b64encode(image_png)
	graph = graph.decode('utf-8')
	buffer.close()
	return graph

def pie_plots(request, *args, **kwargs):
    mask_1 = obesity[(obesity['NObeyesdad']=='Insufficient_Weight')]
    mask_2 = obesity[(obesity['NObeyesdad']=='Normal_Weight')]
    mask_3 = obesity[(obesity['NObeyesdad']=='Overweight_Level_I') | (obesity['NObeyesdad'] == 'Overweight_Level_II')]
    mask_4 = obesity[(obesity['NObeyesdad']=='Obesity_Type_I') | (obesity['NObeyesdad'] == 'Obesity_Type_II') | (obesity['NObeyesdad'] == 'Obesity_Type_III')]

    fig = plt.figure(figsize=(20,30))

    counter = 0

    notDisplay = ["Age", "Height", "Weight", "NObeyesdad"]
    masks = ['Insufficient Weight', 'Normal weight', 'Overweight level I or II', 'Obesity type I, II or III']

    for variable, definition in zip(DataSignification['Variable'], DataSignification['Signification']):
        if variable not in notDisplay:
            for i in range(1,5):
                counter = counter + 1
                ax = plt.subplot(13, 4, counter)
                myTitle = variable + ' ratio' + '\n' + masks[i-1] + "\n(def : " + definition + ")" 
                ax.title.set_text(myTitle)
                
                mask = locals()["mask_"+str(i)]
                variableCount = mask[variable].value_counts()
                variableCount.plot.pie(subplots = True,
                                colors = myBlueColors1,
                                autopct = '%1.1f%%'
                                )
                plt.axis("equal")
                plt.ylabel('')
                plt.subplots_adjust(top = 2)

    graph = get_graph()
    return render(request, "analysis/pie_plots.html",{'chart': graph})


def box_plots(request, *args, **kwargs):
    plt.clf()
    plt.close()
    fig = plt.figure(figsize=(20,5))
    fig.suptitle("Physological data.", fontsize=20)

    counter = 0

    Display = ["Age", "Height", "Weight"]


    for variable, definition in zip(DataSignification['Variable'], DataSignification['Signification']):
        if variable in Display:
            counter = counter + 1
            ax = plt.subplot(1,3, counter)
            
            myTitle = variable + "  values in the dataset. (def : " + definition + ")"
            ax.title.set_text(myTitle)
            ax.set_ylim([obesity[variable].min()*0.96, obesity[variable].max()*1.02])
            
            color = {'boxes': myColor1, 'whiskers': myColor2,
                    'medians': myColor3, 'caps': myColor1}
            obesity[variable].plot.box(subplots = True,
                                    color = color,
                                    )
            
            plt.ylabel(definition)
        
    graph = get_graph()
    return render(request, "analysis/box_plots.html",{'chart': graph})

from scipy.stats import skew
def skewness_test (data):
    skewness = skew(data)
    display = ''

    if skewness > 0:
        display = "Data is Positively or Right skewed. Skewness value: " + str(skewness)
    elif skewness < 0:
        display = "Data is Negatively or Left skewed. Skewness value: " + str(skewness)
    else:
        display = "Data is Normally distributed. Skewness value: " + str(skewness)

    return display

def skewness(request, *args, **kwargs):
    plt.clf()
    plt.close()
    obesity['Age'].hist(bins=50, color = myBlueColors1[3])
    plt.xlabel('Age')
    plt.ylabel('Count')
    graph1 = get_graph()
    sk1 = skewness_test(obesity['Age'])

    plt.clf()
    plt.close()
    obesity['Weight'].hist(bins=50, color = myBlueColors1[3])
    plt.xlabel('Weight')
    plt.ylabel('Count')
    graph2 = get_graph()
    sk2 = skewness_test(obesity['Weight'])

    plt.clf()
    plt.close()
    obesity['Height'].hist(bins=50, color = myBlueColors1[3])
    plt.xlabel('Height')
    plt.ylabel('Count')
    graph3 = get_graph()
    sk3 = skewness_test(obesity['Height'])

    return render(request, "analysis/skewness.html",{'chart1': graph1, 'chart2': graph2, 'chart3': graph3, 'sk1' : sk1, 'sk2' : sk2, 'sk3' : sk3})

def scatter_plots(request, *args, **kwargs):
    plt.clf()
    plt.close()
    fig, axes = plt.subplots(24,1,figsize=(15,200), constrained_layout = True)
    fig.suptitle("Height against the weight with the weight status hilighted for females and males.\nColors for a variable, Size for the weight status", fontsize=15)

    settings_f = ("Female", myColor2)
    settings_h = ("Male", myColor1)

    size_order = ['Obesity_Type_III', 'Obesity_Type_II', 'Obesity_Type_I',
                'Overweight_Level_II', 'Overweight_Level_I', 
                'Normal_Weight', 'Insufficient_Weight'
                ]

    count = 0
    notDisplay = ['Height', 'Age', 'Weight', 'Gender', 'NObeyesdad']

    for variable, definition in zip(DataSignification['Variable'], DataSignification['Signification']):
        for sexe, color in [settings_f, settings_h]:
            if variable not in notDisplay:
                ax = axes[count]
                count = count + 1
                mask = obesity['Gender']==sexe
                data = obesity[mask]

                avg_weight = data.groupby('Height')['Weight'].mean()
                std_weight = data.groupby('Height')['Weight'].std()
                inf = avg_weight - std_weight
                sup = avg_weight + std_weight

                sns.scatterplot(data = data, x = 'Height', y = 'Weight',
                                hue = variable, 
                                size = 'NObeyesdad', sizes = (40, 400), size_order = size_order, alpha = .5,
                                palette = 'cool', ax = ax
                            )
                sns.lineplot(data=avg_weight, color=color, ax=ax)
                ax.fill_between(avg_weight.index, inf, sup, color=color, alpha=0.2)
                ax.set_title(sexe + '\nVariable: ' + variable + " (def : " + definition + ")")

    graph = get_graph()
    return render(request, "analysis/scatter_plots.html",{'chart': graph})

def heatmap(request, *args, **kwargs):
    plt.clf()
    plt.close()
    sns.heatmap(obesity.corr(), annot=True)
    
    graph = get_graph()
    return render(request, "analysis/heatmap.html",{'chart': graph})




def ml(request, *args, **kwargs):
	return render(request, "ML/ml.html",{})

def preprocessing(request, *args, **kwargs):
	return render(request, "ML/preprocessing.html",{})

Xraw = obesity.drop('NObeyesdad', axis = 1)
yRaw = obesity['NObeyesdad']

def x_and_y(request, *args, **kwargs):
    X = Xraw.head(20).to_html()
    y = yRaw.to_frame()
    y = y.head(20).to_html()
    return render(request, "ML/x_and_y.html",{'X':X, 'y':y})

#---Transforming the y array. LabelEncoder is used to encode only one column
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(yRaw)
y = pd.DataFrame(y, columns=["NObeyesdad"])

def encoding_y(request, *args, **kwargs):
    
	return render(request, "ML/encoding_y.html",{'y':y.head(20).to_html()})

X = pd.get_dummies(Xraw, columns = ['Gender',
                    'family_history_with_overweight',
                    'FAVC',
                    'CAEC',
                    'SMOKE',
                    'SCC',
                    'CALC',
                    'MTRANS'
                    ])

def encoding_x(request, *args, **kwargs):
	return render(request, "ML/encoding_x.html",{'X':X.head(20).to_html()})


#---Scaler to normalize quantitative data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X[['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']] = scaler.fit_transform(X[['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']])

def normalizing_x(request, *args, **kwargs):
	return render(request, "ML/normalizing_x.html",{'X':X.head(20).to_html()})

#---Spliting the dataset into a training set and a testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)

def spliting(request, *args, **kwargs):
    train = 'Train set:', X_train.shape
    test = 'Test set:', X_test.shape
    return render(request, "ML/spliting.html",{'train': train, 'test': test})

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def machine_learning(request, *args, **kwargs):
	return render(request, "ML/machine_learning.html",{})

#---Machine learning model --> LinearSVC
from sklearn.svm import LinearSVC

#---Machine learning --> testing all the parameters of one model
from sklearn.model_selection import GridSearchCV

#--Fitting LinearSVC with GridSearch


#---Metric to analyse performance --> confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

#---Metric --> learning curve of our model
from sklearn.model_selection import learning_curve
import json 

def linear_svc(request, *args, **kwargs):
    #--The standard way
    model1 = LinearSVC()
    model1.fit(X_train, y_train.values.ravel())
    var1 = 'Linear SVC standard train score:' + str(model1.score(X_train, y_train))
    var2 = 'Linear SVC standard test score:' + str(model1.score(X_test, y_test))

    #--GridSearchCV
    
    param_grid1 = {'loss': ['hinge', 'squared_hinge'],
              'tol': [1e-4, 1e-5, 1e-6],
              'multi_class': ['ovr', 'crammer_singer'],
              'max_iter': [100000]
             }
    grid1 = GridSearchCV(LinearSVC(), param_grid1, cv=5)
    grid1.fit(X_train, y_train.values.ravel())
    var3 = 'Linear SVC GridSearch best train score obtained:' + str(grid1.best_score_)
    var4 = 'Linear SVC GridSearch best parameters:'
    var5 = grid1.best_params_

    #--Testing
    model1 = grid1.best_estimator_
    score1 = model1.score(X_test, y_test.values.ravel())
    var6 = 'Linear SVC GridSearch score test score:' + str(score1)

    #--CF matrix
    tab = []
    counter = 0
    for state in encoder.classes_:
        string = counter, '-->', state
        tab.append(string)
        counter+=1
    plt.clf()
    plt.close()
    plot_confusion_matrix(model1, X_test, y_test)
    graph1 = get_graph()

    #--Learning Curves
    plt.clf()
    plt.close()
    N1, train_score1, val_score1 = learning_curve(model1, X_train, y_train.values.ravel(), train_sizes = np.linspace(0.1, 1.0, 10), cv=5)
    plt.plot(N1, train_score1.mean(axis=1), label = 'train')
    plt.plot(N1, val_score1.mean(axis=1), label = 'validation')
    plt.xlabel('train_sizes')
    plt.legend()
    graph2 = get_graph()
    return render(request, "ML/linear_svc.html",{'var1':var1, 'var2':var2, 'var3':var3, 'var4':var4, 'var5': var5, 'var6': var6, 'tab': tab, 'chart1': graph1, 'N1':N1, 'chart2': graph2})


#---Machine learning model --> KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

def kneighborsclassifier(request, *args, **kwargs):
    #--The standard way
    model2 = KNeighborsClassifier()
    model2.fit(X_train, y_train.values.ravel())
    var1 = 'KNeighborsClassifier train score:' + str(model2.score(X_train, y_train))
    var2 = 'KNeighborsClassifier test score:' + str(model2.score(X_test, y_test))

    #--GridSearchCV
    param_grid2 = {'n_neighbors': np.arange(1,20),
                'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                'weights': ['uniform', 'distance'],
                'metric':  ['euclidean', 'manhattan']
                }
    grid2 = GridSearchCV(KNeighborsClassifier(), param_grid2, cv=5)
    grid2.fit(X_train, y_train.values.ravel())
    var3 = 'KNeighborsClassifier GridSearch best train score obtained:' + str(grid2.best_score_)
    var4 = 'KNeighborsClassifier GridSearch best parameters:'
    var5 = grid2.best_params_
    #--Testing
    model2 = grid2.best_estimator_
    score2 = model2.score(X_test, y_test.values.ravel())
    var6 = 'KNeighborsClassifier GridSearch score test score:' + str(score2)

    #--CF matrix
    tab = []
    counter = 0
    for state in encoder.classes_:
        string = counter, '-->', state
        tab.append(string)
        counter+=1
    plt.clf()
    plt.close()
    plot_confusion_matrix(model2, X_test, y_test)
    graph1 = get_graph()

    #--Learning Curves
    plt.clf()
    plt.close()
    N2, train_score2, val_score2 = learning_curve(model2, X_train, y_train.values.ravel(), train_sizes = np.linspace(0.1, 1.0, 10), cv=5)
    plt.plot(N2, train_score2.mean(axis=1), label = 'train')
    plt.plot(N2, val_score2.mean(axis=1), label = 'validation')
    plt.xlabel('train_sizes')
    plt.legend()
    graph2 = get_graph()
    return render(request, "ML/kneighborsclassifier.html",{'var1':var1, 'var2':var2, 'var3':var3, 'var4':var4, 'var5': var5, 'var6': var6, 'tab': tab, 'chart1': graph1, 'N2':N2, 'chart2': graph2})


#---Machine learning model --> SVC
from sklearn.svm import SVC

def svc(request, *args, **kwargs):
    #--The standard way
    model3 = SVC()
    model3.fit(X_train, y_train.values.ravel())
    var1 = 'SVC train score:' + str(model3.score(X_train, y_train))
    var2 = 'SVC test score:' + str(model3.score(X_test, y_test))

    #--GridSearchCV
    param_grid3 = {'C': [0.1, 1, 10, 100, 1000],
                'shrinking': [True, False],
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'kernel':  ['poly', 'rbf', 'sigmoid'],
                'decision_function_shape': ['ovo', 'ovr'],
                'probability': [True]
                }
    grid3 = GridSearchCV(SVC(), param_grid3, cv=5)
    grid3.fit(X_train, y_train.values.ravel())
    var3 = 'SVC GridSearch best train score obtained:' + str(grid3.best_score_)
    var4 = 'SVC GridSearch best parameters:'
    var5 = grid3.best_params_

    #--Testing
    model3 = grid3.best_estimator_
    score3 = model3.score(X_test, y_test.values.ravel())
    var6 = 'SVC GridSearch score test score:' + str(score3)

    #--CF matrix
    tab = []
    counter = 0
    for state in encoder.classes_:
        string = counter, '-->', state
        tab.append(string)
        counter+=1
    plt.clf()
    plt.close()
    plot_confusion_matrix(model3, X_test, y_test)
    graph1 = get_graph()

    #--Learning Curves
    plt.clf()
    plt.close()
    N3, train_score3, val_score3 = learning_curve(model3, X_train, y_train.values.ravel(), train_sizes = np.linspace(0.1, 1.0, 10), cv=5)
    plt.plot(N3, train_score3.mean(axis=1), label = 'train')
    plt.plot(N3, val_score3.mean(axis=1), label = 'validation')
    plt.xlabel('train_sizes')
    plt.legend()
    graph2 = get_graph()
    return render(request, "ML/svc.html",{'var1':var1, 'var2':var2, 'var3':var3, 'var4':var4, 'var5': var5, 'var6': var6, 'tab': tab, 'chart1': graph1, 'N3':N3, 'chart2': graph2})
