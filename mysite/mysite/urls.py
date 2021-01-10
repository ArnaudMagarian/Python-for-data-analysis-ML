from django.contrib import admin
from django.urls import path, include
from pages import views

urlpatterns = [
    path('admin/', admin.site.urls),
	path('', views.home, name='home'),
    path('dataset_presentation/', views.dataset_presentation, name='dataset_presentation'),
    path('dataset_presentation/introduction', views.introduction, name='introduction'),
    path('dataset_presentation/features', views.features, name='features'),
    path('dataset_presentation/label', views.label, name='label'),
    path('analysis/', views.analysis, name='analysis'),
    path('analysis/head', views.head, name='head'),
    path('analysis/describe', views.describe, name='describe'),
    path('analysis/data_s', views.data_s, name='data_s'),
    path('analysis/visualization', views.visualization, name='visualization'),
    path('analysis/visualization/pie_plots', views.pie_plots, name='pie_plots'),
    path('analysis/visualization/box_plots', views.box_plots, name='box_plots'),
    path('analysis/visualization/skewness', views.skewness, name='skewness'),
    path('analysis/visualization/scatter_plots', views.scatter_plots, name='scatter_plots'),
    path('analysis/visualization/heatmap', views.heatmap, name='heatmap'),
    path('ml', views.ml, name='ml'),
    path('ml/preprocessing', views.preprocessing, name='preprocessing'),
    path('ml/preprocessing/x_and_y', views.x_and_y, name='x_and_y'),
    path('ml/preprocessing/encoding_y', views.encoding_y, name='encoding_y'),
    path('ml/preprocessing/encoding_x', views.encoding_x, name='encoding_x'),
    path('ml/preprocessing/normalizing_x', views.normalizing_x, name='normalizing_x'),
    path('ml/preprocessing/spliting', views.spliting, name='spliting'),
    path('ml/machine_learning', views.machine_learning, name='machine_learning'),
    path('ml/machine_learning/linear_svc', views.linear_svc, name='linear_svc'),
    path('ml/machine_learning/kneighborsclassifier', views.kneighborsclassifier, name='kneighborsclassifier'),
    path('ml/machine_learning/svc', views.svc, name='svc')
    
]
