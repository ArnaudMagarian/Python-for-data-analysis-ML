MAGARIAN ARNAUD
PYTHON FOR DATA ANALYSIS
ESILV A5 - 2021

Final project, do machine learning on the following dataset : https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+

THE DATASET :
	Estimation of obesity levels based on eating habits and physical condition
	Number of instances: 2111
	Number of Attributes: 17
		Number of features: 16
		Number of target labels: 1
	Missing values: None
	This dataset include data for the estimation of obesity levels in individuals from the countries of Mexico, Peru and Colombia, based on their eating habits and physical condition. The data contains 17 attributes and 2111 records, the records are labeled with the class variable NObesity (Obesity Level), that allows classification of the data using the values of Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II and Obesity Type III. 77% of the data was generated synthetically using the Weka tool and the SMOTE filter, 23% of the data was collected directly from users through a web platform.
	Attributes information : https://www.sciencedirect.com/science/article/pii/S2352340919306985?via%3Dihub

THE GOAL:
	The goal of this dataset is to be able to predict a person’s weight status using the 16 features on the right, Nobeyesdad being the weight status.
	The dataset is a mix of qualitative and quantitative features.
	Qualitative :
		Gender
		family_history_with_overweight
		FAVC
		CAEC
		SMOKE
		SCC
		CALC
		MTRANS
	Quantitative :
		Age
		Weight
		Height
		FCVC
		NCP
		CH2O
		FAF
		TUE

DATA REPARTITION & INTERPRETATION:
	To better understand and interpret the data, I displayed some graphs (pie plots, histograms, scatter plots, box plots).
	The histograms are dedicated to identify if some features are skewed or not and need normalization.

PREPROCESSING:
	SPLITTING:
		Splitting the dataset to create X (features) and y (target label)

	ENCODING y:
		Transforming the qualitative (text) data into numbers so that the ML algorithm can interpret the data.

	ENCODING X:
		Transforming the qualitative (text) data into numbers so that the ML algorithm can interpret the data. We are doing that on specific features as some are not qualitative.

	NORMALIZING X:
		Transforming the quantitative data so that they all have the same scale and rank as some algorithms tend to take into account the features’ order.


MACHINE LEARNING ALGORITHMS:
	LINEAR SVC:
		fitting the standard way: model.fit
		fitting using GridSearchCV
		testing both models
		ploting the confusion matrix
		ploting the learning curves
	KNEIHBORS CLASSIFIER:
		fitting the standard way: model.fit
		fitting using GridSearchCV
		testing both models
		ploting the confusion matrix
		ploting the learning curves
	SVC:
		fitting the standard way: model.fit
		fitting using GridSearchCV
		testing both models
		ploting the confusion matrix
		ploting the learning curves

API:
	The API was entirely developped usign django and SQlite3.