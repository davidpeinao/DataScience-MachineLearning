# Data Science and Machine Learning projects
Business Intelligence / Data Science assignments

In the first project [P1](P1.pdf) a series of supervised learning classification algorithms for the resolution, visualization and experimental analysis of classification and basic preprocessing problems will be studied.

To make this possible I will use KNIME, and also have the help of the Weka, which will be used in certain algorithms or JavaScript views, used for data visualization.

The method followed in this project for the correct validation will be cross-validation of 5 partitions with a fixed seed. To analyze the results I will use: error tables, confusion matrices, ROC curves, measures such as TPR, TNR, FPR, FNR, AUC, F1-Value and G-mean.

In the second project (**P2**) unsupervised learning techniques will be used to analyze a set of data and extract information about it. Said data set will be the one provided by the INE(Spanish National Institute of Statistics).

The set, processed from the original source, consists of 142 variables on sex, age, nationality, studies, employment status, migration and mobility, family status, etc. These data are related to the province of Granada, with a total of 83,499 cases.

The objective is to take certain groups of people and apply clustering algorithms to analyze the results obtained and interpret them to explain different profiles or groups found.

In this case we will focus on:
* Women between 20 and 50 years old
* Born abroad
* People belonging to large families (more than 2 children)

For the analysis of the results, various graphs (dendrograms, heat maps ...) will be used, as well as performance metrics such as Silhouette and the Calinski-Harabaz index.

The algorithms used for each use case will be:
* K-Means
* MiniBatchK-Means
* MeanShift
* AffinityPropagation
* DBSCAN
* Ward (agglomerative clustering)



***

The third project (**P3**) is a [competition on DrivenData](https://www.drivendata.org/competitions/7/pumpit-up-data-mining-the-water-table/page/23/).

In this assignment, the data of Taarifa and the Ministry of Water of Tanzania will be analyzed, with the aim of predicting which water extraction pumps work, which ones need some repairs and which do not work.

The data provided to us on the website of the subject contains 59400 instances, each instance consists of its ID, 39 variables + 1 objective variable (the class). 

The training data given presents several problems, variables dependent on others or straight useless, class imbalance and a multitude of missing values. To deal with these problems a meticulous preparation of the data has been made.

Testing several models, the one that gives better results is the RandomForestClassifier. The model is trained using Stratified K-Folds cross-validation
