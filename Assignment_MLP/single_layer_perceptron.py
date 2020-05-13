import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(123)

#b i - reading from the dataset
data = pd.read_csv('dataset.csv')
data.head()

data['diagnosis'] = data['diagnosis'].map({'M':0,'B':1})
data.head()

#b ii - splitting the data to train and test data
X = data.iloc[:,2:].values
Y = data.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y)
print(X)

print("Target labels : \n")
print(Y)

print(f'Shape X_train: {X_train.shape}')
print(f'Shape y_train: {y_train.shape})')
print(f'Shape X_test: {X_test.shape}')
print(f'Shape y_test: {y_test.shape}')
#b iii
#Initialising learning rate, epochs
class Perceptron(object):
    def __init__(self,learning_rate,epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        
#initialising weights, delta_w is the weight vector       
    def fit(self,X,y):
        self.weights = np.zeros(1+X.shape[1])
        self.errors = []
        for i in range(self.epochs):
            error = 0
            for xi, output in zip(X, Y):
                delta_w = self.learning_rate * (output - self.predict(xi))
                self.weights[1:] += delta_w * xi
                self.weights[0] += delta_w
                error += int(delta_w != 0.0)
            self.errors.append(error)
        return self.weights
    
    def inputs(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]
    
    #b iv - activation function
    def predict(self, X):
        return np.where(self.inputs(X) >= 0.0, 1, 0)
perceptron = Perceptron(0.02,100)
#b v-training the model
trained_weights = perceptron.fit(X_train,y_train)
#b vi - printing the weights and the hyper parameters
print("Learned weights \n: ",trained_weights)
print("\nLearning rate : ",perceptron.learning_rate)
print("\nEpochs : ",perceptron.epochs)
#b vii - predicting the outputs of train and test data
y_predicted_train = perceptron.predict(X_train)
y_predicted_test = perceptron.predict(X_test)

print(y_predicted_train)
#Visualisation
plt.plot(range(1, len(perceptron.errors) + 1), perceptron.errors, marker='o')
plt.xlabel('Epochs[Number of iterations through the dataset]')
plt.ylabel('Number of misclassifications')
plt.show()

#b viii - printing confusion matrix
def performance_measure(y_test, y_predicted_test):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for i in range(len(y_predicted_test)): 
        if y_test[i]==y_predicted_test[i]==1:
            true_positives += 1
        if y_predicted_test[i]==1 and y_test[i]!=y_predicted_test[i]:
            false_positives+= 1
        if y_test[i]==y_predicted_test[i]==0:
            true_negatives += 1
        if y_predicted_test[i]==0 and y_test[i]!=y_predicted_test[i]:
            false_negatives += 1
           
    
    return(true_negatives, false_positives, false_negatives,true_positives)
print("Confusion matrix : \n")
print(np.reshape(np.array(['True negatives','False positives','False negatives','True positives']),(-1,2)))

print("\n\nConfusion matrix (Train data) : \n")
print ("\n",np.reshape(np.array(performance_measure(y_train,y_predicted_train)),(-1,2)))

print("\n\nConfusion matrix (Test data) : \n")
print ("\n",np.reshape(np.array(performance_measure(y_test,y_predicted_test)),(-1,2)))

def trainConfusionMatrix(y_test,y_predicted_test):

    confusion_data = {'Actual Label':np.array(y_test) , 'Predicted Label':  np.array(y_predicted_test)}
    df = pd.DataFrame(confusion_data, columns=['Actual Label','Predicted Label'])
    confusion_matrix = pd.crosstab(df['Actual Label'], df['Predicted Label'], rownames=['Actual'], colnames=['Predicted'])
    sns.heatmap(confusion_matrix, annot=True)
print("                  Train confusion matrix : \n")


print("             Test confusion matrix : \n")
trainConfusionMatrix(y_test, y_predicted_test)

#b viii - printing performance measures
def calculate(matrix):
    TP = matrix[0,0]
    FN = matrix[0,1]
    FP = matrix[1,0]
    TN = matrix[1,1]
    
    accuracy = (TP+TN)/(TP+FN+FP+TN)
    error_rate = 1 - accuracy
    precision =(TP) /(TP+FP)
    recall = TP/(TP+TN)

    print(f'Accuracy: {accuracy * 100} %')
    print(f'Error_rate: {error_rate * 100} %')
    print(f'Precision: {precision * 100} %')
    print(f'Recall: {recall * 100} %')
    
    
print("Test data - performance measures : \n")
calculate(np.reshape(np.array(performance_measure(y_test,y_predicted_test)),(-1,2)))

print("\n\nTrain data - performance measures : \n")
calculate(np.reshape(np.array(performance_measure(y_train,y_predicted_train)),(-1,2)))
trainConfusionMatrix(y_train, y_predicted_train)
