#fazendo modelo supervisionado
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import f1_score,accuracy_score

iris = load_iris()
X = iris.data
Y = iris.target

X_train,X_test,Y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

#Treinamento QDA
qda_model = QuadraticDiscriminantAnalysis()
qda_model.fit(X_train,Y_train)

#Amostra avaliacao
y_pred = qda_model.predict(X_test)

f1= f1_score(y_test,y_pred,average='weighted')

accuracy = accuracy_score(y_test,y_pred)
print("F1-score:",f1)
print("Acurácia:",accuracy)

#Análise Gráfica para conferir a acertividade
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.scatter(X_test[:,0],X_test[:,1],c=y_test,cmap='viridis',marker='o',label='Real')
plt.scatter(X_test[:,0],X_test[:,1],c=y_pred,cmap='viridis',marker='x',label='Previsto')

plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Classificacao Real vs. Prevista (QDA)')
plt.legend()
plt.colorbar(label='Classe')
plt.show()
