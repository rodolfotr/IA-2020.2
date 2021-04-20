import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Função do cáculo da sigmóide
def sigmoid(x):
	return 1/(1+np.exp(-x))

DataSet=pd.read_csv('arruela_.csv')
DataSet.drop(['Hora','Tamanho','Referencia'],axis=1,inplace=True)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
DataScaled=scaler.fit_transform(DataSet)
DataSetScaled=pd.DataFrame(np.array(DataScaled),columns = ['NumAmostra', 'Area', 'Delta', 'Output1','Output2'])

DataSetScaled.head()

X = DataSetScaled.drop(['Output1', 'Output2'],axis=1)
y = DataSet[['Output1','Output2']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=3)

#print(y_test)
#print(X_train)

#Normalização
#from sklearn import preprocessing
#min_max_scaler = preprocessing.MinMaxScaler()

#x_scaled = min_max_scaler.fit_transform(X_train )
#X_train = pd.DataFrame(x_scaled)

#x_scaled = min_max_scaler.fit_transform(X_test)
#X_test = pd.DataFrame(x_scaled)

#print(X_train)


#---------------------------------------------------------------------------------------------
#Tamanho do DataSet de Treinamento
n_records, n_features = X_train.shape

#Arquitetura da MPL
N_input = 3
N_hidden = 4
N_output = 2
learnrate = 0.2

#Pesos da Camada Oculta (Inicialização Aleatória)
weights_input_hidden = np.random.normal(0, scale=0.1, size=(N_input, N_hidden))

#Pesos da Camada de Saída (Inicialização Aleatória)
weights_hidden_output = np.random.normal(0, scale=0.1, size=(N_hidden, N_output))


epochs = 20000
last_loss=None
EvolucaoError=[]
IndiceError=[]

for e in range(epochs):
	delta_w_i_h = np.zeros(weights_input_hidden.shape)
	delta_w_h_o = np.zeros(weights_hidden_output.shape)
	for xi, yi in zip(X_train.values, y_train.values):
		
# Forward Pass
		#Camada oculta
		#Calcule a combinação linear de entradas e pesos sinápticos
		hidden_layer_input = np.dot(xi, weights_input_hidden)
		#Aplicado a função de ativação
		hidden_layer_output = sigmoid(hidden_layer_input)
	
		#Camada de Saída
		#Calcule a combinação linear de entradas e pesos sinápticos
		output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)

		#Aplicado a função de ativação 
		output = sigmoid(output_layer_in)
		#print('As saídas da rede são',output)
#-------------------------------------------    
	
# Backward Pass
		## TODO: Cálculo do Erro
		error = yi - output
	
		# TODO: Calcule o termo de erro de saída (Gradiente da Camada de Saída)
		output_error_term = error * output * (1 - output)

		# TODO: Calcule a contribuição da camada oculta para o erro
		hidden_error = np.dot(weights_hidden_output,output_error_term)
	
		# TODO: Calcule o termo de erro da camada oculta (Gradiente da Camada Oculta)
		hidden_error_term = hidden_error * hidden_layer_output * (1 - hidden_layer_output)
	
		# TODO: Calcule a variação do peso da camada de saída
		delta_w_h_o += output_error_term*hidden_layer_output[:, None]

		# TODO: Calcule a variação do peso da camada oculta
		delta_w_i_h += hidden_error_term * xi[:, None]
		
	#Atualização dos pesos na época em questão
	weights_input_hidden += learnrate * delta_w_i_h / n_records
	weights_hidden_output += learnrate * delta_w_h_o / n_records
	

	# Imprimir o erro quadrático médio no conjunto de treinamento
	
	if  e % (epochs / 20) == 0:
		hidden_output = sigmoid(np.dot(xi, weights_input_hidden))
		out = sigmoid(np.dot(hidden_output,
							weights_hidden_output))
		loss = np.mean((out - yi) ** 2)

		if last_loss and last_loss < loss:
			print("Erro quadrático no treinamento",e,": ", loss, " Atenção: O erro está aumentando")
		else:
			print("Erro quadrático no treinamento",e,": ", loss)
		last_loss = loss
		
		EvolucaoError.append(loss)
		IndiceError.append(e)



# Calcule a precisão dos dados de teste
n_records, n_features = X_test.shape
predictions=0

for xi, yi in zip(X_test.values, y_test.values):

# Forward Pass
		#Camada oculta
		#Calcule a combinação linear de entradas e pesos sinápticos
		hidden_layer_input = np.dot(xi, weights_input_hidden)
		#Aplicado a função de ativação
		hidden_layer_output = sigmoid(hidden_layer_input)
	
		#Camada de Saída
		#Calcule a combinação linear de entradas e pesos sinápticos
		output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)

		#Aplicado a função de ativação 
		output = sigmoid(output_layer_in)

#-------------------------------------------    
	
#Cálculo do Erro da Predição
		## TODO: Cálculo do Erro        
		if (output[0]>output[1]):
			if (yi[0]>yi[1]):
				predictions+=1
				
		if (output[1]>=output[0]):
			if (yi[1]>yi[0]):
				predictions+=1

print("A Acurácia da Predição é de: {:.3f}".format(predictions/n_records))
print("Pessos Camada Oculta Entrada",weights_input_hidden)
print("Pessos Camada Oculta Entrada",weights_hidden_output)

plt.plot(IndiceError, EvolucaoError, 'r') # 'r' is the color red
plt.xlabel('Epochs')
plt.ylabel('Erro Quadrático')
plt.title('Evolução do Erro no treinamento da MPL')
plt.show()