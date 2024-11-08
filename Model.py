# Importa o módulo subprocess, que será utilizado para instalar bibliotecas caso não estejam presentes.
import subprocess
# Importa o módulo sys, que fornecerá o caminho do interpretador para instalar pacotes via pip.
import sys

# Verifica se a biblioteca numpy está instalada; caso contrário, a instala.
try:
    import numpy as np
except ImportError:
    print("Biblioteca 'numpy' não encontrada. Instalando...")
    # Instala a biblioteca "numpy" que será usada para operações numéricas.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])

# Verifica se a biblioteca scikit-learn (sklearn) está instalada; caso contrário, a instala.
try:
    from sklearn.metrics import mean_squared_error
except ImportError:
    print("Biblioteca 'scikit-learn' não encontrada. Instalando...")
    # Instala a biblioteca "scikit-learn" que será usada para escalonamento das features.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])

# Verifica se o TensorFlow está instalado; caso contrário, o instala
try:
    import tensorflow as tf
    # Importa o modelo Sequential e as camadas da biblioteca Keras para construir e treinar redes neurais.
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    # Importa EarlyStopping, que interrompe o treinamento do modelo se o desempenho em validação não melhorar.
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    print("TensorFlow não encontrado. Instalando...")
    # Instala a biblioteca TensorFlow, que inclui o Keras para construção de redes neurais.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])

'''
A ser implementado...

#
custom_loss_functions_setup = {
    "adjmse1":{
        "alpha": 2.0
    },
    "adjmse2":{
        "alpha": 2.5
    },
    "adjmse3":{
        "gamma": 0.25
    },
    "adjmse4":{
        "alpha": 2.5,
        "gamma": 90
    }
}

#
def adjmse1(alpha=2.0):
    
    def loss(y_true, y_pred):
        # Calcular o erro quadrático
        error = tf.square(y_true - y_pred)
        
        # Condição para ajustar o erro: sinais opostos
        sign = tf.sign(y_true * y_pred)
        
        # Penalização assimétrica
        adjusted_error = tf.where(sign < 0, alpha * error, (1 / alpha) * error)
        
        # Retornar a média do erro ajustado
        return tf.reduce_mean(adjusted_error)
    
    return loss

#
def adjmse2(alpha=2.5):
    
    def loss(y_true, y_pred):
        # Calcular o erro quadrático
        error = tf.square(y_true - y_pred)
        
        # Condição para ajustar o erro: sinais opostos
        sign = tf.sign(y_true * y_pred)
        
        # Penalização assimétrica
        adjusted_error = tf.where(sign < 0, alpha * error, error)
        
        # Retornar a média do erro ajustado
        return tf.reduce_mean(adjusted_error)
    
    return loss

#
def adjmse3(gamma=0.25):
    
    def loss(y_true, y_pred):
        # Calcular o erro quadrático
        error = tf.square(y_true - y_pred)
        
        # Condição para ajustar o erro: sinais opostos
        sign = tf.sign(y_true * y_pred)
        
        # Penalização assimétrica
        adjusted_error = tf.where(sign < 0, (1 + gamma) * error, gamma * error)
        
        # Retornar a média do erro ajustado
        return tf.reduce_mean(adjusted_error)
    
    return loss

#
def adj4(alpha=2.5, beta=90.0):
    
    def loss(y_true, y_pred):
        # Calcular o erro quadrático
        error = tf.square(y_true - y_pred)
        
        # Ajuste sigmoidal
        sign_product = y_true * y_pred
        adjustment_factor = alpha / (1 + tf.exp(beta * sign_product))
        
        # Penalização ajustada pela curva sigmoidal
        adjusted_error = adjustment_factor * error
        
        # Retornar a média do erro ajustado
        return tf.reduce_mean(adjusted_error)
    
    return loss
'''

class Model:
    '''
        Description:
            Classe para criação e treinamento de um modelo LSTM para previsão de séries temporais. Essa classe define a estrutura 
            de um modelo LSTM com camadas sequenciais e métodos para configurar e treinar a rede neural.
    '''
    
    def __init__(self, features_number: int, lstm_sequence_length: int) -> None:
        '''
            Description:
                Inicializa a classe Model com o número de características e o comprimento da sequência LSTM.
            Args:
                features_number (int): Número de características (features) em cada amostra de entrada.
                lstm_sequence_length (int): Comprimento da janela móvel da camada LSTM.
            Return:
                None: Esta função inicializa a instância do modelo com o número de características e o comprimento da sequência, 
                sem retornar valores.
        '''
        
        # Define o número de características (features) do modelo.
        self.features_number = features_number
        # Define o comprimento da janela móvel da camada LSTM do modelo.
        self.lstm_sequence_length = lstm_sequence_length

    def create_LSTM_model(self) -> None: 
        '''
            Description:
                Cria e compila o modelo LSTM. O modelo contém camadas LSTM, Dropout entre as camadas e uma camada densa para a saída. 
                É configurado para usar o otimizador Adam e a função de perda mean_squared_error.
            Args:
                Nenhum argumento é passado diretamente, pois a função utiliza os atributos do objeto. 
            Return:
                None: O modelo LSTM é armazenado no atributo `self.model` da instância.
        '''
        
        """
        Este código define, constrói e compila um modelo LSTM (Long Short-Term Memory) para previsões de séries temporais.

        Parâmetros dos métodos utilizados:
        1. `LSTM`:
            - `units`: Número de neurônios na camada LSTM.
            - `return_sequences`: Se `True`, retorna a sequência completa de saídas para cada unidade LSTM; se `False`, retorna apenas 
            a saída final.
            - `input_shape`: Tupla que define a forma da entrada (tamanho da sequência de tempo, número de features).

        2. `Dropout`:
            - `rate`: Fração de neurônios a serem descartados durante o treinamento para evitar overfitting.

        3. `Dense`:
            - `units`: Número de neurônios na camada densa (totalmente conectada). No contexto de regressão, geralmente é 1.

        4. `compile`:
            - `optimizer`: Algoritmo de otimização utilizado para ajustar os pesos do modelo. 'adam' é uma escolha comum para LSTM.
            - `loss`: Função de perda que o modelo tentará minimizar durante o treinamento. Neste caso, 'mean_squared_error' (erro quadrático médio) é usado para problemas de regressão.
        """

        # Definindo o modelo LSTM
        # Utilizamos o modelo 'Sequential', que permite empilhar camadas de forma linear. Esse tipo de modelo é adequado para a maioria das 
        # arquiteturas de rede neural onde as camadas são adicionadas uma após a outra.
        model = Sequential()

        # Adiciona a camada de entrada explicitamente
        model.add(Input(shape=(self.lstm_sequence_length, self.features_number)))

        # Adiciona a primeira camada LSTM com 64 unidades (neurônios).
        model.add(LSTM(units=64, return_sequences=True))

        # Adiciona uma camada de Dropout. Tal camada é usada para evitar overfitting durante o treinamento do modelo. Ao definir `rate=0.2`, 
        # estamos especificando que 20% dos neurônios da camada anterior serão desligados aleatoriamente em cada atualização do ciclo de treinamento.
        model.add(Dropout(0.2))

        # Adiciona uma segunda LSTM. Ao definirmos`return_sequences=False` indicamos que esta é a última camada LSTM na rede, e 
        # ela só retorna a última saída em vez de toda a sequência.
        model.add(LSTM(units=128, return_sequences=False))

        # Adiciona uma segunda camada de Dropout. Assim como antes, 20% dos neurônios serão desativados em cada iteração de treinamento.
        model.add(Dropout(0.2))

        # Adiciona a camada de saída. Tal camada é uma camada densa totalmente conectada com um único neurônio (units=1). Isso é adequado para 
        # problemas de regressão onde a previsão final é um único valor contínuo.
        model.add(Dense(units=1))

        # Compila o modelo criado. O modelo é compilado com o otimizador 'adam', que é eficiente para grandes volumes de dados e adequado para
        # problemas de regressão. Além disso, setamos a função de perda como sendo a 'mean_squared_error' (MSE), que é uma escolha comum para 
        # medir o erro médio ao quadrado entre as previsões e os valores reais.
        model.compile(optimizer='adam', loss="mean_squared_error")

        # O atributo 'model' agora contém o modelo LSTM compilado e pode ser usada para treinamento e previsões.
        self.model = model
    
    def train_model_and_get_results(self, X_train_scaled_sequences, X_test_scaled_sequences ,
                                y_train_scaled_sequences, y_test_scaled_sequences) -> tuple[np.ndarray, float]:
        '''
            Description:
                Treina o modelo LSTM e retorna as previsões e a métrica de erro RMSE.
            Args:
                X_train_scaled_sequences (np.ndarray): Dados de entrada de treinamento, escalados e em janela móvel.
                X_test_scaled_sequences (np.ndarray): Dados de entrada de teste, escalados e em janela móvel.
                y_train_scaled_sequences (np.ndarray): Valores alvo de treinamento, escalados e em janela móvel.
                y_test_scaled_sequences (np.ndarray): Valores alvo de teste, escalados e em janela móvel.
            Return:
                tuple: Uma tupla contendo as previsões (`np.ndarray`) e o valor de erro RMSE (`float`) para o conjunto de teste.
        '''
            
        """
            Este código treina o modelo LSTM usando o método `fit` do módulo Keras da biblioteca tensorflow.

            Parâmetros do método `fit`:
            1. `x`: Dados de entrada para treinamento. No contexto de séries temporais com LSTM, este é um array de sequências escaladas.
            2. `y`: Dados de saída (alvo) para treinamento. Correspondem aos valores que o modelo deve prever com base nos dados de entrada.
            3. `epochs`: Número de vezes que o modelo passará por todo o conjunto de dados de treinamento. Um número maior de épocas pode melhorar o 
                ajuste do modelo, mas também aumenta o risco de overfitting.
            4. `batch_size`: Número de amostras que o modelo processa antes de atualizar os pesos. Tamanhos de batch menores podem resultar em um 
                treinamento mais ruidoso, mas permitem uma melhor generalização.
            5. `validation_split`: Proporção dos dados de treinamento que será usada para validação. Ajuda a monitorar a performance do modelo em 
                dados que ele não viu durante o treinamento.
            6. `verbose`: Nível de verbosidade do processo de treinamento. `verbose=1` exibe uma barra de progresso detalhada durante o treinamento.
        """  
        
        # Configuração de Early Stopping para interromper o treinamento se a perda de validação não melhorar
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Treina o modelo
        # O método `fit` treina o modelo LSTM usando os dados de entrada (`X_train_scaled_sequence`) e as saídas correspondentes 
        # (`y_train_scaled_sequence`).
        self.model.fit(
            X_train_scaled_sequences,  # Dados de entrada de treino (sequências temporais)
            y_train_scaled_sequences,  # Dados de saída de treino (valores de previsão para cada sequência)
            epochs=70,  # Número de épocas: o modelo passará 70 vezes por todo o conjunto de dados de treinamento
            batch_size=32,  # Tamanho do batch: o modelo ajusta os pesos após processar cada lote de 32 amostras
            validation_split=0.2,  # 20% dos dados de treinamento serão usados para validação.
            verbose=0,  # Nível de verbosidade: 1 mostra uma barra de progresso e resultados após cada época
            callbacks=[early_stopping]
        )
        
        """
                Este bloco de código faz previsões sobre um conjunto de dados de teste usando o modelo LSTM treinado acima, e, em seguida,
                avalia a precisão dessas previsões usando a métrica de Root Mean Squared Error (RMSE).
        """

        # Realiza a previsão de valores para dados não vistos até então.
        predicted = self.model.predict(X_test_scaled_sequences)

        # Obter a função de perda ajustada
        #loss_function = adjmse2(alpha=2.5)

        # Calcular o erro ajustado
        #adjusted_error = loss_function(y_test_scaled_sequences, predicted)
        
        # Avalia, utilizando a métrica RMSE, o resultado das previsões feitas acima.
        RMSE = np.sqrt(mean_squared_error(y_test_scaled_sequences,predicted))

        return predicted, RMSE
            
