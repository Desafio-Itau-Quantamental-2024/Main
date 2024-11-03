#
import subprocess
#
import sys
#
#import warnings
#
import pandas as pd
#
import numpy as np
#
import yfinance as yf
#
import statsmodels.api as sm

#
from datetime import datetime, timedelta
# Importa o módulo da biblioteca "sklearn" que será utilizado para escalar as features dos DataFrames que serão usados no modelo LSTM.
from sklearn.preprocessing import MinMaxScaler

# Verifica se a biblioteca arch está instalada; caso contrário, a instala.
try:
    from arch import arch_model
except ImportError:
    print("Biblioteca 'arch' não encontrada. Instalando...")
    # Instala a biblioteca "arch" que será responsável por fornecer o módulo 'arch_model'.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "arch"])

# Verifica se a biblioteca talib está instalada; caso contrário, a instala.
try:
    import talib
except ImportError:
    print("Biblioteca 'talib' não encontrada. Instalando...")
    # Instala a biblioteca "TA-Lib" que será responsável por fornecer a implementação de alguns indicadores técnicos.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Ta-Lib"])
    
class Ticker:
    '''
        Description:
            A classe "Ticker" representa um ativo financeiro identificado por um símbolo único (ticker). 
            Ela é responsável por baixar os dados históricos de preços e calcular vários indicadores técnicos, 
            como retornos aritméticos e logarítmicos, Bandas de Bollinger, RSI (Índice de Força Relativa), 
            ATR (Average True Range) e Momentum. O objetivo principal da classe é fornecer um conjunto padronizado de dados 
            e indicadores para análise de ativos financeiros. Além disso, valida se os dados extraídos estão de acordo com 
            o período solicitado e prepara os dados para uso posterior, como em modelos de aprendizado de máquina.
    '''
    
    def __is_ticker_valid__(self) -> bool:
        '''
            Description:
                Verifica se o ticker é considerado válido. Um ticker será considerado válido quando:
                - O atributo `data` não for vazio.
                - O índice inicial do atributo `data` corresponder à data de extração inicial (`data_extraction_initial_date`).
                - O índice final do atributo `data` corresponder à data de extração final (`data_extraction_final_date` - 1 dia).
            
            Args:
                Nenhum argumento é passado diretamente para esta função, pois ela usa os atributos do objeto 
                (`self.data`, `self.data_extraction_initial_date`, etc.).

            Return:
                bool: Retorna `True` se o ticker for considerado válido de acordo com as condições definidas, caso contrário retorna `False`.
        '''
        
        # Verifica se o atributo data é vazio.
        if(len(self.data) == 0): return False
        
        # Verifica se a data inicial do índice do DataFrame corresponde à data de extração inicial.
        condition_2 = (self.data.index[0].date() == self.data_extraction_initial_date)
        # Verifica se a data final do índice do DataFrame corresponde à data de extração final - 1 dia.
        condition_3 = (self.data.index[len(self.data.index)-1].date() == self.data_extraction_final_date - timedelta(days=1))
        
        # Observação: Irá ocorrer problemas com as condições acima caso self.data_extraction_initial_date ou self.data_extraction_final_data 
        # não sejam dias de negociação. Visto que, caso isso ocorra, tais condições serão falsas e consequentemente o Ticker em questão 
        # não será considerado válido.
        
        # Retorna False se alguma das condições não for satisfeita.
        if(not(condition_2) or not(condition_3)):
            return False
        
        # Retorna True caso contrário (se todas as condições forem satisfeitas).
        return True
        
        
    def __get_ticker_data__(self) -> pd.DataFrame:
        '''
            Description:
                Baixa os dados do ticker usando a biblioteca `yfinance` para o período de tempo definido pelos atributos 
                `data_extraction_initial_date` e `data_extraction_final_date`.
                
            Args:
                Nenhum argumento é passado diretamente para esta função, pois ela utiliza os atributos do objeto 
                (`self.symbol`, `self.data_extraction_initial_date`, etc.).
            
            Return:
                pd.DataFrame: Um DataFrame contendo os dados históricos do ticker para o período solicitado. 
                Retorna um DataFrame vazio se ocorrer algum erro.
        '''
        
        try:
            # Faz o download dos dados do ticker em questão.
            return yf.download(self.symbol, 
                               start=self.data_extraction_initial_date, 
                               end=self.data_extraction_final_date)
        except Exception as e:  
            # Captura e imprime a exceção caso ocorra um erro ao baixar os dados.
            print(f"Erro ao baixar os dados para {self.symbol}: {e}")
            # Retorna um DataFrame vazio em caso de erro.
            return pd.DataFrame()  
        
    def __get_arithmetic_returns__(self) -> pd.Series:
        '''
            Description:
                Calcula os retornos aritméticos baseados nos preços ajustados de fechamento (Adj Close).
                O retorno aritmético mede a variação percentual do preço de fechamento ajustado em relação ao período anterior.
                
            Args:
                Nenhum argumento é passado diretamente, pois a função utiliza os atributos do objeto.
            
            Return:
                pd.Series: Uma série com os retornos aritméticos para o período definido, divididos por 100 para normalizar os valores.
        '''
        
        # Obtém os preços ajustados de fechamento.
        daily_prices = self.data['Adj Close']

        # Calcula os retornos aritméticos usando o método ROC da talib.
        arithmetic_returns = talib.ROC(daily_prices, timeperiod=self.__features_time_period__['returns_time_period']) 
        
        # Normaliza os retornos dividindo por 100 (ROC retorna valores em percentual).
        return (arithmetic_returns/100)
    
    def __get_logarithmic_returns__(self) -> pd.Series:
        '''
            Description:
                Calcula os retornos logarítmicos baseados nos retornos aritméticos. 
                O retorno logarítmico é uma medida que transforma os retornos aritméticos em uma forma que facilita a análise estatística e 
                agregação de múltiplos retornos.
            
            Args:
                Nenhum argumento é passado diretamente, pois a função utiliza os atributos do objeto.
            
            Return:
                pd.Series: Uma série com os retornos logarítmicos.
        '''
        
        # Obtém os retornos aritméticos.
        arithmetic_returns = self.__get_arithmetic_returns__()
        
        # Calcula os retornos logarítmicos aplicando a função log(1 + retorno aritmético).
        logarithmic_returns = arithmetic_returns.apply(lambda x: np.log(1 + x))
        
        return logarithmic_returns
    
    def __get_bollinger_bands_with_exponential_moving_average__(self) -> pd.Series:
        '''
            Description:
                Calcula as Bandas de Bollinger com base na média móvel exponencial dos preços ajustados de fechamento (Adj Close).
                As Bandas de Bollinger são usadas para medir a volatilidade e identificar potenciais pontos de reversão de tendência.
            
            Args:
                Nenhum argumento é passado diretamente, pois a função utiliza os atributos do objeto.
            
            Return:
                Tuple[pd.Series, pd.Series, pd.Series]: Um tuple contendo três séries: a banda superior, a banda média (EMA) e a banda inferior.
        '''
        
        # Obtém os preços ajustados de fechamento.
        daily_prices = self.data['Adj Close']
        
        # Calcula as Bandas de Bollinger usando o talib.
        upper_band, middle_band, lower_band = talib.BBANDS(daily_prices, 
                                                           timeperiod=self.__features_time_period__['exponential_moving_average_time_period'],
                                                           nbdevup=2, nbdevdn=2, matype=1) # Tipo de média móvel: 1 é a média móvel exponencial.
        
        return upper_band, middle_band, lower_band
        
    def __get_relative_strength_index__(self) -> pd.Series:
        '''
            Description:
                Calcula o Índice de Força Relativa (RSI) com base nos preços ajustados de fechamento (Adj Close).
                O RSI é um indicador de momentum usado para identificar condições de sobrecompra ou sobrevenda de um ativo.
            
            Args:
                Nenhum argumento é passado diretamente, pois a função utiliza os atributos do objeto.
            
            Return:
                pd.Series: Uma série contendo os valores do RSI para o período definido.
        '''
        
        # Obtém os preços ajustados de fechamento.
        daily_prices = self.data['Adj Close']
        
        # Calcula o RSI usando o talib.
        relative_strength_index = talib.RSI(daily_prices, timeperiod=self.__features_time_period__['relative_strength_index_time_period'])
        
        return relative_strength_index
    
    def __get_average_true_range__(self) -> pd.Series:
        '''
            Description:
                Calcula o Average True Range (ATR) baseado nas séries de preços máximos, mínimos e de fechamento.
                O ATR é um indicador de volatilidade que mede a variação média entre o preço máximo e o preço mínimo do período.
            
            Args:
                Nenhum argumento é passado diretamente, pois a função utiliza os atributos do objeto.
            
            Return:
                pd.Series: Uma série contendo os valores do ATR para o período definido.
        '''
        
        # Obtém as séries de preços máximos, mínimos e de fechamento.
        high_prices = self.data['High']
        low_prices = self.data['Low']
        close_prices = self.data['Close']
        
        # Calcula o ATR usando o talib.
        average_true_range = talib.ATR(high_prices, low_prices, close_prices, timeperiod=self.__features_time_period__['average_true_range_time_period'])

        return average_true_range
    
    def __get_momemtum__(self) -> pd.Series:
        '''
            Description:
                Calcula o indicador de momentum baseado nos preços ajustados de fechamento (Adj Close).
                O momentum mede a taxa de variação dos preços, indicando a velocidade e direção das mudanças de preços.
            
            Args:
                Nenhum argumento é passado diretamente, pois a função utiliza os atributos do objeto.
            
            Return:
                pd.Series: Uma série contendo os valores de momentum para o período definido.
        '''
        
        # Obtém os preços ajustados de fechamento.
        daily_prices = self.data['Adj Close']
        
        # Calcula o momentum usando o talib
        momemtum = talib.MOM(daily_prices, timeperiod=self.__features_time_period__['momemtum_time_period'])

        return momemtum
        
    def __get_MSGARCH_results__(self) -> tuple[pd.Series, pd.DataFrame]:
        '''
            Description:

            Args:
                test é a série temporal dos retornos acumulados
            Return:

        '''
        
        #
        cumulated_arithmetic_returns = self.data['Adj Close'].pct_change(periods=self.__features_time_period__['pct_change_period'])
        
        #
        cumulated_arithmetic_returns = cumulated_arithmetic_returns*100
            
        #
        cumulated_arithmetic_returns.dropna(inplace=True)
        
        #
        markov_model = sm.tsa.MarkovAutoregression(cumulated_arithmetic_returns, k_regimes=2, order=1, switching_variance=True)
        #
        markov_fit = markov_model.fit()

        #
        garch_model = arch_model(cumulated_arithmetic_returns, vol='Garch', p=1, q=1)
        #
        garch_fit = garch_model.fit(disp="off")
        
        # print(garch_fit.summary())
        
        # Probabilidades suavizadas dos regimes
        smoothed_probs = markov_fit.smoothed_marginal_probabilities

        # Calcular a variância condicional de cada regime
        conditional_volatility = garch_fit.conditional_volatility

        return smoothed_probs, conditional_volatility
    
    def __init__(self, symbol: str, data_extraction_initial_date: datetime.date , data_extraction_final_date: datetime.date,
                 features_time_period: dict, strategy_time_period: int) -> None:
        '''
            Description:
                Inicializa uma instância da classe "Ticker", definindo os parâmetros essenciais como o símbolo do ativo financeiro,
                o período de extração de dados e os períodos de tempo utilizados para calcular indicadores financeiros. Durante a inicialização,
                os dados do ticker são baixados e validados.

            Args:
                symbol (str): O símbolo do ativo financeiro (ticker) cujos dados históricos serão extraídos.
                data_extraction_initial_date (datetime.date): A data inicial para a extração dos dados.
                data_extraction_final_date (datetime.date): A data final para a extração dos dados.
                features_time_period (dict): Um dicionário que contém os parâmetros de tempo para o cálculo dos indicadores financeiros, 
                                            como RSI, Bandas de Bollinger, etc.
            
            Return:
                None: O construtor não retorna nada, mas prepara a instância para que os dados e indicadores possam ser utilizados posteriormente.
        '''
        
        # Define o símbolo do ticker.
        self.symbol = symbol
        
        # Define a data inicial para extração dos dados.
        self.data_extraction_initial_date = data_extraction_initial_date
        
        # Define a data final para extração dos dados.
        self.data_extraction_final_date = data_extraction_final_date
        
        # Define os períodos de tempo para calcular os indicadores financeiros.
        self.__features_time_period__ = features_time_period
        
        #
        self.__strategy_time_period__ = strategy_time_period
        
        # Baixa os dados do ticker.
        self.data = self.__get_ticker_data__()
        
        # Verifica se o ticker é válido.
        self.is_valid = self.__is_ticker_valid__()
        
        #
        self.__adjust_data__()
        
        # 
        self.__y_train__ = None
        self.__y_test__ = None
        self.__y_test_scaled_sequences__ = None
    
    def __set_features__(self) -> None:
        '''
            Description:
                Calcula e adiciona alguns indicadores financeiros como colunas ao DataFrame 'data' da instância. 
                Esses indicadores incluem retornos logarítmicos, Bandas de Bollinger com média móvel exponencial, RSI, ATR e Momentum.
            
            Args:
                Nenhum argumento é passado diretamente, pois a função utiliza os atributos do objeto.
            
            Return:
                None: A função não retorna nada, mas modifica o DataFrame 'data' da instância.
        '''
        
        # Adiciona os retornos logarítmicos ao DataFrame.
        self.data['Log returns'] = self.__get_logarithmic_returns__()
        
        # Adiciona as Bandas de Bollinger (banda superior, EMA, banda inferior) ao DataFrame.
        self.data['B. upper bands'], self.data['EMA'], self.data['B. lower bands'] = self.__get_bollinger_bands_with_exponential_moving_average__()
        
        # Adiciona o Índice de Força Relativa (RSI) ao DataFrame.
        self.data['RSI'] = self.__get_relative_strength_index__()
        
        # Adiciona o Average True Range (ATR) ao DataFrame.
        self.data['ATR'] = self.__get_average_true_range__()
        
        # Adiciona o indicador de Momentum ao DataFrame.
        self.data['MOM'] = self.__get_momemtum__()
        
        # Defina o símbolo do VIX (Volatility Index)
        vix_symbol = '^VIX'

        # Baixe os dados históricos do VIX
        vix_data = yf.download(vix_symbol, start=self.data_extraction_initial_date, 
                               end=self.data_extraction_final_date, interval=self.__features_time_period__['vix_time_period'])

        #
        self.data['VIX'] = vix_data['Adj Close'].pct_change(periods=self.__features_time_period__['pct_change_period'])
    
    def __remove_some_features__(self) -> None:
        '''
            Description:
                Remove algumas colunas desnecessárias do DataFrame, especificamente as colunas 'Open', 'High', 'Low', e 'Close'.
            
            Args:
                Nenhum argumento é passado diretamente, pois a função utiliza os atributos do objeto.
            
            Return:
                None: A função não retorna nada, mas modifica o DataFrame 'data' da instância.
        '''
        
        # Remove as colunas 'Open', 'High', 'Low' e 'Close' do DataFrame.
        self.data.drop(columns=['Open','High','Low','Close'], inplace=True)
    
    def __get_low_val_period_probability__(self):
        '''
            Description:
            
            Args:
            
            Return:
        '''
        
        self.low_volatility_periods_probabilities = []
        
        for i in range(0,len(self.smoothed_probs),self.__strategy_time_period__):
            aux = 0
            for j in range(self.__strategy_time_period__):
                aux += self.smoothed_probs.iloc[(i+j),0]
            self.low_volatility_periods_probabilities.append(aux/self.__strategy_time_period__)
        
        self.low_volatility_periods_probabilities = pd.Series(self.low_volatility_periods_probabilities)

    def __adjust_data__(self) -> None:
        '''
            Description:
                Prepara os dados do ticker, calculando e adicionando os indicadores financeiros (features) e realizando
                a limpeza dos dados, caso o ticker seja válido. Este método é útil para garantir que o DataFrame 'data'
                contenha as informações necessárias para análises subsequentes.
            
            Args:
                Nenhum argumento é passado diretamente, pois a função utiliza os atributos do objeto.
            
            Return:
                None: A função não retorna nada, mas modifica o DataFrame 'data' da instância ao adicionar indicadores e
                remover colunas e linhas desnecessárias.
        '''
        
        # Se o ticker for considerado válido:
        if self.is_valid:
            # Calcula e adiciona as features (indicadores financeiros) ao DataFrame.
            self.__set_features__()
            
            #
            self.smoothed_probs, self.conditional_volatility = self.__get_MSGARCH_results__()
            
            # Remove algumas colunas que não serão necessárias.
            self.__remove_some_features__()
            
            # Remove todas as linhas que contenham valores nulos (NaN) no DataFrame.
            self.data.dropna(inplace=True)
            
            # Define um índice de corte, que é o índice máximo inicial comum entre os DataFrames (data, smoothed_probs e conditional_volatility).
            cut_index = max(self.data.index[0],self.smoothed_probs.index[0],self.conditional_volatility.index[0])
            
            # Aplica o índice de corte para alinhar as séries temporais a partir de uma data comum.
            self.data = self.data[self.data.index > cut_index]
            self.smoothed_probs = self.smoothed_probs[self.smoothed_probs.index > cut_index]
            self.conditional_volatility = self.conditional_volatility[self.conditional_volatility.index > cut_index]
            
            # Padroniza a quantidade de pontos nas séries temporais para ser múltipla de 5, facilitando estratégias de 5 ou 20 dias.
            self.data = self.data.iloc[:(len(self.data.index) - len(self.data.index) % 5)]
            self.smoothed_probs = self.smoothed_probs.iloc[:(len(self.smoothed_probs.index) - len(self.smoothed_probs.index) % 5)]
            self.conditional_volatility = self.conditional_volatility.iloc[:(len(self.conditional_volatility.index) - len(self.conditional_volatility.index) % 5)]
            
            #
            self.__get_low_val_period_probability__()
            
            
            
    
    def __train_test_split__(self, test_initial_day: datetime.date, test_final_day: datetime.date) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        '''
            Description:
                Divide os dados em conjuntos de treino e teste com base nas datas fornecidas. As features (X) e o alvo (y) são separados
                e divididos em dados de treino e teste.

            Args:
                test_initial_day (datetime.date): Data que define o início do período de teste.
                test_final_day (datetime.date): Data que define o final do período de teste.

            Return:
                tuple: Retorna quatro elementos - X_train (features de treino), X_test (features de teste), y_train (alvo de treino) e y_test (alvo de teste).
        '''
        
        # Separa as features (X) e o target (y).
        X = self.data.drop(columns=['Adj Close'])
        y = self.data['Adj Close']
        
        # Verifica se a data 'test_initial_day' não existe em 'self.data.index'.
        if(test_initial_day not in self.data.index):
            # Caso 'test_initial_day' não exista em 'self.data.index', substitui 'test_initial_day' pela data mais próxima presente 
            # em 'self.data.index'.
        
            #
            new_test_initial_day = self.data.index[self.data.index.get_indexer([test_initial_day], method='nearest')[0]]
            
            #
            test_initial_day = new_test_initial_day
            
            #
            #warnings.warn(f"A data {test_initial_day.date()} não é um dia de negociação válida. Por conta disso, ela será substituida pela data de negociação mais próxima, que no caso é {new_test_initial_day.date()}")
        
        # Verifica se a data 'test_initial_day' não existe em 'self.data.index'.
        if(test_final_day not in self.data.index):
            # Caso 'test_initial_day' não exista em 'self.data.index', substitui 'test_initial_day' pela data mais próxima presente 
            # em 'self.data.index'.
        
            #
            new_test_final_day = self.data.index[self.data.index.get_indexer([test_final_day], method='nearest')[0]]
            
            #
            test_final_day = new_test_final_day
            
            #
            #warnings.warn(f"A data {test_final_day.date()} não é um dia de negociação válida. Por conta disso, ela será substituida pela data de negociação mais próxima, que no caso é {new_test_final_day.date()}")
            
        # Separa o conjunto de séries temporais das features em dados de treino e dados de teste.
        X_train, X_test = X[X.index < test_initial_day], X[(X.index >= test_initial_day) & (X.index <= test_final_day)]
        
        # Separa a série temporal do target em dados de treino e dados de teste.
        y_train, y_test = y[y.index < test_initial_day], y[(y.index >= test_initial_day) & (y.index <= test_final_day)]

        return X_train, X_test, y_train, y_test
    
    def __scale_data__(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        '''
            Description:
                Normaliza as features e o target tanto para os conjuntos de treino quanto de teste usando o MinMaxScaler. Além disso,
                redimensiona o alvo (y) para uma matriz bidimensional antes da normalização.

            Args:
                X_train (pd.DataFrame): Conjunto de treino das features.
                X_test (pd.DataFrame): Conjunto de teste das features.
                y_train (pd.Series): Conjunto de treino do alvo.
                y_test (pd.Series): Conjunto de teste do alvo.

            Return:
                tuple: Retorna quatro elementos - X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled.
        '''
        
        # Cria uma instancia do MinMaxScaler para as features do ticker em questão.
        scaler_features = MinMaxScaler()

        # Ajusta a instância criada acima ao conjunto de treino das features do ticker em questão.
        X_train_scaled = scaler_features.fit_transform(X_train)  

        # Redimensiona y_train para que ele seja bidimensional (Necessário para o MinMaxScaler).
        resized_y_train = y_train.values.reshape(-1, 1)  

        # Cria uma instancia do MinMaxScaler para o target do ticker em questão.
        scaler_target = MinMaxScaler()

        # Ajusta a instância criada acima ao conjunto de treino do target do ticker em questão.
        y_train_scaled = scaler_target.fit_transform(resized_y_train)  

        # Normaliza o conjunto de teste das features do ticker em questão usando a instância que foi criada e ajustada aos dados de treino
        # desse mesmo ticker.
        X_test_scaled = scaler_features.transform(X_test)  

        # Redimensiona y_test para que seja bidimensional (Necessário para o MinMaxScaler).
        resized_y_test = y_test.values.reshape(-1, 1) 

        # Normaliza o conjunto de teste do target  do ticker em questão usando a instância que foi criada e ajustada aos dados de treino desse
        # mesmo ticker.
        y_test_scaled = scaler_target.transform(resized_y_test)
        
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled
    
    def __create_time_sequences_for_lstm__(self, X_train_scaled: np.ndarray, X_test_scaled: np.ndarray, y_train_scaled: np.ndarray,
                              y_test_scaled: np.ndarray, sequence_length: int) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        '''
            Description:
                Constrói sequências temporais a partir dos dados normalizados de treino e teste, criando janelas móveis
                de tamanho 'sequence_length'. Essas sequências são necessárias para treinar uma LSTM.

            Args:
                X_train_scaled (np.ndarray): Conjunto de treino normalizado das features.
                X_test_scaled (np.ndarray): Conjunto de teste normalizado das features.
                y_train_scaled (np.ndarray): Conjunto de treino normalizado do alvo.
                y_test_scaled (np.ndarray): Conjunto de teste normalizado do alvo.
                sequence_length (int): O comprimento das sequências temporais usadas para treinar a LSTM.

            Return:
                tuple: Retorna quatro elementos - X_train_scaled_sequences, X_test_scaled_sequences, 
                    y_train_scaled_sequences, y_test_scaled_sequences (sequências temporais normalizadas para treino e teste).
        '''
    
        # Verifica se o sequence_length é maior que o número de amostras disponíveis
        if sequence_length > len(X_train_scaled):
            raise ValueError("O 'sequence_length' não pode ser maior que o número de amostras em 'X_train_scaled'.")
        if sequence_length > len(X_test_scaled):
            raise ValueError("O 'sequence_length' não pode ser maior que o número de amostras em 'X_test_scaled'.")
        
        # Calcula o comprimento dos intervalos de treino e teste, assumindo que 'X_train' e 'y_train' têm o mesmo comprimento, tal como 'X_test' e
        # 'y_test'.
        train_interval_length = len(X_train_scaled) - sequence_length
        test_interval_length = len(X_test_scaled) - sequence_length
        
        # Inicializa listas para armazenar as sequências temporais dos dados de treino e teste.
        X_train_sequences = []
        X_test_sequences = []
        y_train_sequences = []
        y_test_sequences = []
    
        # Cria sequências temporais para os dados de treino
        for i in range(train_interval_length):
            # Cria uma sequência temporal de 'sequence_length' dias para as features de treino.
            X_train_sequence = X_train_scaled[i: (i + sequence_length)]
            # O alvo será o valor no dia seguinte após a sequência temporal.
            y_train_sequence = y_train_scaled[i + sequence_length]
            # Adiciona as sequências temporais às listas correspondentes.
            X_train_sequences.append(X_train_sequence)
            y_train_sequences.append(y_train_sequence)

        # Cria sequências temporais para os dados de teste
        for j in range(test_interval_length):
            # Cria uma sequência temporal de 'sequence_length' dias para as features de teste.
            X_test_sequence = X_test_scaled[j: (j + sequence_length)]
            # O alvo será o valor no dia seguinte após a sequência temporal.
            y_test_sequence = y_test_scaled[j + sequence_length]
            # Adiciona as sequências temporais às listas correspondentes.
            X_test_sequences.append(X_test_sequence)
            y_test_sequences.append(y_test_sequence)
            
        # Converte as listas em arrays.
        X_train_scaled_sequences = np.array([np.array(arr) for arr in X_train_sequences])
        X_test_scaled_sequences = np.array([np.array(arr) for arr in X_test_sequences])
        y_train_scaled_sequences = np.array(y_train_sequences)
        y_test_scaled_sequences = np.array(y_test_sequences)
        
        return X_train_scaled_sequences, X_test_scaled_sequences, y_train_scaled_sequences, y_test_scaled_sequences
    
    def prepare_data_for_lstm(self, test_initial_day: datetime.date, test_final_day: datetime.date, lstm_time_sequences_length: int) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        '''
            Description:
                Prepara os dados para serem usados em uma LSTM. O processo envolve dividir os dados em conjuntos de treino e teste,
                normalizar os dados e criar sequências temporais para a LSTM.
                
                Observação: A rede neural será treinada com todos os dados antes da data "test_initial_day" e fará predições para 
                            todas as datas entre test_initial_day e test_final_day (inclusos).

            Args:
                test_initial_day (datetime.date): Data que define o início do período de teste.
                test_final_day (datetime.date): Data que define o final do período de teste.
                lstm_time_sequences_length (int): O comprimento das sequências temporais usadas para treinar a LSTM.

            Return:
                tuple: Retorna quatro elementos - X_train_scaled_sequences, X_test_scaled_sequences, 
                    y_train_scaled_sequences, y_test_scaled_sequences (sequências temporais normalizadas para treino e teste).
        '''

        
        # Divide os dados em conjuntos de treino e teste.
        X_train, X_test, self.__y_train__, self.__y_test__ = self.__train_test_split__(test_initial_day, test_final_day)
        
        # Normaliza os dados.
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = self.__scale_data__(X_train, X_test, self.__y_train__, self.__y_test__)
        
        # Cria sequências temporais para a LSTM.
        X_train_scaled_sequences, X_test_scaled_sequences, y_train_scaled_sequences, self.__y_test_scaled_sequences__ = self.__create_time_sequences_for_lstm__(X_train_scaled, X_test_scaled,
                                                                                                                   y_train_scaled, y_test_scaled,
                                                                                                                   lstm_time_sequences_length)
        
        return X_train_scaled_sequences, X_test_scaled_sequences, y_train_scaled_sequences, self.__y_test_scaled_sequences__