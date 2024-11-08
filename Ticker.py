# Importa o módulo subprocess, que será utilizado para instalar bibliotecas caso não estejam presentes.
import subprocess
# Importa o módulo sys, que fornecerá o caminho do interpretador para instalar pacotes via pip.
import sys

# Importa a biblioteca logging para capturar e armazenar os logs das requisições feitas via yfinance.
import logging

# Importa o módulo datetime da biblioteca padrão para manipulação de datas e horas.
from datetime import datetime, timedelta

# Importa o módulo warnings, usado para controle e supressão de avisos.
import warnings

# Verifica se a biblioteca pandas está instalada; caso contrário, a instala.
try:
    import pandas as pd
except ImportError:
    print("Biblioteca 'pandas' não encontrada. Instalando...")
    # Instala a biblioteca "pandas" que será usada para manipulação de dados.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    
# Verifica se a biblioteca numpy está instalada; caso contrário, a instala.
try:
    import numpy as np
except ImportError:
    print("Biblioteca 'numpy' não encontrada. Instalando...")
    # Instala a biblioteca "numpy" que será usada para operações numéricas.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    
# Verifica se a biblioteca yfinance está instalada; caso contrário, a instala.
try:
    import yfinance as yf
except ImportError:
    print("Biblioteca 'yfinance' não encontrada. Instalando...")
    # Instala a biblioteca "yfinance" que será usada para obter dados financeiros.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    
# Verifica se a biblioteca statsmodels está instalada; caso contrário, a instala.
try:
    import statsmodels.api as sm
except ImportError:
    print("Biblioteca 'statsmodels' não encontrada. Instalando...")
    # Instala a biblioteca "statsmodels" que será usada para modelagem estatística.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels"])

# Verifica se a biblioteca scikit-learn (sklearn) está instalada; caso contrário, a instala.
try:
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    print("Biblioteca 'scikit-learn' não encontrada. Instalando...")
    # Instala a biblioteca "scikit-learn" que será usada para escalonamento das features.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])

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
            A classe "Ticker" representa um ativo financeiro identificado por um símbolo único (.symbol).
            Ela é responsável por baixar dados históricos de preços e calcular diversos indicadores técnicos,
            incluindo retornos aritméticos e logarítmicos, Bandas de Bollinger, RSI (Índice de Força Relativa),
            ATR (Average True Range) e Momentum. Além disso, modela a volatilidade condicional de cada ativo 
            com base em um modelo inspirado no MSGARCH.

            O principal objetivo da classe é fornecer um conjunto padronizado de dados e indicadores para análise
            de ativos financeiros. Para tanto, a classe valida se os dados extraídos correspondem ao período solicitado
            e prepara esses dados para uso em modelos de aprendizado de máquina, especialmente redes LSTM.
            
        Observations:
            1. 
                O método `__is_ticker_valid__` considera que as datas de extração (`data_extraction_initial_date` e 
                `data_extraction_final_date`) são dias de negociação. Se uma dessas datas não corresponder a um dia de negociação 
                (por exemplo, feriados ou finais de semana), as condições de validação do ticker retornarão `False`, pois os dados 
                extraídos não coincidirão exatamente com o período solicitado. Isso pode fazer com que um ticker válido seja erroneamente 
                considerado inválido.
            2. 
                Os períodos de cálculo para cada indicador técnico, são definidos pelo dicionário `__features_time_period__`, 
                que é fornecido como argumento no construtor da classe.
    '''
    
    # Configura o sistema de logging para capturar todas as mensagens de log, incluindo sucesso e falha, das requisições feitas via yfinance, 
    # e redireciona essas mensagens para um arquivo de log.
    logging.basicConfig(
                filename='yfinance_log.txt', # Define o arquivo onde os logs serão armazenados
                level=logging.DEBUG,  # Captura todos os tipos de logs (desde DEBUG até CRITICAL)
                format='%(asctime)s - %(levelname)s - %(message)s') # Formato do log (inclui data, nível e mensagem)
    
    # Desativa o logging do yfinance, configurando o logger dessa biblioteca para não exibir logs no console.
    # Apenas logs de nível CRITICAL serão registrados, suprimindo a exibição de barras de progresso e outras mensagens.
    logging.getLogger('yfinance').setLevel(logging.CRITICAL)
        
    def __is_ticker_valid__(self) -> bool:
        '''
            Description:
                Verifica se o ticker é considerado válido. Um ticker será considerado válido quando:
                - O atributo `data` não for vazio.
                - O índice inicial do atributo `data` corresponder à data de extração inicial (`data_extraction_initial_date`).
                - O índice final do atributo `data` corresponder à data de extração final (`data_extraction_final_date` - 1 dia).
            Args:
                Nenhum argumento é passado diretamente, pois a função utiliza os atributos do objeto.
            Return:
                bool: Retorna `True` se o ticker for considerado válido de acordo com as condições definidas, caso contrário retorna `False`.
        '''
        
        # Verifica se o atributo data é vazio.
        if(len(self.data) == 0): return False
        
        # Verifica se a data inicial do índice do DataFrame corresponde à data de extração inicial.
        condition_2 = (self.data.index[0].date() == self.data_extraction_initial_date)
        # Verifica se a data final do índice do DataFrame corresponde à data de extração final - 1 dia.
        condition_3 = (self.data.index[len(self.data.index)-1].date() == self.data_extraction_final_date - timedelta(days=1))
        
        # Observação: Irão ocorrer problemas com as condições acima caso self.data_extraction_initial_date ou self.data_extraction_final_data 
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
                Nenhum argumento é passado diretamente, pois a função utiliza os atributos do objeto.  
            Return:
                pd.DataFrame: Um DataFrame contendo os dados históricos do ticker para o período solicitado. 
                Retorna um DataFrame vazio se ocorrer algum erro.
        '''

        try:
            # Faz o download dos dados do ticker em questão.
            data =  yf.download(self.symbol, 
                               start=self.data_extraction_initial_date, 
                               end=self.data_extraction_final_date,
                               progress=False)
            
            # Registra uma mensagem de log indicando que o download foi bem-sucedido.
            logging.info(f"Dados baixados com sucesso para {self.symbol}.")
            
            return data
        except Exception as e:  
            # Em caso de erro durante o download, captura e registra a exceção no log.
            logging.error(f"Erro ao baixar os dados para {self.symbol}: {e}")
            
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
                pd.Series: Uma série com os retornos aritméticos (divididos por 100 para normalizar os valores).
        '''
        
        # Obtém os preços ajustados de fechamento.
        daily_prices = self.data['Adj Close']

        # Calcula os retornos aritméticos usando o método ROC da talib.
        arithmetic_returns = talib.ROC(daily_prices, timeperiod=self.__features_time_period__['returns_time_period']) 
        
        # Normaliza os retornos dividindo por 100 (ROC retorna valores em percentual). # Será que isso aqui dá problema ?
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
                Calcula e retorna resultados de um modelo híbrido de Markov com GARCH aplicado aos retornos acumulados.
                Esse modelo identifica regimes de volatilidade (baixa e alta) e estima a volatilidade condicional 
                dos retornos acumulados com base nos preços ajustados de fechamento (Adj Close).
            Args:
                Nenhum argumento é passado diretamente, pois a função utiliza os atributos do objeto.
            Return:
                tuple[pd.Series, pd.DataFrame]: Uma tupla contendo:
                    - pd.Series: As probabilidades suavizadas para cada regime, resultantes do modelo de Markov.
                    - pd.DataFrame: A volatilidade condicional dos retornos acumulados, estimada pelo modelo GARCH.
        '''
        
        # Suprime o aviso do tipo ValueWarning
        warnings.filterwarnings("ignore", category=UserWarning) # O warning que aparece aqui é relacionado a frequência do DataFrame
                                                                # "cumulated_arithmetic_returns" que é passado para o método
                                                                # "sm.tsa.MarkovAutoregression". Contudo, pelos meus testes, a influência
                                                                # de definir a frequência dos dados do DataFrame é nula. Por conta disso,
                                                                # convém, por hora, suprimir esse warning. Embora seja importante rever isso
                                                                # com mais calma, no futuro.

        # Calcula os retornos acumulados aritméticos para o período definido.
        cumulated_arithmetic_returns = self.data['Adj Close'].pct_change(periods=self.__features_time_period__['pct_change_period'])
        
        # Multiplica por 100 para converter os retornos calculados acima para porcentagem.
        cumulated_arithmetic_returns = cumulated_arithmetic_returns*100
            
        # Remove valores nulos da série temporal dos retornos acumulados aritméticos.
        cumulated_arithmetic_returns.dropna(inplace=True)
        
        # Define o modelo de autorregressão de Markov com 2 regimes e variância alternada entre os regimes.
        # Este modelo ajusta uma série temporal aos retornos acumulados, permitindo alternância entre dois regimes
        # de volatilidade (por exemplo, alta e baixa volatilidade). A alternância é modelada por uma cadeia de Markov.
        # O parâmetro `switching_variance=True` indica que cada regime terá uma variância distinta.
        markov_model = sm.tsa.MarkovAutoregression(cumulated_arithmetic_returns, k_regimes=2, order=1, switching_variance=True)
            
        # Ajusta o modelo criado acima aos dados de retorno acumulado. Aqui, os parâmetros do modelo em questão são ajustados
        # para encontrar os regimes (ou estados) de volatilidade que melhor descrevem a série de retornos acumulados.
        markov_fit = markov_model.fit()

        # Define o modelo GARCH(1,1) aplicado aos retornos acumulados. O modelo GARCH (Generalized Autoregressive 
        # Conditional Heteroskedasticity) é configurado para capturar a volatilidade condicional. `p=1` e `q=1` especificam 
        # uma ordem GARCH(1,1), com uma defasagem de 1 para o termo autorregressivo e 1 para o termo de média móvel.
        garch_model = arch_model(cumulated_arithmetic_returns, vol='Garch', p=1, q=1)
        
        # Ajusta o modelo GARCH aos dados. Isto é, realiza o ajuste (estimativa) dos parâmetros do modelo GARCH usando os 
        # retornos acumulados, sem exibir a saída (`disp="off"`).
        garch_fit = garch_model.fit(disp="off")
        
        # print(garch_fit.summary())
        
        # Obtém as probabilidades suavizadas dos regimes do modelo de Markov. `smoothed_probs` retorna uma série com as 
        # probabilidades de cada ponto da série temporal estar em um determinado regime (ou estado).
        smoothed_probs = markov_fit.smoothed_marginal_probabilities

        # Obtém a volatilidade condicional do modelo GARCH. `conditional_volatility` fornece a volatilidade esperada 
        # em cada ponto da série temporal com base no modelo GARCH. Essa volatilidade muda ao longo do tempo, 
        # conforme estimada pelo modelo, e idealmente reflete o risco ou a variabilidade esperada.
        conditional_volatility = garch_fit.conditional_volatility

        return smoothed_probs, conditional_volatility
    
    def __init__(self, symbol: str, data_extraction_initial_date: datetime.date , data_extraction_final_date: datetime.date,
                 features_time_period: dict, strategy_time_period: int) -> None:
        '''
            Description:
                Inicializa uma instância da classe "Ticker" para análise financeira de um ativo específico, definindo os parâmetros
                necessários para a extração de dados históricos e configuração de períodos de cálculo de indicadores financeiros.
                Durante a inicialização, os dados históricos do ativo são baixados, validados e ajustados.
            Args:
                symbol (str): Símbolo do ativo financeiro (ex.: 'AAPL' para Apple Inc.) para extração de dados históricos.
                data_extraction_initial_date (datetime.date): Data inicial do período de extração dos dados históricos.
                data_extraction_final_date (datetime.date): Data final do período de extração dos dados históricos.
                features_time_period (dict): Dicionário com períodos de tempo para calcular indicadores financeiros, 
                                            como 'RSI' (Índice de Força Relativa), 'Bollinger Bands' (Bandas de Bollinger),
                                            'EMA' (Média Móvel Exponencial), entre outros indicadores.
                strategy_time_period (int): Número de dias a ser considerado como horizonte temporal para a estratégia de análise.
            Return:
                None: Este método inicializador não retorna valores, mas configura a instância com dados e parâmetros prontos para análise.
        '''
        
        # Define o símbolo do ticker.
        self.symbol = symbol
        
        # Define a data inicial para extração dos dados.
        self.data_extraction_initial_date = data_extraction_initial_date
        
        # Define a data final para extração dos dados.
        self.data_extraction_final_date = data_extraction_final_date
        
        # Define os períodos de tempo para calcular os indicadores financeiros.
        self.__features_time_period__ = features_time_period
        
        # Define o período de tempo para a estratégia de análise.
        self.__strategy_time_period__ = strategy_time_period
        
        # Baixa os dados do ticker.
        self.data = self.__get_ticker_data__()
        
        # Verifica se o ticker é válido.
        self.is_valid = self.__is_ticker_valid__()
        
        # Ajusta os dados para alinhamento e compatibilidade com os cálculos necessários.
        self.__adjust_data__()
        
        # Inicializa os atributos para armazenar os valores de treino e teste da variável alvo (Adj. Close) do modelo de ML que será criado.
        self.__y_train__ = None
        self.__y_test__ = None
        self.__y_test_scaled_sequences__ = None
    
    def __set_features__(self) -> None:
        '''
            Description:
                Calcula e adiciona indicadores técnicos ao DataFrame 'data' da instância. 
                Os indicadores calculados incluem retornos logarítmicos, Bandas de Bollinger (com média móvel exponencial), 
                Índice de Força Relativa (RSI), Média de Verdadeiro Alcance (ATR), Momentum e o Índice de Volatilidade (VIX).
                Cada indicador é calculado com base nos períodos especificados nos atributos da instância.
            Args:
                None: Esta função utiliza apenas os atributos do objeto e não recebe argumentos externos.
            Return:
                None: Esta função não retorna valores, mas modifica o DataFrame 'data' da instância ao adicionar colunas de indicadores técnicos.
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

        try:
            # Baixe os dados históricos do VIX
            vix_data = yf.download(vix_symbol, start=self.data_extraction_initial_date, 
                                end=self.data_extraction_final_date, interval=self.__features_time_period__['vix_time_period'], progress=False) 
            # OBS: Caso eu precise de algum log do vix, devo setar como True o parâmetro progress acima.
            
            # Registra uma mensagem de log indicando que o download foi bem-sucedido.
            logging.info(f"Dados do ^VIX baixados com sucesso para {self.symbol}.")
            
            # Calcula a variação percentual do VIX para um determinado número de períodos e a adiciona ao DataFrame.
            self.data['VIX'] = vix_data['Adj Close'].pct_change(periods=self.__features_time_period__['pct_change_period'])
        except Exception as e:
            # Em caso de erro durante o download, captura e registra a exceção no log.
            logging.error(f"Erro ao baixar os dados do ^VIX para {self.symbol}: {e}")

           
    
    def __remove_some_features__(self) -> None:
        '''
            Description:
                Remove algumas colunas do atributo data (DataFrame) que serão desnecessárias ('Open', 'High', 'Low', e 'Close').
            Args:
                Nenhum argumento é passado diretamente, pois a função utiliza os atributos do objeto.       
            Return:
                None: A função não retorna nada, mas modifica o atributo 'data' da instância.
        '''
        
        # Remove as colunas 'Open', 'High', 'Low' e 'Close' do DataFrame.
        self.data.drop(columns=['Open','High','Low','Close'], inplace=True)
    
    def __get_low_val_period_probability__(self):
        '''
            Description:
                Calcula e armazena a probabilidade média de períodos de baixa volatilidade para o ativo que representa a instância atual,
                utilizando as probabilidades suavizadas dos regimes de volatilidade (`smoothed_probs`) derivadas do modelo MSGARCH (Markov 
                Switching GARCH). Essa função divide as probabilidades suavizadas em blocos de tempo definidos por `__strategy_time_period__`
                e calcula a média das probabilidades em cada bloco. O resultado é uma série de probabilidades médias para períodos distintos,
                indicando a presença de baixa volatilidade.
            Args:
                None: A função utiliza os atributos do objeto e não recebe argumentos externos.
            Return:
                None: A função não retorna um valor explícito, mas armazena as probabilidades médias em `low_volatility_periods_probabilities`,
                como uma série pandas.
        '''
        
        # Inicializa uma lista para armazenar as probabilidades médias de períodos de baixa volatilidade.
        self.low_volatility_periods_probabilities = []
        
        # Itera sobre os dados de probabilidades suavizadas em blocos definidos pelo período da estratégia.
        for i in range(0,len(self.smoothed_probs),self.__strategy_time_period__):
            aux = 0 # Variável auxiliar para acumular as probabilidades dentro do bloco atual.
            # Soma as probabilidades de cada ponto dentro do bloco atual.
            for j in range(self.__strategy_time_period__):
                aux += self.smoothed_probs.iloc[(i+j),0]
                
            # Calcula a média de probabilidade de baixa volatilidade para o bloco e adiciona à lista de probabilidades médias.    
            self.low_volatility_periods_probabilities.append(aux/self.__strategy_time_period__)
        
        # Converte a lista de probabilidades médias para uma série pandas.
        self.low_volatility_periods_probabilities = pd.Series(self.low_volatility_periods_probabilities)

    def __adjust_data__(self) -> None:
        '''
            Description:
                Prepara os dados do ticker, calculando e adicionando os indicadores financeiros e realizando
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
            
            # Calcula as probabilidades suavizadas de regimes e a volatilidade condicional com o modelo MSGARCH. Armazena os resultados 
            # nas variáveis de instância 'smoothed_probs' e 'conditional_volatility'.
            self.smoothed_probs, self.conditional_volatility = self.__get_MSGARCH_results__()
            
            # Remove colunas específicas que não serão utilizadas em análises futuras.
            self.__remove_some_features__()
            
            # Remove todas as linhas que contenham valores nulos (NaN) no DataFrame.
            self.data.dropna(inplace=True)
            
            # Define o índice de corte: o índice mais alto inicial comum entre 'data', 'smoothed_probs' e 'conditional_volatility'. Esse corte 
            # é necessário para alinhar as séries temporais, permitindo que elas comecem a partir de uma data comum.
            cut_index = max(self.data.index[0],self.smoothed_probs.index[0],self.conditional_volatility.index[0])
            
            # Aplica o índice de corte para alinhar as séries temporais (data, smoothed_probs e conditional_volatility) a partir de uma data comum.
            self.data = self.data[self.data.index > cut_index]
            self.smoothed_probs = self.smoothed_probs[self.smoothed_probs.index > cut_index]
            self.conditional_volatility = self.conditional_volatility[self.conditional_volatility.index > cut_index]
            
            # Padroniza a quantidade de pontos nas séries temporais para ser múltipla de 5, facilitando estratégias de 5 ou 20 dias.
            self.data = self.data.iloc[:(len(self.data.index) - len(self.data.index) % 5)]
            self.smoothed_probs = self.smoothed_probs.iloc[:(len(self.smoothed_probs.index) - len(self.smoothed_probs.index) % 5)]
            self.conditional_volatility = self.conditional_volatility.iloc[:(len(self.conditional_volatility.index) - len(self.conditional_volatility.index) % 5)]
            
            # Calcula a probabilidade média de períodos de baixa volatilidade com base nos resultados obtidos a partir do modelo 
            # MSGARCH implementado anteriormente.
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
                tuple: Retorna quatro elementos - X_train (features de treino), X_test (features de teste), 
                       y_train (alvo de treino) e y_test (alvo de teste).
        '''
        
        # Separa as features (X) e o target (y).
        X = self.data.drop(columns=['Adj Close'])
        y = self.data['Adj Close']
        
        # Verifica se a data 'test_initial_day' não existe em 'self.data.index'.
        if(test_initial_day not in self.data.index):
            # Caso 'test_initial_day' não exista em 'self.data.index', substitui 'test_initial_day' pela data mais próxima presente 
            # em 'self.data.index'.
        
            # Define a data de negociação mais próxima disponível como 'new_test_initial_day'.
            new_test_initial_day = self.data.index[self.data.index.get_indexer([test_initial_day], method='nearest')[0]]
            
            # Atualiza 'test_initial_day' para a data de negociação mais próxima.
            test_initial_day = new_test_initial_day
            
            # Emite um aviso para o usuário informando sobre a substituição de 'test_initial_day' (comentado para evitar execução).
            #warnings.warn(f"A data {test_initial_day.date()} não é um dia de negociação válida. Por conta disso, ela será substituida pela data de negociação mais próxima, que no caso é {new_test_initial_day.date()}")
        
        # Verifica se a data 'test_final_day' não existe em 'self.data.index'.
        if(test_final_day not in self.data.index):
            # Caso 'test_final_day' não exista em 'self.data.index', substitui 'test_final_day' pela data mais próxima presente 
            # em 'self.data.index'.
        
            # Define a data de negociação mais próxima disponível como 'new_test_final_day'.
            new_test_final_day = self.data.index[self.data.index.get_indexer([test_final_day], method='nearest')[0]]
            
            # Atualiza 'test_final_day' para a data de negociação mais próxima.
            test_final_day = new_test_final_day
            
            # Emite um aviso para o usuário informando sobre a substituição de 'test_final_day' (comentado para evitar execução).
            #warnings.warn(f"A data {test_final_day.date()} não é um dia de negociação válida. Por conta disso, ela será substituida pela data de negociação mais próxima, que no caso é {new_test_final_day.date()}")
            
        # Separa o conjunto de séries temporais das features em dados de treino e dados de teste, com base nas dadas especificadas
        # (ou nas datas mais próximas a estas).
        X_train, X_test = X[X.index < test_initial_day], X[(X.index >= test_initial_day) & (X.index <= test_final_day)]
        
        # Separa a série temporal do target em dados de treino e dados de teste, com base nas dadas especificadas
        # (ou nas datas mais próximas a estas).
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
                de tamanho 'sequence_length' (Essas sequências são necessárias para treinar uma LSTM). 
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
    
        # Verifica se o sequence_length é maior que o número de amostras disponíveis.
        if sequence_length > len(X_train_scaled):
            raise ValueError("O 'sequence_length' não pode ser maior que o número de amostras em 'X_train_scaled'.")
        if sequence_length > len(X_test_scaled):
            raise ValueError("O 'sequence_length' não pode ser maior que o número de amostras em 'X_test_scaled'.")
        
        # Calcula o comprimento dos intervalos de treino e teste, assumindo que 'X_train' e 'y_train' têm o mesmo comprimento, 
        # tal como 'X_test' e 'y_test' (o que é esperado).
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
            Args:
                test_initial_day (datetime.date): Data que define o início do período de teste.
                test_final_day (datetime.date): Data que define o final do período de teste.
                lstm_time_sequences_length (int): O comprimento das sequências temporais usadas para treinar a LSTM.
            Return:
                tuple: Retorna quatro elementos - X_train_scaled_sequences, X_test_scaled_sequences, 
                    y_train_scaled_sequences, y_test_scaled_sequences (sequências temporais normalizadas para treino e teste).
            Observação: 
                A rede neural será treinada com todos os dados antes da data "test_initial_day" e fará predições para 
                todas as datas entre test_initial_day e test_final_day (inclusos).
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