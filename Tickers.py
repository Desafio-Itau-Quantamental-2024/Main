# Importa o módulo subprocess, que será utilizado para instalar bibliotecas caso não estejam presentes.
import subprocess
# Importa o módulo sys, que fornecerá o caminho do interpretador para instalar pacotes via pip.
import sys

# Importa o módulo datetime da biblioteca padrão para manipulação de datas e horas.
from datetime import datetime, timedelta

# Importa o módulo ProcessPoolExecutor da biblioteca concurrent.futures para realizar a 
# execução de tarefas em paralelo utilizando múltiplos processos.
from concurrent.futures import ProcessPoolExecutor # (A paralelização ainda será implementada)

# Importa a classe Ticker, que será utilizada para obter e manipular dados financeiros de um ativo específico.
from Ticker import Ticker

# Verifica se a biblioteca numpy está instalada; caso contrário, a instala.
try:
    import numpy as np
except ImportError:
    print("Biblioteca 'numpy' não encontrada. Instalando...")
    # Instala a biblioteca "numpy" que será usada para operações numéricas.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    
# Verifica se a biblioteca tqdm está instalada; caso contrário, a instala.
try:
    from tqdm.notebook import tqdm # Remover essa biblioteca caso eu precise de logs da criação do objeto "Tickers".
except ImportError:
    print("Biblioteca 'numpy' não encontrada. Instalando...")
    # Instala a biblioteca "tqdm" que será usada para a exibição de barras de carregamento.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])

class Tickers:
    '''
        Description:
            A classe "Tickers" serve como um "contêiner" para armazenar e padronizar múltiplos objetos "Ticker", garantindo que todos os
            dados extraídos dos tickers tenham o mesmo intervalo de datas de negociação. Além disso, a classe "Tickers" também é útil para
            filtrar, por período, os objetos "Ticker" com maior probabilidade de estarem em regime de baixa volatilidade.
    '''
    
    def __get_tickers_data__(self) -> None:
        '''
            Description:
                Itera sobre a lista de símbolos de tickers (tickers_list), cria um objeto Ticker para cada símbolo e salva os dados 
                no atributo "data" caso o ticker seja válido.       
            Args:
                Nenhum argumento é passado diretamente, pois a função utiliza os atributos do objeto.      
            Return:
                None: A função não retorna nada, mas preenche o array "data" com os objetos "Ticker" válidos.
        '''
        
        # Itera sobre cada um dos símbolos presentes em "symbols_list" com barra de progresso
        for i, symbol in enumerate(tqdm(self.symbols_list, desc="Baixando e processando os tickers")):
            # Cria um objeto "Ticker" para o símbolo atual
            ticker = Ticker(symbol, self.data_extraction_initial_date, self.data_extraction_final_date, 
                            self.features_time_period, self.strategy_time_period)
            
            # Se o ticker for válido, salva o objeto "Ticker" no atributo "symbols"
            if ticker.is_valid:
                self.symbols[i] = ticker
            
    def __init__(self, symbols_list: list, data_extraction_initial_date: datetime.date,
                 data_extraction_final_date: datetime.date, features_time_period: dict, 
                 strategy_time_period: int, regime_probability_threshold: float, number_of_tickets_to_be_traded: int) -> None:
        '''
            Description:
                Inicializa a classe "Tickers", criando uma lista de objetos "Ticker" com base na lista de símbolos fornecida e nas 
                datas de extração. A classe armazena apenas os objetos "Ticker" válidos no atributo "symbols" após filtrar os dados.
            Args:
                symbols_list (list): Lista de símbolos (tickers) cujos dados serão extraídos.
                data_extraction_initial_date (datetime.date): Data inicial para a extração dos dados.
                data_extraction_final_date (datetime.date): Data final para a extração dos dados.
                features_time_period (dict): Dicionário contendo os períodos de tempo para diferentes indicadores, 
                                             usado no cálculo de indicadores e outras características relevantes para cada ticker.
                strategy_time_period (int): Número de dias de negociação que cada período de tempo da estratégia de negociação terá.
                regime_probability_threshold (float): Probabilidade limite para identificar regimes de negociação, isto é, todas as ações 
                                                      que tiverem probabilidade média maior que esse limite estarão aptas a serem negociadas
                                                      no período em questão.
                number_of_tickets_to_be_traded (int): Número máximo de símbolos (tickers) que podem ser incluídos em um único período de negociação.
            Return:
                None: Esta função inicializa a instância com os tickers válidos e seus dados, sem retornar valores.
        '''
        
        # Inicializa um array para armazenar os objetos "Ticker" válidos.
        self.symbols =  np.empty(len(symbols_list), dtype=object)

        # Define a lista de símbolos para extração de dados.
        self.symbols_list = symbols_list
        
        # Define a data de início para extração de dados.
        self.data_extraction_initial_date = data_extraction_initial_date
        
        # Define a data final para extração de dados.
        self.data_extraction_final_date = data_extraction_final_date
        
        # Define o dicionário com períodos específicos para cálculos de indicadores.
        self.features_time_period = features_time_period
        
        # Define o período de tempo da estratégia de negociação.
        self.strategy_time_period = strategy_time_period
        
        # Define o limiar para a probabilidade de estar em um regime de baixa volatilidade (todas as ações que tiverem probabilidade média no
        # período em questão maior que esse limiar, serão consideradas aptas a serem negociadas).
        self.regime_probability_threshold = regime_probability_threshold
        
        # Define o número máximo de tickers que poderão ser negociados em um único período.
        self.number_of_tickets_to_be_traded = number_of_tickets_to_be_traded
        
        # Preenche o array "tickers" com os tickers válidos.
        self.__get_tickers_data__()
        
        # Remove os valores nulos (None) do array "symbols".
        self.symbols = self.symbols[self.symbols != None]
        
    def get_tickers_to_trade(self) -> list[list]:
        '''
            Description:
                Seleciona os tickers que têm alta probabilidade de estarem em regime de baixa volatilidade, com base no limiar de probabilidade
                definido (`regime_probability_threshold`). Para cada período analisado, retorna os tickers classificados pela menor volatilidade
                condicional, até o limite estabelecido pelo número de ativos a serem negociados (`number_of_tickets_to_be_traded`).
            Args:
                Nenhum argumento é passado diretamente, pois a função utiliza os atributos do objeto.      
            Return:
                list[lists]: Uma lista de listas, onde cada sublista contém os índices dos tickers selecionados para trading em cada período.
        '''
        
        # Assumimos que todos os tickers têm o mesmo número de períodos de previsão.
        periods = len(self.symbols[0].low_volatility_periods_probabilities)

        # Lista que armazena os tickers selecionados para cada período.
        tickers_to_trade = []

        for period in range(periods):
            # Lista para armazenar resultados do período atual.
            periods_results = []
            for index in range(len(self.symbols)):
                # Verifica se a probabilidade de estar em um regime de baixa volatilidade do ticker respeita o limiar definido.
                if self.symbols[index].low_volatility_periods_probabilities[period] >= self.regime_probability_threshold:
                    # Cria um dicionário com o índice do ticker e sua volatilidade condicional.
                    period_results = {
                        "ticker_index": index,
                        "ticker_conditional_volatility": self.symbols[index].conditional_volatility[period]
                    }
                    # Caso o fluxo de execução chegue até aqui, significa que, para o período em questão, o ticker atual possui uma 
                    # probabilidade de estar em um regime de baixa volatilidade que respeita o limiar definido. Por conta disso, tal ticker
                    # será adicionado a lista "periods_results", indicando que ele é um ticker candidato a ser negociado no período em questão.
                    periods_results.append(period_results)
                    
            # Ordena os tickers pela volatilidade condicional em ordem crescente, no intuito de priorizar os de menor volatilidade 
            # para negociação.
            periods_results.sort(key=lambda x: x["ticker_conditional_volatility"])
            
            # Seleciona para negociação no período em questão os tickers com menor volatilidade condicional, limitados pelo atributo
            # "number_of_tickets_to_be_traded", e armazena seus índices.
            tickers_to_trade.append([element["ticker_index"] for element in periods_results[:self.number_of_tickets_to_be_traded]])
            
        return tickers_to_trade