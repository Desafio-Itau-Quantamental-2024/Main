#
from Ticker import Ticker
#
import numpy as np
#
from datetime import datetime
#
from concurrent.futures import ProcessPoolExecutor

class Tickers:
    '''
        Description:
            A classe "Tickers" serve como um contêiner para armazenar e padronizar múltiplos objetos "Ticker", garantindo que todos os
            dados extraídos dos tickers tenham o mesmo intervalo de datas de negociação. Isso é útil para análises financeiras que
            exigem comparabilidade entre diferentes ativos ao longo de um período comum. A classe também valida os tickers e armazena
            apenas aqueles que possuem dados válidos.
            
            FALAR SOBRE A UTILIDADE DESSA CLASSE EM PARALELIZAR OS PROCESSOS DE OBTENÇÃO E AJUSTE DE CADA TICKER
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
        
        # Itera sobre cada um dos símbolos presentes em "tickers_list".
        for i, symbol in enumerate(self.symbols_list):
            # Cria um objeto "Ticker" para o símbolo atual.
            ticker = Ticker(symbol, self.data_extraction_initial_date, self.data_extraction_final_date, self.features_time_period, 
                            self.strategy_time_period)
            # Se o ticker for válido, salva o objeto "Ticker" no array "data".
            if(ticker.is_valid): self.symbols[i] = ticker
    
    def __init__(self, symbols_list: list, data_extraction_initial_date: datetime.date,
                 data_extraction_final_date: datetime.date, features_time_period: dict, 
                 strategy_time_period: int, regime_probability_threshold: float, number_of_tickets_to_be_traded: int) -> None:
        '''
            Description:
                Inicializa a classe "Tickers", criando uma lista de objetos "Ticker" com base na lista de símbolos e nas datas de extração
                fornecidas. A classe também armazena os objetos "Ticker" válidos no atributo "data".
            
            Args:
                symbols_list (list): A lista de símbolos (tickers) para os quais os dados serão extraídos.
                data_extraction_initial_date (datetime.date): A data inicial para a extração dos dados.
                data_extraction_final_date (datetime.date): A data final para a extração dos dados.
                features_time_period (dict): Dicionário contendo ... 
                
            Return:
                None: A função não retorna nada, mas inicializa a instância com os tickers válidos e seus dados.
        '''
        
        # Cria um array que guardará os objetos "Ticker" válidos.
        self.symbols =  np.empty(len(symbols_list), dtype=object)

        # Define a lista de símbolos de tickers.
        self.symbols_list = symbols_list
        
        # Define a data inicial para a extração dos dados.
        self.data_extraction_initial_date = data_extraction_initial_date
        
        # Define a data final para a extração dos dados.
        self.data_extraction_final_date = data_extraction_final_date
        
        #
        self.features_time_period = features_time_period
        
        #
        self.strategy_time_period = strategy_time_period
        
        #
        self.regime_probability_threshold = regime_probability_threshold
        
        #
        self.number_of_tickets_to_be_traded = number_of_tickets_to_be_traded
        
        # Preenche o array "tickers" com os tickers válidos.
        self.__get_tickers_data__()
        
        # Remove os valores nulos (None) do array "symbols".
        self.symbols = self.symbols[self.symbols != None]
        
    def get_tickers_to_trade(self):
        '''
            Description:

            Args:

            Return:

        '''
        
        # Assumo que todos os symbols tem o mesmo número de períodos
        periods = len(self.symbols[0].low_volatility_periods_probabilities)

        tickers_to_trade = []

        for period in range(periods):
            periods_results = []
            for index in range(len(self.symbols)):
                #
                if self.symbols[index].low_volatility_periods_probabilities[period] >= self.regime_probability_threshold:
                    period_results = {
                        "ticker_index": index,
                        "ticker_conditional_volatility": self.symbols[index].conditional_volatility[period]
                    }
                    #
                    periods_results.append(period_results)
            #    
            periods_results.sort(key=lambda x: x["ticker_conditional_volatility"])
            #    
            tickers_to_trade.append([element["ticker_index"] for element in periods_results[:self.number_of_tickets_to_be_traded]]) # Passar só o nome dos tickers ou o índice na lista tickers
            
        return tickers_to_trade