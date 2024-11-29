from sqlalchemy import create_engine
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

try:
    print('Obtendo Dados...')
    load_dotenv()
    
    host = os.getenv('DB_HOST')
    port = os.getenv('DB_PORT')
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    database =os.getenv('DB_DATABASE')

    engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}')

    tb_roubo = pd.read_sql('basedp_roubo_rua', engine)

    tb_dp = pd.read_sql('basedp', engine)
except ImportError as e:
    print(f'Erro: {e}')
    exit()

try:
    df_roubodp = pd.merge(tb_roubo, tb_dp, on='cod_ocorrencia')
    # print(df_roubodp.head(20))
    df_roubodp = df_roubodp[(df_roubodp['ano'] >= 2022) & (df_roubodp['ano'] <= 2023)]
    df_roubodp = df_roubodp.groupby(['aisp']).sum(['roubo_rua']).reset_index()
    array_rouboderua = np.array(df_roubodp['roubo_rua'])

    media_roubo = np.mean(array_rouboderua)
    mediana_roubo = np.median(array_rouboderua)
    distancia_roubo = abs((media_roubo - mediana_roubo) / mediana_roubo) * 100

    maximo = np.max(array_rouboderua)
    minimo = np.min(array_rouboderua)
    amplitude = maximo - minimo

    q1 = np.quantile(array_rouboderua, 0.25, method='weibull')
    q2 = np.quantile(array_rouboderua, 0.50, method='weibull')
    q3 = np.quantile(array_rouboderua, 0.75, method='weibull')
    iqr = q3 - q1
    limite_superior = q3 + (1.5 * iqr)
    limite_inferior = q1 - (1.5 * iqr)

    df_roubodp_outliers_sup = df_roubodp[df_roubodp['roubo_rua'] > limite_superior]
    df_roubodp_outliers_inf = df_roubodp[df_roubodp['roubo_rua'] < limite_inferior]

    assimetria = df_roubodp['roubo_rua'].skew()
    curtose = df_roubodp['roubo_rua'].kurtosis()

    print('\nMEDIDAS DE TENDÊNCIA CENTRAL')
    print(30*'=')
    print(f'MÉDIA: {media_roubo:.2f}')
    print(f'MEDIANA: {mediana_roubo:.2f}')
    print(f'DISTÂNCIA: {distancia_roubo:.2f}%')

    print('\nMedidas de Posição e Dispersão')
    print(30*'-')
    print(f'Menor valor: {minimo}')
    print(f'Limite inferior: {limite_inferior}')
    print(f'Q1: {q1}')
    print(f'Q3: {q3}')
    print(f'Limite superior: {limite_superior}')
    print(f'Maior valor: {maximo}')
    print(f'IQR: {iqr}')
    print(f'Amplitude total: {amplitude}')

    print('\nDPs com roubos de rua superiores as demais:')
    print(30*'-')
    if len(df_roubodp_outliers_sup) == 0:
        print('Não existem DPs com valores discrepantes supreiores')
    else:
        print(df_roubodp_outliers_sup.sort_values(
            by='roubo_rua', ascending=False))

    print('\nDPs com roubos de rua inferiores as demais:')
    print(30*'-')
    if len(df_roubodp_outliers_inf) == 0:
        print('Não existem DPs com valores discrepantes inferiores')
    else:
        print(df_roubodp_outliers_inf.sort_values(
            by='roubo_rua', ascending=True))

    # DPS que menos recuperaram veículos
    df_rouboderua_q1 = df_roubodp[df_roubodp['roubo_rua'] < q1]

    print('\nDPs com menos ocorrências de roubo de rua:')
    print(30*'-')
    print(df_rouboderua_q1.sort_values(
        by='roubo_rua', ascending=True))
    
except ImportError as e:
    print(f'Erro: {e}')
    exit()

try:
    plt.subplots(2, 2, figsize=(16, 7))
    plt.suptitle('Análise das ocorrências de roubos de rua no RJ')

    plt.subplot(2, 2, 1)    
    plt.boxplot(array_rouboderua, vert=False, showmeans=True)
    plt.title('Roubos de rua')

    plt.subplot(2, 2, 2)
    plt.hist(array_rouboderua, bins=50, edgecolor='black')
    plt.axvline(media_roubo, color='g', linewidth=1)
    plt.axvline(mediana_roubo, color='y', linewidth=1)
    plt.title('Histograma Roubos de Rua')

    plt.subplot(2, 2, 3)
    plt.text(0.1, 0.9, f'Média: {media_roubo:.2f}', fontsize=12)
    plt.text(0.1, 0.8, f'Mediana: {mediana_roubo:.2f}', fontsize=12)
    plt.text(0.1, 0.7, f'Distância: {distancia_roubo:.2f}%', fontsize=12)
    plt.text(0.1, 0.6, f'Menor valor: {minimo}', fontsize=12) 
    plt.text(0.1, 0.5, f'Limite inferior: {limite_inferior}', fontsize=12)
    plt.text(0.1, 0.4, f'Q1: {q1}', fontsize=12)
    plt.text(0.1, 0.3, f'Q3: {q3}', fontsize=12)
    plt.text(0.1, 0.2, f'Limite superior: {limite_superior}', fontsize=12)
    plt.text(0.1, 0.1, f'Maior valor: {maximo}', fontsize=12)
    plt.text(0.1, 0.0, f'Amplitude Total: {amplitude}', fontsize=12)
    plt.text(0.1, 1.0, f'IQR: {iqr}', fontsize=12)
    plt.title('Medidas Observadas')

    plt.axis('off')
    plt.tight_layout()

    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.5, f'Assimetria: {assimetria:.2f}', fontsize=12)
    plt.text(0.1, 0.4, f'Curtose: {curtose:.2f}', fontsize=12)
    plt.title('Impressão de Medidas Estatísticas')
    
    plt.axis('off')
    plt.tight_layout()
    
    plt.show()

except ImportError as e:
    print(f'Erro ao visualizar dados: {e}')
    exit()