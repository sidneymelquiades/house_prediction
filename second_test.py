import requests
from bs4 import BeautifulSoup
from time import sleep


import pandas as pd

# Criar o DataFrame vazio com as colunas especificadas
df = pd.DataFrame(columns=[
    'valor',
    'metragem',
    'nquartos',
    'nsuites',
    'nchuveiros',
    'nelevador',
    'nvagas',
    'condominio',
    'proximometro'
])


def coletar_informacoes(url):
    headers = {
        'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.152 Safari/537.36'}
    response = requests.get(url)
    sleep(1)
    content = response.content
    soup = BeautifulSoup(content, 'html.parser')
    container = soup.find(class_='Imovel_ecInfosContainer___s6Zx')
    valor = soup.find(class_='ValueBox_ecValueBoxHeaderPrice__8x0rb')
    try:
        valor = valor.get_text(strip=True).replace(u"\xa0", ' ').replace("R$ ", '').replace(",00", '').replace(".", '')
    except:
        valor = 0

    try:
        container = container.get_text(strip=True).split("Ícone: ")
    except:
        container = []

    return [valor, container]

for pag in range(500):
# URL da página inicial
    base_url_pag = f'https://emcasa.com/imoveis/sp/sao-paulo?pagina={pag}'  # Substitua pela URL real
    base_url="https://emcasa.com/"
    # Faz a requisição para obter o conteúdo da página inicial
    response = requests.get(base_url_pag)
    content = response.content

    # Usa o BeautifulSoup para analisar o HTML
    soup = BeautifulSoup(content, 'html.parser')

    # Encontra todos os links que começam com /imoveis
    links = soup.find_all('a', href=True)
    imoveis_links = [link['href'] for link in links if link['href'].startswith('/imoveis')]
    imoveis_links= list(set([s for s in imoveis_links if 'id-' in s]))


    # Função para coletar informações de cada link


    # Coleciona as informações de todas as páginas de imóveis
    imoveis_info = []
    for link in imoveis_links:
        # Constrói a URL completa
        full_url = base_url + link
        info = coletar_informacoes(full_url)
        try:
            if len(info[1])>10:
                dict_info={

                    'valor' : int(info[0]),
                    'metragem': int(info[1][1].replace('áreaMetragemÁrea total', '').replace('m²','')),
                    'nquartos' : int(info[1][2].replace('camaTotal de quartosInclui suíte','')),
                    'nsuites': int(info[1][3].replace('camaSuítesQuarto com banheiro','')),
                    'nchuveiros': int(info[1][4].replace('chuveiroBanheiros sociaisExclui lavabo e suíte','')),
                    'nelevador': int(info[1][6].replace('elevadorElevador', '')),
                    'nvagas': int(info[1][9].replace('carroVagas de garagem', '')),
                    'condominio': int(info[1][8].replace('casaValor do condomínioValor mensalR$', '').replace(u'\xa0', '').replace(',00', '').replace('.', '')),
                    'proximometro': info[1][10][-3:]


                }
                df = pd.concat([df, pd.DataFrame([dict_info])], ignore_index=True)
        except:
            pass


print(df)
df.to_csv('./results.csv')