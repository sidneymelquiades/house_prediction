##### primeiro teste ##############
import requests
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep

for pag in range(1):
    URL = f"https://emcasa.com/imoveis/sp/sao-paulo?pagina={pag}"
    headers = {'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.152 Safari/537.36' }
    r = requests.get(URL)
    soup = BeautifulSoup(r.content) # If this line causes an error, run 'pip install html5lib' or install html5lib
    #print(soup.prettify())

    precos=soup.findAll("p",class_="Typography_ecTypographyParagraph__fYHaQ Typography_ecTypographyBold__qPve0")
    tipografias=soup.findAll("div",class_="ListingCard_ecListingCardList__TypiF")

    list_df=[]
    for i in range(len(precos)):
        try:
            if precos[i].contents[0].getText().startswith("R$"):
                preco= precos[i].contents[0].getText().replace('R$','').replace(" ", "")
            else:
                continue
        except:
            continue
        try:
            area=tipografias[i].contents[0].getText().replace('m²','').replace('área','').replace(' ','')
        except:
            continue
        try:
            quartos=tipografias[i].contents[1].getText()
        except:
            quartos=str(0)
        try:
            suites=tipografias[i].contents[2].getText()
        except:
            suites=str(0)
        try:
            vagas=tipografias[i].contents[3].getText()
        except:
            vagas=str(0)
        list_row=[preco, area, quartos, suites, vagas]
        list_df.append(list_row)
    sleep(10)

df = pd.DataFrame(list_df, columns=['preco', 'area', 'quartos', 'suites', 'vagas'])
print(df)