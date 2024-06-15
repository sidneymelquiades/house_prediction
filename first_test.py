

import requests
from bs4 import BeautifulSoup

URL = "https://emcasa.com/imoveis/sp/sao-paulo?pagina=1"
headers = {'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.152 Safari/537.36' }
r = requests.get(URL)
soup = BeautifulSoup(r.content, 'html5lib') # If this line causes an error, run 'pip install html5lib' or install html5lib
#print(soup.prettify())

anuncios=soup.findAll("p",class_="Typography_ecTypographyParagraph__fYHaQ Typography_ecTypographyBold__qPve0")
print(anuncios[2].contents[0].getText())