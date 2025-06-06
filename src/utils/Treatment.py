import re
import requests
from googlesearch import search
from bs4 import BeautifulSoup

def diseaseDetail(term):
    diseases = [term]
    ret = term + "\n"
    for dis in diseases:
        query = dis + ' wikipedia'
        for sr in search(query, tld="co.in", stop=10, pause=0.5): 
            match = re.search(r'wikipedia', sr)
            filled = 0
            if match:
                wiki = requests.get(sr, verify=False)
                soup = BeautifulSoup(wiki.content, 'html5lib')
                info_table = soup.find("table", {"class":"infobox"})
                if info_table is not None:
                    for row in info_table.find_all("tr"):
                        data = row.find("th", {"scope": "row"})
                        if data is not None:
                            symptom = str(row.find("td"))
                            symptom = symptom.replace('.', '')
                            symptom = symptom.replace(';', ',')
                            symptom = symptom.replace('<b>', '<b> \n')
                            symptom = re.sub(r'<a.*?>', '', symptom)
                            symptom = re.sub(r'</a>', '', symptom)
                            symptom = re.sub(r'<[^<]+?>', ' ', symptom)
                            symptom = re.sub(r'\[.*\]', '', symptom)
                            symptom = symptom.replace("&gt", ">")
                            ret += data.get_text() + " - " + symptom + "\n"
                            filled = 1
                if filled:
                    break
    return ret
