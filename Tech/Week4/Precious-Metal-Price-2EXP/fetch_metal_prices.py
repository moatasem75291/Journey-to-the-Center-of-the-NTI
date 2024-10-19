# fetch_gold_prices.py
import requests
from bs4 import BeautifulSoup
import pandas as pd


def fetch_metal_prices(metal: str = "gold") -> pd.DataFrame | None:

    url = f"https://goldbroker.com/charts/{metal}-price#live-chart"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        html_content = response.content
        soup = BeautifulSoup(html_content, "html.parser")
        table = soup.find_all("table", {"class": "table bg-secondary"})[0]

        data = []
        headers = [header.text.strip() for header in table.find_all("th")]

        for row in table.find_all("tr")[1:]:
            cols = [col.text.strip() for col in row.find_all("td")]
            data.append(cols)

        df = pd.DataFrame(data, columns=headers)

        return df
    else:
        return None
