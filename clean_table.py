from lxml import etree
from urllib.request import urlopen

import pandas as pd

DATA_URL = "https://en.wikipedia.org/w/index.php?title=Spacecraft_propulsion&oldid=760799107"


parser = etree.HTMLParser()
with urlopen(DATA_URL) as fp:
    all_html = etree.parse(fp, parser)

tables = [t for t in all_html.xpath(r"//table[@class='wikitable sortable']")]
table = tables[0]

# Clear sort values
for span in table.xpath(r"//span[@style='display:none']"):
    span.text = ""

with open("table_stripped_final.html", "w") as fp:
    print(etree.tostring(table, pretty_print=True).decode("utf-8"), file=fp)

# TODO: We could save an intermediate file here
data = pd.read_html("table_stripped_final.html")[0]
data.columns = data.iloc[0].values

# Remove top and bottom headers
data = data.iloc[1:-1]

# Use None values
data = data.replace(["?"], [None])

# Replace non-ASCII character
data = data.replace(["(.*) – (.*)"], [r"\1 ~ \2"], regex=True)

# Remove citation marks
data = data.replace([r"(.*)\[(full )?citation needed\]"], [r"\1"], regex=True)
data = data.replace(["([^\]]*)(\[.+\])+"], [r"\1"], regex=True)

# Final cleanup of corner cases
data.at[12, "Thrust (N)"] = "9/km2 ~ 230/km2"
data.at[12, "Technology readiness level"] = "5: Light-sail validated in lit vacuum\'"

data.at[25, "Technology readiness level"] = data.at[25, "Thrust (N)"]
data.at[25, "Thrust (N)"] = None
data.at[25, "Effective exhaust velocity (km/s)"] = None

data.at[31, "Technology readiness level"] = data.at[31, "Thrust (N)"]
data.at[31, "Thrust (N)"] = None
data.at[31, "Effective exhaust velocity (km/s)"] = None

# Export
data.to_csv("table.csv", index=False)

# Simplify LaTeX output
data["TRL"] = data.pop("Technology readiness level").replace(["(\d):.*"], [r"\1"], regex=True).astype(int)

# http://tex.stackexchange.com/a/63592/2488
with open("table_long.tex", "w") as fp:
    data.fillna("?").to_latex(fp, index=False, longtable=True)
