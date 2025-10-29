import pandas as pd
import plotly.express as px

# 1. Load BTS Freight Analysis Framework (FAF5)
url = "https://data.bts.gov/resource/dv3p-7ye6.json?$limit=50000"
df = pd.read_json(url)

# 2. Clean and transform
df = df[['year', 'mode', 'tons', 'value_2017_dollars']].dropna()
df['year'] = pd.to_numeric(df['year'])
df['tons'] = pd.to_numeric(df['tons'])
df['value_2017_dollars'] = pd.to_numeric(df['value_2017_dollars'])

# 3. Aggregate by mode and year
summary = df.groupby(['year', 'mode'], as_index=False).agg({
    'tons': 'sum',
    'value_2017_dollars': 'sum'
})

# 4. Plot freight volume by mode
fig1 = px.line(summary,
               x='year', y='tons', color='mode',
               title='U.S. Freight Volume Trends by Mode (FAF5)',
               labels={'tons': 'Tons (Billions)', 'year': 'Year'})
fig1.write_html('freight_tons_trend.html')
fig1.show()

# 5. Plot total value by mode
fig2 = px.line(summary,
               x='year', y='value_2017_dollars', color='mode',
               title='U.S. Freight Value by Mode (2017 Dollars)',
               labels={'value_2017_dollars': 'Value (2017 USD)', 'year': 'Year'})
fig2.write_html('freight_value_trend.html')
fig2.show()

# 6. Identify top modes by tonnage (latest year)
latest_year = summary['year'].max()
top_modes = summary[summary['year'] == latest_year].sort_values('tons', ascending=False)
print(f"\n📊 Top freight modes in {latest_year}:")
print(top_modes[['mode', 'tons', 'value_2017_dollars']].head())
