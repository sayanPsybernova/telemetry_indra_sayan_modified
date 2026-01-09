import json
import pandas as pd

# Check session output
with open('output/sessions.json') as f:
    data = json.load(f)
action = data['sessions'][0]['actions'][0]
print('Fields in action:', list(action.keys()))
print('Full action:', action)

# Check raw CSV
df = pd.read_csv('chaitali_all_data.csv', nrows=3)
print('\nCSV columns:', list(df.columns))
print(df.head())
