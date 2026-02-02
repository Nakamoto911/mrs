import pandas as pd
try:
    df = pd.read_csv('/Volumes/PRO-G40/Code/mrs/data/fred_md/FRED-MD_historic_appendix.csv')
    group_6 = df[df['group'] == 6]['fred'].tolist()
    print(f"Found {len(group_6)} series in Group 6")
    print(group_6[:10])
except Exception as e:
    print(e)
