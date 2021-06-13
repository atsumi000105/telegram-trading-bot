'''
df["Column_name"] - shoes all values for specific column
df.loc[0:2, 'Hobbyist':'Employment'] - display 3 rows for those columns

df['Close_dif'] = df['Close'].diff() - calcualte difference between rows in column
df['Diff_%'] = df['Open'].pct_change() - same but in %

df["Diff_%"].max() - getting min/max from column


df.set_index("Some_existing_column") - make column as index column
'''

