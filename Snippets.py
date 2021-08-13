
# save the df to h5
import pandas_datareader.data as web
import h5py

df = web.DataReader(name='SP500', data_source='fred', start=2009).squeeze()
print(df.info())
hf = h5py.File('SP500.h5', 'w')
hf.create_dataset('dataset', data=df)
