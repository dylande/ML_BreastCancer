#imports
import pandas as pd

#import the data.
data = pd.read_csv('book_data.csv')

#preview the first 5 lines of the data.
data[['book_desc', 'genres']]
