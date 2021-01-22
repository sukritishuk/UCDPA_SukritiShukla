# UCD PA Course1 - Final Assignment Python Code:
# BRICS Members' - Human Development Index Indicators

## Getting Data from Human Development Index API for all the 3 Factor Indicators -
# importing the library:
import requests
# invoking a get request to the Human Development Index API url:
response = requests.get('http://hdr.undp.org/en/countries')
# printing the response status generated after making get request:
print(response)


## Step 1 - Getting data from API & importing it into Pandas DataFrame:
# importing the library:
import pandas as pd
# creating a Function to get data from the API using a url for each of the 3 indicators:
def get_data(url_1, url_2, url_3):
    response_1 = requests.get(url_1)
    response_2 = requests.get(url_2)
    response_3 = requests.get(url_3)
# putting the url data into a json format:
    data_json_1 = response_1.json()
    data_json_2 = response_2.json()
    data_json_3 = response_3.json()
# importing the data from json file into Pandas DataFrame:
    ind_dict_1 = pd.DataFrame(data_json_1['indicator_value'].items())
    ind_dict_2 = pd.DataFrame(data_json_2['indicator_value'].items())
    ind_dict_3 = pd.DataFrame(data_json_3['indicator_value'].items())
# renaming the columns of each of the Pandas DataFrames:
    ind_dict_1.rename(columns={0: 'Country_Code', 1: 'Indicator'}, inplace=True)
    ind_dict_2.rename(columns={0: 'Country_Code', 1: 'Indicator'}, inplace=True)
    ind_dict_3.rename(columns={0: 'Country_Code', 1: 'Indicator'}, inplace=True)
# merging the DataFrames into one combined DataFrame:
    comb_dict = ind_dict_1.merge(ind_dict_2, on='Country_Code', how='outer', suffixes=('_1', '_2'))
    combined_dict = comb_dict.merge(ind_dict_3, on='Country_Code', how='left')
    combined_dict.rename(columns={'Indicator': 'Indicator_3'}, inplace=True)

    return combined_dict

