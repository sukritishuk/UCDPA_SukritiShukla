
### Data Source 1 - Data from Kaggle Public Dataset: Country Statistics – UNData:

## Step 1 - Importing csv File into Pandas DataFrame and understanding the data therein -
# loading Pandas library (alias pd):
import pandas as pd
# reading data from the csv file and storing it into the variable data:
data = pd.read_csv('country_profile_variables.csv')
# studying the data size (rows & columns) using the shape attribute:
print(data.shape)
# getting a snapshot of first 5 rows of data using the head() method:
print(data.head())
# checking if any value in the dataset is missing (detect NaN values):
print(data.isnull().values.any())


## Step 2 - Slicing the entire dataset for BRICS member countries data only -
# creating an array of BRICS member countries:
BRICS_members = ['Brazil','Russian Federation','India','China','South Africa']
# slicing the entire dataset rows for those where country column value is in the above iterable array:
BRICS_country_stats = data.loc[data['country'].isin(BRICS_members)]
BRICS_country_stats
# slicing only demographic & income indicators columns for BRICS member countries:
BRICS_country_stats2 = BRICS_country_stats[['country','Region','Population in thousands (2017)','Population density (per km2, 2017)','GDP per capita (current US$)']]
# setting the index for the BRICS member countries DataFrame:
BRICS_country_stats2.set_index('country',inplace = True)
# printing the DataFrame:
print(BRICS_country_stats2)


## Step 3 - Slicing the data for the respective Regions of each BRICS member countries only-
# loading the Numpy library (alias np):
import numpy as np
# grouping the entire dataset by "Region" and storing it in variable grouped2:
grouped2 = data.groupby(['Region'])
# slicing the grouped dataset for 3 indicators and aggregating each of them by sum and mean values:
# slicing the population density column & aggregating for sum & mean; also rounding to 2 decimal places
pop_density_group = grouped2['Population density (per km2, 2017)'].agg([np.sum,np.mean]).round(2)
# slicing the GDP per capita column & aggregating for sum & mean; also rounding to 2 decimal places
GDP_per_capita_group = grouped2['GDP per capita (current US$)'].agg([np.sum,np.mean]).round(2)
# slicing the total population column & aggregating for sum & mean; also rounding to 2 decimal places
pop_group = grouped2['Population in thousands (2017)'].agg([np.sum,np.mean]).round(2)

# indexing and slicing regional statistics for respective Regions of BRICS member countries only:
# indexing and slicing total population data:
regional_tot_pop = pop_group.loc[['EasternAsia','EasternEurope','SouthAmerica','SouthernAfrica','SouthernAsia']]
# indexing and slicing population density data:
regional_pop_density = pop_density_group.loc[['EasternAsia','EasternEurope','SouthAmerica','SouthernAfrica','SouthernAsia']]
# indexing and slicing GDP per capita data:
regional_GDP_percap = GDP_per_capita_group.loc[['EasternAsia','EasternEurope','SouthAmerica','SouthernAfrica','SouthernAsia']]

# joining regional statistics DataFrames for tot_pop, pop_density and GDP per cap and storing it in variable - regional_stats:
regional_comb = regional_tot_pop.join(regional_pop_density,lsuffix='_tot_pop',rsuffix='_pop_dens')
regional_stats = regional_comb.join(regional_GDP_percap,lsuffix='_pop_dens',rsuffix='_GDP_percap')

# renaming the columns for the combined Regional statistics DataFrame as regional_stats:
regional_stats.rename(columns={'sum':'sum_GDP_percap','mean': 'mean_GDP_percap'},inplace=True)
# printing the DataFrame:
print(regional_stats)


## Chart 1 - Visualizing Dempgraphic & Income Indicators of BRICS member's vs. their respective Regions in Bar Chart Subplots:
# loading Matplotlib library (alias as plt):
import matplotlib.pyplot as plt
# creating a Figure and Axes objects (2 rows and 3 columns) and defining the figure size of the chart:
fig,ax = plt.subplots(2,3,figsize=(15,8))

# plotting the Regional Graphs on Bottom row of the Subplot for all 3 indicators and defining color of each country's region bar:
ax[1][0].bar(regional_stats.index,regional_stats['sum_tot_pop'],color=['black', 'red', 'green', 'blue', 'cyan'])
ax[1][1].bar(regional_stats.index,regional_stats['sum_pop_dens'],color=['black', 'red', 'green', 'blue', 'cyan'])
ax[1][2].bar(regional_stats.index,regional_stats['sum_GDP_percap'],color=['black', 'red', 'green', 'blue', 'cyan'])

# plotting each BRICS member country Graphs on Top row of the Subplot for all 3 indicators and defining the color of each country same as its region above:
ax[0][0].bar(BRICS_country_stats2.index,BRICS_country_stats2['Population in thousands (2017)'],color=['green','black','cyan','red','blue'])
ax[0][1].bar(BRICS_country_stats2.index,BRICS_country_stats2['Population density (per km2, 2017)'],color=['green','black','cyan','red','blue'])
ax[0][2].bar(BRICS_country_stats2.index,BRICS_country_stats2['GDP per capita (current US$)'],color=['green','black','cyan','red','blue'])

# labeling the x-axes ticks for each Regional subplots:
ax[1][0].set_xticklabels(regional_stats.index,rotation=25)
ax[1][1].set_xticklabels(regional_stats.index,rotation=25)
ax[1][2].set_xticklabels(regional_stats.index,rotation=25)

# labeling the x-axes ticks for each BRICS member subplots:
ax[0][0].set_xticklabels(BRICS_country_stats2.index,rotation=25)
ax[0][1].set_xticklabels(BRICS_country_stats2.index,rotation=25)
ax[0][2].set_xticklabels(BRICS_country_stats2.index,rotation=25)

# labeling the y-axes for each of the subplots (both regional & BRICS members):
ax[1][0].set_ylabel('Population in thousands (2017)')
ax[0][0].set_ylabel('Population in thousands (2017)')
ax[1][1].set_ylabel('Population density (per km2, 2017)')
ax[0][1].set_ylabel('Population density (per km2, 2017)')
ax[1][2].set_ylabel('GDP per capita (current US$)')
ax[0][2].set_ylabel('GDP per capita (current US$)')

# adding plot title for each subplot:
ax[0][0].set_title("BRICS members' total population (2017)")
ax[0][1].set_title("BRICS members' population density (2017)")
ax[0][2].set_title("BRICS members' GDP per capita (2017)")

ax[1][0].set_title("Region-wise total population (2017)")
ax[1][1].set_title("Region-wise population density (2017)")
ax[1][2].set_title("Region-wise GDP per capita (2017)")

# adjusting the spacing between subplots to minimize the overlaps:
plt.tight_layout()
# displaying the final plot:
plt.show()


## Chart 2 - Scatter plot showing the Relation between GDP per capita, Total Population (2017) & Popualtion Density levels among BRICS members:
# creating a Figure and Axes objects and defining the figure size of the chart:
fig, ax = plt.subplots(figsize=(8, 6))

# plotting the Scatter plot showing the Total population on x-axis and GDP per capita on the y-axis:
# also showing the size of each point on the plot equal to 2-times the Population density (per km2) for that country:
ax.scatter(x = BRICS_country_stats2['Population in thousands (2017)'], y = BRICS_country_stats2['GDP per capita (current US$)'],
           s = BRICS_country_stats2['Population density (per km2, 2017)']*2)

# labeling the axes for the scatter plot:
plt.xlabel("Total Population (in thousands 2017)",fontsize=13)
plt.ylabel("GDP per capita (current US$)",fontsize=13)
# adding the plot title:
plt.title('GDP per capita vs. Total Population among BRICS Members (2017)',fontsize=14)

# configuring the grid lines on the face of plot (using the Matplotlib's grid function) as color grey:
plt.grid(b=True,color='grey',linestyle='--')

# labelling each data point on the scatter plot to show the names of BRICS countries:
label = ['Brazil','China','India','Russian Federation','South Africa']
for i, txt in enumerate(label):
    ax.annotate(txt, (BRICS_country_stats2['Population in thousands (2017)'][i], BRICS_country_stats2['GDP per capita (current US$)'][i]),
                size=12)
# displaying the plot finally:
plt.show()


## Chart 3 - Visualizing the contribution of different Sectors to each BRICS member economy (2017) as a Stacked Bar Chart -
## Step 1 - Slicing, Cleaning & Formatting the data before plotting the chart -
# slicing the csv file for relevant columns and importing them into Pandas DataFrame:
BRICS_economy = BRICS_country_stats[['country','Region','Economy: Agriculture (% of GVA)','Economy: Industry (% of GVA)','Economy: Services and other activity (% of GVA)']]
# setting country column as the index for the sliced DataFrame:
BRICS_economy.set_index('country',inplace = True)
# checking the data type for one value of Economy: Agriculture (% of GVA) column:
BRICS_economy['Economy: Agriculture (% of GVA)']['South Africa']
# printing each column data type as it is:
print(BRICS_economy.dtypes)
# changing the data type of the 3 Indicator columns to float:
BRICS_economy['Economy: Agriculture (% of GVA)'] = BRICS_economy['Economy: Agriculture (% of GVA)'].astype(float)
# printing the data types for columns after the change:
print(BRICS_economy.dtypes)
# printing the sliced DataFrame:
print(BRICS_economy)

## Step 2 - Plotting the Percent Stacked Bar Chart -
# creating a Figure and Axes objects and defining the figure size of the chart:
fig,ax = plt.subplots(figsize=(10,6))
# plotting the bar chart for all the Economy indicators stacked on top of each other for every BRICS member country:
ax.bar(BRICS_economy.index,BRICS_economy['Economy: Agriculture (% of GVA)'],label = 'Agriculture')
ax.bar(BRICS_economy.index,BRICS_economy['Economy: Industry (% of GVA)'],bottom= BRICS_economy['Economy: Agriculture (% of GVA)'], label = 'Industry')
ax.bar(BRICS_economy.index,BRICS_economy['Economy: Services and other activity (% of GVA)'], bottom=(BRICS_economy['Economy: Agriculture (% of GVA)'] + BRICS_economy['Economy: Industry (% of GVA)']),label = 'Services (& Other)')

# adding the tick labels for x-axis as country names and rotating them:
ax.set_xticklabels(BRICS_economy.index,rotation=15)
# adding the y-axis labels:
ax.set_ylabel('Share as % of GVA',fontsize=13)
# adding a title to the stacked plot by splitting the title in 2 different lines:
ax.set_title('BRICS Members: Sectoral Contributions to Economy in 2017 \n (as % of Gross Value Added (GVA))',fontsize=14)
# adding the legend, defining its location & giving the legend a title:
ax.legend(title="Sectors of Economy:",loc='upper right')

# looping to add the text for each value on each bar chart part:
# Step 1 - creating a list of each indicator value to be displayed as text:
list_values = (BRICS_economy['Economy: Agriculture (% of GVA)'].tolist()
                + BRICS_economy['Economy: Industry (% of GVA)'].tolist()
                + BRICS_economy['Economy: Services and other activity (% of GVA)'].tolist())
# Step 2 - getting the height & width of each bar rectangle and add the text by aligning it to the center of the rectangle
# both horizontally & vertically:
for rect, value in zip(ax.patches, list_values):
    h = rect.get_height() /2.
    w = rect.get_width() /2.
    x, y = rect.get_xy()
    ax.text(x+w, y+h,value,horizontalalignment='center',verticalalignment='center')

# finally displaying the plot:
plt.show()




## Chart 4 - Visualizing International Trade Indicators for BRICS members (2017) as a Horizontal Bar Plot-
## Step 1 - Slicing, Cleaning & Formatting the data before plotting the chart -
# slicing the csv file for relevant columns and importing them into Pandas DataFrame:
BRICS_int_trade = BRICS_country_stats[['country','Region','International trade: Exports (million US$)','International trade: Imports (million US$)','International trade: Balance (million US$)','Balance of payments, current account (million US$)']]
# setting country column as the index for sliced DataFrame:
BRICS_int_trade.set_index('country',inplace = True)
# renaming the column names to shorten them:
BRICS_int_trade.rename(columns={'International trade: Exports (million US$)':'Exports (million US$)','International trade: Imports (million US$)': 'Imports (million US$)',
                        'International trade: Balance (million US$)':'Bal of Trade (million US$)',
                                'Balance of payments, current account (million US$)': 'Bal of Paymnt (Current) (million US$)'},inplace=True)

# checking the data type for each column of the sliced DataFrame:
print(BRICS_int_trade.dtypes)
# changing the data type for the 4 Indicator columns to float:
BRICS_int_trade['Exports (million US$)'] = BRICS_int_trade['Exports (million US$)'].astype(float)
BRICS_int_trade['Imports (million US$)'] = BRICS_int_trade['Imports (million US$)'].astype(float)
BRICS_int_trade['Bal of Trade (million US$)'] = BRICS_int_trade['Bal of Trade (million US$)'].astype(float)
BRICS_int_trade['Bal of Paymnt (Current) (million US$)'] = BRICS_int_trade['Bal of Paymnt (Current) (million US$)'].astype(float)
# printing the data types for columns after the change:
print(BRICS_int_trade.dtypes)


## Step 2 - Plotting the Horizontal Bar Plot -
# plotting the bar plot for each Trade Indicator:
BRICS_int_trade[['Exports (million US$)','Imports (million US$)','Bal of Trade (million US$)','Bal of Paymnt (Current) (million US$)']].plot(kind='barh',figsize=(12,8))
# adding a title to the chart:
plt.title('Levels of International Trade Components among BRICS Members (2017)',fontsize=14)
# adding x-axis & y-axis labels to the chart:
plt.xlabel('Trade Components (in million US$)',fontsize=13)
plt.ylabel('BRICS member countries',fontsize=13)

# setting x-axis limit ranges:
plt.xlim([-500000,2200000])

# configuring the grid lines on the face of plot (using the Matplotlib's grid function) and
# customizing it with color & linestyle:
plt.grid(b=True,color='grey',linestyle='--')

# adding the legend, defining its location & giving the legend a title:
plt.legend(['Exports','Imports','Balance of Trade','Balance of Payment (Current)'],
           loc='upper right',title='Trade Variables')

# annotating the plot with text "Trade Deficit" for India values:
plt.annotate("Trade Deficit", xy=(-97000,2),horizontalalignment='right', verticalalignment='center',fontsize=12,
xytext=(-2,1),arrowprops={'arrowstyle':"->","color":"black"})

# displaying the plot finally:
plt.show()





### Data Source 2 - Data from Human Development Data Center - Using HDRO Statistical Data API -

# importing the request library to make the HTTP request:
import requests
# packaging the request, sending it & catching the response for HDR webpage url:
response = requests.get('http://hdr.undp.org/en/countries')
# printing the response and the response status code:
print(response)


## Step 1 - Querying the API, Retrieving the Data and Importing it into Pandas DataFrame -
## Data for all 3 Health Indicators:-
# Health Indicator 1 - Life expectancy at birth (years) for 2019 (indicator_id : 69206)
# importing the Pandas library (alias pd):
import pandas as pd
# packaging the request, sending it to retrieve BRICS members' data:
response = requests.get('http://ec2-54-174-131-205.compute-1.amazonaws.com/API/HDRO_API.php/country_code=BRA,RUS,IND,CHN,ZAF/indicator_id=69206/year=2019/structure=ciy')
# getting the API response in JSON format and assigning it to a variable:
Life_exp_birth = response.json()
# importing the JSON response into Pandas DataFrame, iterating over its content and assigning it to another variable:
my_dict3 = pd.DataFrame(Life_exp_birth['indicator_value'].items())
# renaming the columns extracted from the DataFrame:
my_dict3.rename(columns={0:'Country_Code',1:'Health_Indicator'}, inplace=True)

# Health Indicator 2 - Life expectancy at birth, female (years) for 2019 (indicator_id : 120606)
# packaging the request, sending it to retrieve BRICS members' data:
response = requests.get('http://ec2-54-174-131-205.compute-1.amazonaws.com/API/HDRO_API.php/country_code=BRA,RUS,IND,CHN,ZAF/indicator_id=120606/year=2019')
# getting the API response in JSON format and assigning it to a variable:
Life_exp_birth_female = response.json()
# importing the JSON response into Pandas DataFrame, iterating over its content and assigning it to another variable:
my_dict4 = pd.DataFrame(Life_exp_birth_female['indicator_value'].items())
# renaming the columns extracted from the DataFrame:
my_dict4.rename(columns={0:'Country_Code',1:'Health_Indicator'}, inplace=True)

# Health Indicator 3 - Life expectancy at birth, male (years) for 2019 (indicator_id : 121106)
# packaging the request, sending it to retrieve BRICS members' data:
response = requests.get('http://ec2-54-174-131-205.compute-1.amazonaws.com/API/HDRO_API.php/country_code=BRA,RUS,IND,CHN,ZAF/indicator_id=121106/year=2019/structure=ciy')
# getting the API response in JSON format and assigning it to a variable:
Life_exp_birth_male = response.json()
# importing the JSON response into Pandas DataFrame, iterating over its content and assigning it to another variable:
my_dict5 = pd.DataFrame(Life_exp_birth_male['indicator_value'].items())
# renaming the columns extracted from the DataFrame:
my_dict5.rename(columns={0:'Country_Code',1:'Health_Indicator'}, inplace=True)

## Step 2 - Combining all the Indicator DataFrames into one -
# merging all 3 Health indicator DataFrames and assigning it to a variable:
Health_dict = my_dict3.merge(my_dict4,on='Country_Code',how='outer',suffixes=('_lifexp','_lifexp_fem'))
Health_dict2 = Health_dict.merge(my_dict5,on='Country_Code',how='left')
# renaming the columns of merged Health DataFrame -
Health_dict2.rename(columns={'Health_Indicator_lifexp':'Life expectancy at birth','Health_Indicator_lifexp_fem':'Life expectancy at birth, Female',
                            'Health_Indicator': 'Life expectancy at birth, Male'}, inplace=True)
# printing the Health DataFrame:
print(Health_dict2)

## Step 3 - Creating a DataFrame for indicator values data -
values = []
for i in range(0,5):
    for j in range(1,4):
        data = Health_dict2.loc[i][j].values()
        for item in data:
            ind_val = item.values()
            for s in ind_val:
                values.append(s)
                df = pd.DataFrame(values)
# printing the values DataFrame:
print(df)

## Step 4 - Creating a DataFrame for year values data -
year = []
for i in range(0,5):
    for j in range(1,4):
        data = Health_dict2.loc[i][j].values()
        for item in data:
            ind_val2 = item.keys()
            for s in ind_val2:
                year.append(s)
                df2 = pd.DataFrame(year)
# printing the year DataFrame:
print(df2)

## Step 5 - Data Cleaning for each column of Combined Health DataFrame:
# cleaning the Life expectancy at birth column:
for i in range(0,5):
    if i == 0:
        Health_dict2.iloc[i][1]=df.loc[0]
    else:
        for k in range(i*3):
            Health_dict2.iloc[i][1]=df.loc[k+1]
# cleaning the Life expectancy at birth, Female column:
for i in range(0,5):
    if i == 0:
        Health_dict2.iloc[i][2]=df.loc[1]
    else:
        for k in range(i*3+1):
            Health_dict2.iloc[i][2]=df.loc[k+1]
# cleaning the Life expectancy at birth, Male column:
for i in range(0,5):
    if i == 0:
        Health_dict2.iloc[i][3]=df.loc[2]
    else:
        for k in range(i*3+2):
            Health_dict2.iloc[i][3]=df.loc[k+1]
# adding the year data to cleaned DataFrame by slicing df2:
year = df2.loc[0:4]
# renaming the year data column as 'Year':
Health_dict2['Year'] = year
# printing the cleaned Health DataFrame:
print(Health_dict2)

## Step 6 - Data Formatting for each column of Combined Health DataFrame:
# changing the Datatype for 3 Health indicator values columns to float:
Health_dict2["Life expectancy at birth"] = Health_dict2['Life expectancy at birth'].astype('float')
Health_dict2["Life expectancy at birth, Female"] = Health_dict2['Life expectancy at birth, Female'].astype('float')
Health_dict2["Life expectancy at birth, Male"] = Health_dict2['Life expectancy at birth, Male'].astype('float')
# rechecking the data type post change:
print(Health_dict2.dtypes)

## Step 7 - Printing the cleaned & formatted Health DataFrame:
print(Health_dict2)


## Step 1 - Querying the API, Retrieving the Data and Importing it into Pandas DataFrame -
## Data for all 2 Education Indicators:-
# Education Indicator 1 - Expected years of schooling (years) for 2019 (indicator_id : 69706)
# packaging the request, sending it to retrieve BRICS members' data:
response = requests.get('http://ec2-54-174-131-205.compute-1.amazonaws.com/API/HDRO_API.php/country_code=BRA,RUS,IND,CHN,ZAF/indicator_id=69706/year=2019/structure=ciy')
# getting the API response in JSON format and assigning it to a variable:
Exp_yrs_schooling = response.json()
# importing the JSON response into Pandas DataFrame, iterating over its content and assigning it to another variable:
my_dict12 = pd.DataFrame(Exp_yrs_schooling['indicator_value'].items())
# renaming the columns extracted from the DataFrame:
my_dict12.rename(columns={0:'Country_Code',1:'Education_Indicator'}, inplace=True)

# Education Indicator 2 - Mean years of schooling (years) for 2019 (indicator_id : 103006)
# packaging the request, sending it to retrieve BRICS members' data:
response = requests.get('http://ec2-54-174-131-205.compute-1.amazonaws.com/API/HDRO_API.php/country_code=BRA,RUS,IND,CHN,ZAF/indicator_id=103006/year=2019/structure=ciy')
# getting the API response in JSON format and assigning it to a variable:
Mean_yrs_schooling = response.json()
# importing the JSON response into Pandas DataFrame, iterating over its content and assigning it to another variable:
my_dict13 = pd.DataFrame(Mean_yrs_schooling['indicator_value'].items())
# renaming the columns extracted from the DataFrame:
my_dict13.rename(columns={0:'Country_Code',1:'Education_Indicator'}, inplace=True)

## Step 2 - Combining all the Indicator DataFrames into one -
# merging all 2 Education indicator Dataframes and assigning it to a variable:
Educational_dict = my_dict12.merge(my_dict13,on='Country_Code',how='outer',suffixes=('_Exp_yrs_schooling','_Mean_yrs_schooling'))
# renaming the columns of merged Education DataFrame:
Educational_dict.rename(columns={'Education_Indicator_Exp_yrs_schooling':'Expected years of schooling (years)','Education_Indicator_Mean_yrs_schooling':'Mean years of schooling (years)'}, inplace=True)
# printing the Education DataFrame:
print(Educational_dict)

## Step 3 - Creating a DataFrame for indicator values data -
values = []
for i in range(0,5):
    for j in range(1,3):
        data = Educational_dict.loc[i][j].values()
        for item in data:
            ind_val = item.values()
            for s in ind_val:
                values.append(s)
                df = pd.DataFrame(values)
# printing the values DataFrame:
print(df)

## Step 4 - Creating a DataFrame for year values data -
year = []
for i in range(0,5):
    for j in range(1,3):
        data = Educational_dict.loc[i][j].values()
        for item in data:
            ind_val2 = item.keys()
            for s in ind_val2:
                year.append(s)
                df2 = pd.DataFrame(year)
# printing the year DataFrame:
print(df2)

## Step 5 - Data Cleaning for each column of Combined Education DataFrame:
# cleaning the Expected years of schooling (years) column:
for i in range(0,5):
    if i == 0:
        Educational_dict.iloc[i][1]=df.loc[0]
    else:
        for k in range(i*2):
            Educational_dict.iloc[i][1]=df.loc[k+1]
# cleaning the Mean years of schooling (years) column:
for i in range(0,5):
    if i == 0:
        Educational_dict.iloc[i][2]=df.loc[1]
    else:
        for k in range(i*2+1):
            Educational_dict.iloc[i][2]=df.loc[k+1]
# adding the year data to cleaned DataFrame by slicing df2:
year = df2.loc[0:4]
# renaming the year data column as 'Year':
Educational_dict['Year'] = year
# printing the cleaned Education DataFrame:
print(Educational_dict)

## Step 6 - Data Formatting for each column of Combined Education DataFrame:
# changing the Datatype for 2 Education indicator values columns to float:
Educational_dict["Expected years of schooling (years)"] = Educational_dict['Expected years of schooling (years)'].astype('float')
Educational_dict["Mean years of schooling (years)"] = Educational_dict['Mean years of schooling (years)'].astype('float')
# rechecking the data type post change:
print(Educational_dict.dtypes)

## Step 7 - Printing the cleaned & formatted Education DataFrame:
print(Educational_dict)




## Chart 1 - Visualizing Life Expectancy and Educational Schooling for BRICS Members in 2019 -
# loading Matplotlib library (alias as plt):
import matplotlib.pyplot as plt
# slicing the Health & Education DataFrames for specific columns:
df12 = Health_dict2[['Country_Code','Life expectancy at birth, Male','Life expectancy at birth, Female']]
df13 = Educational_dict[['Country_Code','Expected years of schooling (years)','Mean years of schooling (years)']]
# combining sliced Health & Education DataFrames (using concat function) and assigning it to variable df:
df = pd.concat([df12.set_index('Country_Code'), df13.set_index('Country_Code')], axis=1)
# plotting the combined df DataFrame as a bar chart:
df.plot.bar(figsize=(10, 8))
# adding a title to the plot:
plt.title('Years of Life Expectancy and Schooling among BRICS Members (2019)',fontsize=14)
# labeling the x-axes ticks on the plot:
plt.xticks(rotation=0)
# labeling the axes of the plot:
plt.ylabel('Years',fontsize=13)
plt.xlabel('BRICS member countries',fontsize=13)
# adding a legend to the plot and formatting itd location, fontsize & transparency level:
plt.legend(loc='upper right',fontsize=12,framealpha=0.5)
# showing the grid lines on the plot & formatting them:
plt.grid(b=True,color='grey',linestyle='--')
# displaying the final plot:
plt.show()


## Step 1 - Querying the API, Retrieving the Data and Importing it into Pandas DataFrame -
## Data for all 3 Demography Indicators:-
# Demography Indicator 1 - Total population (millions) for 2019 (indicator_id : 44206)
# packaging the request, sending it to retrieve BRICS members' data:
response = requests.get('http://ec2-54-174-131-205.compute-1.amazonaws.com/API/HDRO_API.php/country_code=BRA,RUS,IND,CHN,ZAF/indicator_id=44206/year=2019/structure=ciy')
# getting the API response in JSON format and assigning it to a variable:
Total_pop = response.json()
# importing the JSON response into Pandas DataFrame, iterating over its content and assigning it to another variable:
my_dict6 = pd.DataFrame(Total_pop['indicator_value'].items())
# renaming the columns extracted from the DataFrame:
my_dict6.rename(columns={0:'Country_Code',1:'Demographic_Indicator'}, inplace=True)

# Demography Indicator 2 - Sex ratio at birth (male to female births) for 2019 (indicator_id : 49006)
# packaging the request, sending it to retrieve BRICS members' data:
response = requests.get('http://ec2-54-174-131-205.compute-1.amazonaws.com/API/HDRO_API.php/country_code=BRA,RUS,IND,CHN,ZAF/indicator_id=49006/year=2019')
# getting the API response in JSON format and assigning it to a variable:
Sex_ratio = response.json()
# importing the JSON response into Pandas DataFrame, iterating over its content and assigning it to another variable:
my_dict7 = pd.DataFrame(Sex_ratio['indicator_value'].items())
# # renaming the columns extracted from the DataFrame:
my_dict7.rename(columns={0:'Country_Code',1:'Demographic_Indicator'}, inplace=True)

# Demography Indicator 3 - Urban population (%) for 2019 (indicator_id : 45106)
# packaging the request, sending it to retrieve BRICS members' data:
response = requests.get('http://ec2-54-174-131-205.compute-1.amazonaws.com/API/HDRO_API.php/country_code=BRA,RUS,IND,CHN,ZAF/indicator_id=45106/year=2019')
# getting the API response in JSON format and assigning it to a variable:
Urban_pop = response.json()
# importing the JSON response into Pandas DataFrame, iterating over its content and assigning it to another variable:
my_dict8 = pd.DataFrame(Urban_pop['indicator_value'].items())
# renaming the columns extracted from the DataFrame:
my_dict8.rename(columns={0:'Country_Code',1:'Demographic_Indicator'}, inplace=True)

## Step 2 - Combining all the Indicator DataFrame into one -
# merging all 3 Demography indicator DataFrames and assigning it to a variable:
Demographic_dict = my_dict6.merge(my_dict7,on='Country_Code',how='outer',suffixes=('_tot_pop','_sex_ratio'))
Demographic_dict2 = Demographic_dict.merge(my_dict8,on='Country_Code',how='left')
# renaming the columns of merged Demography DataFrame:
Demographic_dict2.rename(columns={'Demographic_Indicator_tot_pop':'Total population (millions)','Demographic_Indicator_sex_ratio':'Sex ratio at birth (M to F)',
                            'Demographic_Indicator': 'Urban population (%)'}, inplace=True)
# printing the Health DataFrame:
print(Demographic_dict2)

## Step 3 - Creating a DataFrame for indicator values data -
values = []
for i in range(0,5):
    for j in range(1,4):
        data = Demographic_dict2.loc[i][j].values()
        for item in data:
            ind_val = item.values()
            for s in ind_val:
                values.append(s)
                df = pd.DataFrame(values)
# printing the values DataFrame:
print(df)

## Step 4 - Creating a DataFrame for year values data -
year = []
for i in range(0,5):
    for j in range(1,4):
        data = Demographic_dict2.loc[i][j].values()
        for item in data:
            ind_val2 = item.keys()
            for s in ind_val2:
                year.append(s)
                df2 = pd.DataFrame(year)
# printing the year DataFrame:
print(df2)


## Step 5 - Data Cleaning for each column of Combined Demography DataFrame:
# cleaning the Total population (millions) column:
for i in range(0,5):
    if i == 0:
        Demographic_dict2.iloc[i][1]=df.loc[0]
    else:
        for k in range(i*3):
            Demographic_dict2.iloc[i][1]=df.loc[k+1]
# cleaning the Sex ratio at birth (M to F) column:
for i in range(0,5):
    if i == 0:
        Demographic_dict2.iloc[i][2]=df.loc[1]
    else:
        for k in range(i*3+1):
            Demographic_dict2.iloc[i][2]=df.loc[k+1]
# cleaning the Urban population (%) column:
for i in range(0,5):
    if i == 0:
        Demographic_dict2.iloc[i][3]=df.loc[2]
    else:
        for k in range(i*3+2):
            Demographic_dict2.iloc[i][3]=df.loc[k+1]
# adding the year data to cleaned DataFrame by slicing df2:
year = df2.loc[0:4]
# renaming the year data column as 'Year':
Demographic_dict2['Year'] = year
# printing the cleaned Health DataFrame:
print(Demographic_dict2)

## Step 6 - Data Formatting for each column of Combined Demography DataFrame:
# changing the Datatype for 3 Demography indicator values columns to float:
Demographic_dict2["Total population (millions)"] = Demographic_dict2['Total population (millions)'].astype('float')
Demographic_dict2["Sex ratio at birth (M to F)"] = Demographic_dict2['Sex ratio at birth (M to F)'].astype('float')
Demographic_dict2["Urban population (%)"] = Demographic_dict2['Urban population (%)'].astype('float')
# rechecking the data type post change:
print(Demographic_dict2.dtypes)

## Step 7 - Printing the cleaned & formatted Demography DataFrame:
print(Demographic_dict2)



## Chart 2 - Visualizing Demographic Indicators across BRICS Members in 2019 -
# creating a Figure and Axes objects (2 rows and 1 column) and defining the figure size:
fig,ax = plt.subplots(2,1,figsize=(10,8))
# Making Subplot 1 - plotting the top plot as a mix of Total population & Share of Urban Population in 2019:
# plotting the bar chart for Total population in 2019 on the first y-axis object:
ax[0].bar(Demographic_dict2['Country_Code'], Demographic_dict2['Total population (millions)'], color="red",alpha = 0.5)
# setting the y-axis label and customizing it:
ax[0].set_ylabel("Population (millions)",color="red",fontsize=13)
# creating a twin object for two different y-axis on the first subplot:
ax2 = ax[0].twinx()
# making a line plot of Share of Urban Population with different y-axis using second axis object:
ax2.plot(Demographic_dict2['Country_Code'], Demographic_dict2["Urban population (%)"],color="blue",marker="o")
# setting the  label for twin y-axis and customizing it:
ax2.set_ylabel("Urban population (%)",color="blue",fontsize=13)

# Making Subplot 2 - plotting the bottom plot as a mix of Total population & Sex ratio at birth (M to F) in 2019:
# plotting the bar chart for Total population in 2019 on the first y-axis object:
ax[1].bar(Demographic_dict2['Country_Code'], Demographic_dict2['Total population (millions)'], color="red",alpha = 0.5)
# setting the x-axis label on the bottom plot axis:
ax[1].set_xlabel("BRICS member countries ",fontsize=13)
# setting y-axis label and customizing it:
ax[1].set_ylabel("Population (millions)",color="red",fontsize=13)
# creating a twin object for two different y-axis on the second subplot:
ax2=ax[1].twinx()
# making a line plot of Sex ratio at birth with different y-axis using second axis object:
ax2.plot(Demographic_dict2['Country_Code'], Demographic_dict2["Sex ratio at birth (M to F)"],color="blue",marker="o")
# setting the  label for twin y-axis and customizing it:
ax2.set_ylabel("Sex ratio at birth (M to F)",color="blue",fontsize=13)

# setting the chart title for both the subplots:
ax[0].set_title('Total Population vs. Share of Urban Population across BRICS Members (2019)',fontsize=14)
ax[1].set_title('Total Population vs. Male to Female Sex Ratio across BRICS Members (2019)',fontsize=14)
# displaying the final plot:
plt.show()


## Step 1 - Querying the API, Retrieving the Data and Importing it into Pandas DataFrame -
## Data for all 2 Trade & Finance Indicators:-
# Trade Finance Indicator 1 - Exports and imports (% of GDP) for 2019 (indicator_id : 133206)
# packaging the request, sending it to retrieve BRICS members' data:
response = requests.get('http://ec2-54-174-131-205.compute-1.amazonaws.com/API/HDRO_API.php/country_code=BRA,RUS,IND,CHN,ZAF/indicator_id=133206/year=2019/structure=ciy')
# getting the API response in JSON format and assigning it to a variable:
Exports_and_Imports = response.json()
# importing the JSON response into Pandas DataFrame, iterating over its content and assigning it to another variable:
my_dict14 = pd.DataFrame(Exports_and_Imports['indicator_value'].items())
# renaming the columns extracted from the DataFrame:
my_dict14.rename(columns={0:'Country_Code',1:'Trade_Indicator'}, inplace=True)

# Trade Finance Indicator 2 - Foreign direct investment, net inflows (% of GDP) for 2019 (indicator_id : 53506)
# packaging the request, sending it to retrieve BRICS members' data:
response = requests.get('http://ec2-54-174-131-205.compute-1.amazonaws.com/API/HDRO_API.php/country_code=BRA,RUS,IND,CHN,ZAF/indicator_id=53506/year=2019/structure=ciy')
# getting the API response in JSON format and assigning it to a variable:
FDI_Net_Inflows = response.json()
# importing the JSON response into Pandas DataFrame, iterating over its content and assigning it to another variable:
my_dict15 = pd.DataFrame(FDI_Net_Inflows['indicator_value'].items())
# renaming the columns extracted from the DataFrame:
my_dict15.rename(columns={0:'Country_Code',1:'Trade_Indicator'}, inplace=True)


## Step 2 - Combining all the Indicator DataFrame into one -
# merging all 3 Trade Finance indicator DataFrames and assigning it to a variable:
Trade_Fin_dict = my_dict14.merge(my_dict15,on='Country_Code',how='outer',suffixes=('_Exports_and_Imports','_FDI_Net_Inflows'))
# renaming the columns of merged Trade Finance DataFrame -
Trade_Fin_dict.rename(columns={'Trade_Indicator_Exports_and_Imports':'Exports and imports (% of GDP)','Trade_Indicator_FDI_Net_Inflows':'FDI, net inflows (% of GDP)'}, inplace=True)
# printing the Trade Finance DataFrame:
print(Trade_Fin_dict)


## Step 3 - Creating a DataFrame for indicator values data -
values = []
for i in range(0,5):
    for j in range(1,3):
        data = Trade_Fin_dict.loc[i][j].values()
        for item in data:
            ind_val = item.values()
            for s in ind_val:
                values.append(s)
                df = pd.DataFrame(values)
# printing the values DataFrame:
print(df)


## Step 4 - Creating a DataFrame for year values data -
year = []
for i in range(0,5):
    for j in range(1,3):
        data = Trade_Fin_dict.loc[i][j].values()
        for item in data:
            ind_val2 = item.keys()
            for s in ind_val2:
                year.append(s)
                df2 = pd.DataFrame(year)
# printing the year DataFrame:
print(df2)


## Step 5 - Data Cleaning for each column of Combined Trade Finance DataFrame:
# cleaning the Exports and imports (% of GDP) column:
for i in range(0,5):
    if i == 0:
        Trade_Fin_dict.iloc[i][1]=df.loc[0]
    else:
        for k in range(i*2):
            Trade_Fin_dict.iloc[i][1]=df.loc[k+1]
# cleaning the Foreign direct investment, net inflows (% of GDP) column:
for i in range(0,5):
    if i == 0:
        Trade_Fin_dict.iloc[i][2]=df.loc[1]
    else:
        for k in range(i*2+1):
            Trade_Fin_dict.iloc[i][2]=df.loc[k+1]
# adding the year data to cleaned DataFrame by slicing df2:
year = df2.loc[0:4]
# renaming the year data column as 'Year':
Trade_Fin_dict['Year'] = year
# printing the cleaned Health DataFrame:
print(Trade_Fin_dict)

## Step 6 - Data Formatting for each column of Combined Trade Finance DataFrame:
# changing the Datatype for 2 Trade Finance indicator values columns to float:
Trade_Fin_dict["Exports and imports (% of GDP)"] = Trade_Fin_dict['Exports and imports (% of GDP)'].astype('float')
Trade_Fin_dict["FDI, net inflows (% of GDP)"] = Trade_Fin_dict['FDI, net inflows (% of GDP)'].astype('float')
# rechecking the data type post change:
print(Trade_Fin_dict.dtypes)

## Step 7 - Printing the cleaned & formatted Trade Finance DataFrame:
print(Trade_Fin_dict)


## Step 1 - Querying the API, Retrieving the Data and Importing it into Pandas DataFrame -
## Data for all 3 Income Indicators:-
# Income Indicator 1 - Gross national income (GNI) per capita for 2019 (indicator_id : 195706)
# packaging the request, sending it to retrieve BRICS members' data:
response = requests.get('http://ec2-54-174-131-205.compute-1.amazonaws.com/API/HDRO_API.php/country_code=BRA,RUS,IND,CHN,ZAF/indicator_id=195706/year=2019/structure=ciy')
# getting the API response in JSON format and assigning it to a variable:
per_cap_GNI = response.json()
# importing the JSON response into Pandas DataFrame, iterating over its content and assigning it to another variable:
my_dict9 = pd.DataFrame(per_cap_GNI['indicator_value'].items())
# renaming the columns extracted from the DataFrame:
my_dict9.rename(columns={0:'Country_Code',1:'Income_Indicator'}, inplace=True)

# Income Indicator 2 - Gross Domestic Product (GDP) per capita for 2019 (indicator_id : 194906)
# packaging the request, sending it to retrieve BRICS members' data:
response = requests.get('http://ec2-54-174-131-205.compute-1.amazonaws.com/API/HDRO_API.php/country_code=BRA,RUS,IND,CHN,ZAF/indicator_id=194906/year=2019/structure=ciy')
# getting the API response in JSON format and assigning it to a variable:
per_cap_GDP = response.json()
# importing the JSON response into Pandas DataFrame, iterating over its content and assigning it to another variable:
my_dict10 = pd.DataFrame(per_cap_GDP['indicator_value'].items())
# renaming the columns extracted from the DataFrame:
my_dict10.rename(columns={0:'Country_Code',1:'Income_Indicator'}, inplace=True)

# Income Indicator 3 - Total Unemployment (% of labour force) for 2019 (indicator_id : 140606)
# packaging the request, sending it to retrieve BRICS members' data:
response = requests.get('http://ec2-54-174-131-205.compute-1.amazonaws.com/API/HDRO_API.php/country_code=BRA,RUS,IND,CHN,ZAF/indicator_id=140606/year=2019/structure=ciy')
# getting the API response in JSON format and assigning it to a variable:
Tot_Unemp = response.json()
# importing the JSON response into Pandas DataFrame, iterating over its content and assigning it to another variable:
my_dict11 = pd.DataFrame(Tot_Unemp['indicator_value'].items())
# renaming the columns extracted from the DataFrame:
my_dict11.rename(columns={0:'Country_Code',1:'Income_Indicator'}, inplace=True)


## Step 2 - Combining all the Indicator DataFrame into one -
# merging all 3 Income indicator DataFrames and assigning it to a variable:
Income_dict = my_dict9.merge(my_dict10,on='Country_Code',how='outer',suffixes=('_per_cap_GNI','_per_cap_GDP'))
Income_dict2 = Income_dict.merge(my_dict11,on='Country_Code',how='left')

# renaming the columns of merged Income DataFrame -
Income_dict2.rename(columns={'Income_Indicator_per_cap_GNI':'Gross national income (GNI) per capita','Income_Indicator_per_cap_GDP':'Gross Domestic Product (GDP) per capita',
                            'Income_Indicator': 'Total Unemployment (% of labour force)'}, inplace=True)
# printing the Income DataFrame:
print(Income_dict2)

## Step 3 - Creating a DataFrame for indicator values data -
values = []
for i in range(0,5):
    for j in range(1,4):
        data = Income_dict2.loc[i][j].values()
        for item in data:
            ind_val = item.values()
            for s in ind_val:
                values.append(s)
                df = pd.DataFrame(values)
# printing the values DataFrame:
print(df)

## Step 4 - Creating a DataFrame for year values data -
year = []
for i in range(0,5):
    for j in range(1,4):
        data = Income_dict2.loc[i][j].values()
        for item in data:
            ind_val2 = item.keys()
            for s in ind_val2:
                year.append(s)
                df2 = pd.DataFrame(year)
# printing the year DataFrame:
print(df2)


## Step 5 - Data Cleaning for each column of Combined Income DataFrame:
# cleaning the Gross national income (GNI) per capita column:
for i in range(0,5):
    if i == 0:
        Income_dict2.iloc[i][1]=df.loc[0]
    else:
        for k in range(i*3):
            Income_dict2.iloc[i][1]=df.loc[k+1]
# cleaning the GDP per capita column:
for i in range(0,5):
    if i == 0:
        Income_dict2.iloc[i][2]=df.loc[1]
    else:
        for k in range(i*3+1):
            Income_dict2.iloc[i][2]=df.loc[k+1]
# cleaning the Total Unemployment (% of labour force) column:
for i in range(0,5):
    if i == 0:
        Income_dict2.iloc[i][3]=df.loc[2]
    else:
        for k in range(i*3+2):
            Income_dict2.iloc[i][3]=df.loc[k+1]
# adding the year data to cleaned DataFrame by slicing df2:
year = df2.loc[0:4]
# renaming the year data column as 'Year':
Income_dict2['Year'] = year
# printing the cleaned Health DataFrame:
print(Income_dict2)


## Step 6 - Data Formatting for each column of Combined Income DataFrame:
# changing the Datatype for 3 Income indicator values columns to float:
Income_dict2["Gross national income (GNI) per capita"] = Income_dict2['Gross national income (GNI) per capita'].astype('float')
Income_dict2["Gross Domestic Product (GDP) per capita"] = Income_dict2['Gross Domestic Product (GDP) per capita'].astype('float')
Income_dict2["Total Unemployment (% of labour force) "] = Income_dict2['Total Unemployment (% of labour force)'].astype('float')
# rechecking the data type post change:
print(Income_dict2.dtypes)

## Step 7 - Printing the cleaned & formatted Income DataFrame:
print(Income_dict2)


## Step 1 - Querying the API, Retrieving the Data and Importing it into Pandas DataFrame -
## Data for all 3 Trade & Financial Flows Indicators:-
# Trade & Financial Flow Indicator 1 - Gross fixed capital formation (% of GDP) for 2019 (indicator_id : 65606):
# packaging the request, sending it to retrieve BRICS members' data:
response = requests.get('http://ec2-54-174-131-205.compute-1.amazonaws.com/API/HDRO_API.php/country_code=BRA,RUS,IND,CHN,ZAF/indicator_id=65606/year=2019/structure=ciy')
# getting the API response in JSON format and assigning it to a variable:
Gross_fix_cap = response.json()
# importing the JSON response into Pandas DataFrame, iterating over its content and assigning it to another variable:
my_dict20 = pd.DataFrame(Gross_fix_cap['indicator_value'].items())
# renaming the columns extracted from the DataFrame:
my_dict20.rename(columns={0:'Country_Code',1:'TradeF_Indicator'}, inplace=True)

# Trade & Financial Flow Indicator 2 - Private capital flows (% of GDP) for 2019 (indicator_id : 11306):
# packaging the request, sending it to retrieve BRICS members' data:
response = requests.get('http://ec2-54-174-131-205.compute-1.amazonaws.com/API/HDRO_API.php/country_code=BRA,RUS,IND,CHN,ZAF/indicator_id=111306/year=2019/structure=ciy')
# getting the API response in JSON format and assigning it to a variable:
Pvt_cap_flow = response.json()
# importing the JSON response into Pandas DataFrame, iterating over its content and assigning it to another variable:
my_dict21 = pd.DataFrame(Pvt_cap_flow['indicator_value'].items())
# renaming the columns extracted from the DataFrame:
my_dict21.rename(columns={0:'Country_Code',1:'TradeF_Indicator'}, inplace=True)

# Trade & Financial Flow Indicator 3 - Remittances, inflows (% of GDP) for 2019 (indicator_id : 52606):
# packaging the request, sending it to retrieve BRICS members' data:
response = requests.get('http://ec2-54-174-131-205.compute-1.amazonaws.com/API/HDRO_API.php/country_code=BRA,RUS,IND,CHN,ZAF/indicator_id=52606/year=2019/structure=ciy')
# getting the API response in JSON format and assigning it to a variable:
Remittance_inf = response.json()
# importing the JSON response into Pandas DataFrame, iterating over its content and assigning it to another variable:
my_dict22 = pd.DataFrame(Remittance_inf['indicator_value'].items())
# renaming the columns extracted from the DataFrame:
my_dict22.rename(columns={0:'Country_Code',1:'TradeF_Indicator'}, inplace=True)


## Step 2 - Combining all the Indicator DataFrame into one -
# merging all 3 Trade & Financial Flow indicator DataFrames and assigning it to a variable:
Trade_Flows_dict = my_dict20.merge(my_dict21,on='Country_Code',how='outer',suffixes=('_Gross_fix_cap','_Pvt_cap_flow'))
Trade_Flows_dict2 = Trade_Flows_dict.merge(my_dict22,on='Country_Code',how='left')

# renaming the columns of merged Trade & Financial Flow DataFrame -
Trade_Flows_dict2.rename(columns={'TradeF_Indicator_Gross_fix_cap':'Gross fixed capital formation (% of GDP)','TradeF_Indicator_Pvt_cap_flow':'Private capital flows (% of GDP)',
                            'TradeF_Indicator': 'Remittances, inflows (% of GDP)'}, inplace=True)
# printing the Health DataFrame:
print(Trade_Flows_dict2)


# Missing data from API Query - Replacing Missing Value of China Gross fixed capital formation (% of GDP) data from NaN to actual value 42.8
# Step 1 - Fill NaN with a string of nested dictionary value for China data i.e.{'65606': {'2019': 42.80}}
Trade_Flows_dict2["Gross fixed capital formation (% of GDP)"].fillna("{'65606': {'2019': 42.80}}", inplace = True)
# printing the DataFrame after filling NaN:
print(Trade_Flows_dict2)

# Step 2 - Used ast library to Convert a String representation of a Dictionary to a dictionary for China data:
# using ast.literal_eval function:
Trade_Flows_dict2['Gross fixed capital formation (% of GDP)'].fillna("{'65606': {'2019': 42.80}}", inplace = True)
print(type(Trade_Flows_dict2['Gross fixed capital formation (% of GDP)'].loc[4][1]))
# using ast.literal_eval()
import ast
# initializing the string
Trade_Flows_dict2.loc[4][1] = ast.literal_eval(Trade_Flows_dict2.loc[4][1])
# printing the type of filled in data by slicing it from DataFrame:
print(type(Trade_Flows_dict2.loc[4][1]))


## Step 3 - Creating a DataFrame for indicator values data -
values = []
for i in range(0,5):
    for j in range(1,4):
        data = Trade_Flows_dict2.loc[i][j].values()
        for item in data:
            ind_val = item.values()
            for s in ind_val:
                values.append(s)
                df = pd.DataFrame(values)
# printing the values DataFrame:
print(df)


## Step 4 - Creating a DataFrame for year values data -
year = []
for i in range(0,5):
    for j in range(1,4):
        data = Trade_Flows_dict2.loc[i][j].values()
        for item in data:
            ind_val2 = item.keys()
            for s in ind_val2:
                year.append(s)
                df2 = pd.DataFrame(year)
# printing the year DataFrame:
print(df2)


## Step 5 - Data Cleaning for each column of Combined Trade & Financial Flow DataFrame:
# cleaning the Gross fixed capital formation (% of GDP) column:
for i in range(0,5):
    if i == 0:
        Trade_Flows_dict2.iloc[i][1]=df.loc[0]
    else:
        for k in range(i*3):
            Trade_Flows_dict2.iloc[i][1]=df.loc[k+1]
# cleaning the Private capital flows (% of GDP) column:
for i in range(0,5):
    if i == 0:
        Trade_Flows_dict2.iloc[i][2]=df.loc[1]
    else:
        for k in range(i*3+1):
            Trade_Flows_dict2.iloc[i][2]=df.loc[k+1]
# cleaning the Remittances, inflows (% of GDP) column:
for i in range(0,5):
    if i == 0:
        Trade_Flows_dict2.iloc[i][3]=df.loc[2]
    else:
        for k in range(i*3+2):
            Trade_Flows_dict2.iloc[i][3]=df.loc[k+1]
# adding the year data to cleaned DataFrame by slicing df2:
year = df2.loc[0:4]
# renaming the year data column as 'Year':
Trade_Flows_dict2['Year'] = year
# printing the cleaned Health DataFrame:
print(Trade_Flows_dict2)


## Step 6 - Data Formatting for each column of Combined Trade & Financial Flow DataFrame:
# changing the Datatype for 3 Trade & Financial Flow indicator values columns to float:
Trade_Flows_dict2["Gross fixed capital formation (% of GDP)"] = Trade_Flows_dict2['Gross fixed capital formation (% of GDP)'].astype('float')
Trade_Flows_dict2["Private capital flows (% of GDP)"] = Trade_Flows_dict2['Private capital flows (% of GDP)'].astype('float')
Trade_Flows_dict2["Remittances, inflows (% of GDP)"] = Trade_Flows_dict2['Remittances, inflows (% of GDP)'].astype('float')
# rechecking the data type post change:
print(Trade_Flows_dict2.dtypes)

## Step 7 - Printing the cleaned & formatted Trade & Financial Flow DataFrame:
print(Trade_Flows_dict2)


## Step 8 - Merging the Trade Finance & Trade & Financial Flows DataFrames into one:
Trade_Finance_merged = Trade_Fin_dict.merge(Trade_Flows_dict2, on=['Country_Code'],how='left')
# renaming the columns of the merged Trade Finance DataFrame:
Trade_Finance_merged.rename(columns={'TradeF_Indicator_Gross_fix_cap':'Exports and imports (% of GDP)','TradeF_Indicator_Pvt_cap_flow':'FDI, net inflows (% of GDP)'}, inplace=True)
# checking the data type for merged DataFrame columns:
print(Trade_Finance_merged.dtypes)


## Chart 3 - Visualizing the Level of Trade & Financial Flows among BRICS Members in 2019 -
# loading Matplotlib library (alias as plt):
import matplotlib.pyplot as plt
# slicing the merged Trade Finance DataFrame by indexing the column names:
df = Trade_Finance_merged[['Country_Code', 'Exports and imports (% of GDP)', 'FDI, net inflows (% of GDP)',
                           'Gross fixed capital formation (% of GDP)', 'Private capital flows (% of GDP)',
                           'Remittances, inflows (% of GDP)']]
# setting the Country_Code column in the merged DataFrame as the index:
df.set_index('Country_Code', inplace=True)
# defining the indicator column names as a list & assigning it to a variable label to be used later in the chart legend:
label = ['Exports & Imports', 'FDI (net inflows)', 'Gross fixed capital formation', 'Private capital flows',
         'Remittances (inflows)']

# plotting the merged DataFrame as a bar Chart:
df.plot.bar(figsize=(12, 8))
# adding the plot title:
plt.title('Trade & Financial Flow Levels among BRICS Members in 2019', fontsize=14)

# customizing the axes ticks:
plt.xticks(rotation=0)
plt.yticks(fontsize=11)

# adding the axes labels and customizing them:
plt.ylabel('Share of GDP (%)', fontsize=13)
plt.xlabel('BRICS member countries', fontsize=13)

# adding a legend to the plot and customizing it:
plt.legend(label, loc='upper left', title="Trade & Finance Flow Indicators", fontsize=12,framealpha=0.5)
# setting the y-axis limits of the plot:
plt.ylim(-10, 65)
# plotting a horizontal line at y=0 inorder to highlight the zero-value x-axis:
plt.axhline(y=0, color='r', linestyle='--')
# displaying the final plot:
plt.show()





### Calculating Human Development Index (HDI) Value for 2019 for each BRICS Member:-
### Using the HDRO Statistical Data API -
## Step 1 - Creating a BRICS HDI Value DataFrame as a sample at first:
BRICS_HDI_values = Health_dict2[["Country_Code","Year"]]
# printing the sample DataFrame created:
print(BRICS_HDI_values)

## Step 2 - Setting the Minimum and Maximum value Goalposts for each of the 3 Dimensions -
# loading the Pandas library (alias as pd):
import pandas as pd
# initializing a list of lists to create goalpost values for each HDI Dimension:
goalposts = [['Health', 'Life expectancy (years)', 20, 85], ['Education', 'Expected years of schooling (years)', 0, 18],
             ['Education', 'Mean years of schooling (years)', 0, 15],
             ['Standard of living', 'GNI per capita (2017 PPP$)', 100, 75000]]
# creating a pandas DataFrame for goalposts and assigning relevant column names:
goalposts_df = pd.DataFrame(goalposts, columns=['Dimension', 'Indicator', 'Minimum', 'Maximum'])
# printing the Goalposts DataFrame:
print(goalposts_df)

# assigning Health Indicator Goalposts to respective variables:
Health_Minimum_value = goalposts_df['Minimum'][0]
Health_Maximum_value = goalposts_df['Maximum'][0]

# assigning Education Indicator Goalposts to respective variables:
Edu_exp_yrs_school_Minimum_value = goalposts_df['Minimum'][1]
Edu_exp_Yrs_school_Maximum_value = goalposts_df['Maximum'][1]
Edu_mean_Yrs_school_Minimum_value = goalposts_df['Minimum'][2]
Edu_mean_Yrs_school_Maximum_value = goalposts_df['Maximum'][2]

# assigning Standard of living Indicator Goalposts to respective variables:
Living_Stand_Minimum_value = goalposts_df['Minimum'][3]
Living_Stand_Maximum_value = goalposts_df['Maximum'][3]


## Step 3 - Calculating each of the Dimension Indices -
# using formula: Dimension index = (actual value – minimum value) / (maximum value – minimum value)

# Dimension Index For Health -
# slicing the relevant columns from Health DataFrame and assigning it to variable Health_Indicators:
Health_Indicators = Health_dict2[["Country_Code","Year", "Life expectancy at birth"]]

# calculating the Health Dimension index (Life expectancy at birth) for each BRICS member using above formula:
Health_Index_BRICS = []
for i in range(5):
    Health_Dimension_Index = (Health_Indicators.iloc[i][2] - Health_Minimum_value) / (Health_Maximum_value - Health_Minimum_value)
    Health_Index_BRICS.append(Health_Dimension_Index.round(4))

# printing the Health Dimension index for all members:
print(Health_Index_BRICS)



# Dimension Index For Education -
# slicing the relevant columns from Education DataFrame and assigning it to variables Edu_Indicator_1
# and Edu_Indicator_2:
Edu_Indicator_1 = Educational_dict[["Country_Code", "Year", "Expected years of schooling (years)"]]
Edu_Indicator_2 = Educational_dict[["Country_Code", "Year", "Mean years of schooling (years)"]]

# calculating the Education Dimension index (Expected years of schooling) for each BRICS member using above formula:
Edu_Index_1_BRICS = []
for i in range(5):
    Edu_Dimension_Index_1 = (Edu_Indicator_1.iloc[i][2] - Edu_exp_yrs_school_Minimum_value) / (
                Edu_exp_Yrs_school_Maximum_value - Edu_exp_yrs_school_Minimum_value)
    Edu_Index_1_BRICS.append(Edu_Dimension_Index_1.round(4))

# calculating the Education Dimension index (Mean years of schooling) for each BRICS member using above formula:
Edu_Index_2_BRICS = []
for i in range(5):
    Edu_Dimension_Index_2 = (Edu_Indicator_2.iloc[i][2] - Edu_mean_Yrs_school_Minimum_value) / (
                Edu_mean_Yrs_school_Maximum_value - Edu_mean_Yrs_school_Minimum_value)
    Edu_Index_2_BRICS.append(Edu_Dimension_Index_2.round(4))

# calculating the Arithmetic Mean of the two resulting Education indices -
import numpy as np
Combined_Edu_Index_BRICS = []
# looping in values of each Education index to calculate its Arithmetic mean using Numpy mean function:
for i in range(5):
    Combined_Edu_Dimension_Index = np.mean((Edu_Index_1_BRICS[i],Edu_Index_2_BRICS[i]), dtype=np.float64)
# appending the calculated mean value to an empty list and rounding it to 2 decimal places:
    Combined_Edu_Index_BRICS.append(Combined_Edu_Dimension_Index.round(4))

# printing the Mean Education Dimension index for all members:
print(Combined_Edu_Index_BRICS)


# Dimension Index For Standard of Living -
# slicing the relevant columns from Income DataFrame and assigning it to variable Living_Stand_Indicator:
Living_Stand_Indicator = Income_dict2[["Country_Code","Year", "Gross national income (GNI) per capita"]]

# calculating the natural logarithm of the actual values for Living Standard Index and appending it to an empty list:
Living_Stand_Indicator_BRICS = []
for i in range(5):
    Living_Stand_Indicator_BRICS.append((np.log(Living_Stand_Indicator.iloc[i][2])).round(4))
# printing the calculated values appended to the list:
print(Living_Stand_Indicator_BRICS)
# calculating the natural logarithm of the minimum and maximum values for Living Standard Index:
Living_Stand_Minimum_value2 = np.log(Living_Stand_Minimum_value)
Living_Stand_Maximum_value2 = np.log(Living_Stand_Maximum_value)

# calculating the Living Standard Dimension index for each BRICS member using above formula:
Living_Stand_BRICS = []
for i in range(5):
    Living_Stand_BRICS.append(((Living_Stand_Indicator_BRICS[i] - Living_Stand_Minimum_value2) / (
                Living_Stand_Maximum_value2 - Living_Stand_Minimum_value2)).round(4))

# printing the Living Standard Dimension index for all members:
print(Living_Stand_BRICS)


## Step 4 - Aggregating the Dimensional Indices (Note - HDI is the geometric mean of the three dimensional indices):
# importing gmean function from scipy to calculate the geometric mean of indices:
from scipy.stats.mstats import gmean

# calculating the geometric mean of 3 dimensional indices for each of the BRICS members:
BRICS_HDI_2019 = []
for i in range(5):
    HDI = gmean([Health_Index_BRICS[i], Combined_Edu_Index_BRICS[i], Living_Stand_BRICS[i]])
    BRICS_HDI_2019.append(HDI.round(3))

# printing the calculated HDI values appended to the empty list:
print(BRICS_HDI_2019)

# appending the calculated HDI values as a ne column to the sample DataFrame created at first in Step 1:
BRICS_HDI_values['HDI Value'] = BRICS_HDI_2019
# finally printing the BRICS_HDI_values sample DataFrame:
print(BRICS_HDI_values)


## Step 4 - Calculating HDI Values using its Components
# Merging DataFrames to create a DataFrame of HDI values & its Components for BRICS members:
# Step 1 - merging Health Indicator to BRICS_HDI_values DataFrame:
HDI_and_Components = pd.merge(BRICS_HDI_values,Health_dict2[['Country_Code','Life expectancy at birth']],
                              on='Country_Code',how='left')

# Step 2 - merging Education Indicators to BRICS_HDI_values DataFrame:
HDI_and_Components2 = pd.merge(HDI_and_Components,Educational_dict[['Country_Code','Expected years of schooling (years)','Mean years of schooling (years)']],
                              on='Country_Code',how='left')

# Step 3 - merging Income Indicator to BRICS_HDI_values DataFrame:
BRICS_HDI_and_Components = pd.merge(HDI_and_Components2,Income_dict2[['Country_Code','Gross national income (GNI) per capita']],
                              on='Country_Code',how='left')

print(BRICS_HDI_and_Components)

# Sort Pandas DataFrame in descending order of HDI Values:
BRICS_HDI_and_Components2 = BRICS_HDI_and_Components.sort_values(by=['HDI Value'],ascending=False)
print(BRICS_HDI_and_Components2)


# Normalizing some columns of Dataframe to better visualize them and draw insights:
# Normalizing just a GNI per capitacolumn: normalizing GNI per capita in the range 0 to 1:
# rename the normalized value column of GNI per capita to GNI per capita_norm:
BRICS_HDI_and_Components["GNI per capita norm"] = ((BRICS_HDI_and_Components["Gross national income (GNI) per capita"] -
                                                    BRICS_HDI_and_Components[
                                                        "Gross national income (GNI) per capita"].min()) /
                                                   (BRICS_HDI_and_Components[
                                                        "Gross national income (GNI) per capita"].max() -
                                                    BRICS_HDI_and_Components[
                                                        "Gross national income (GNI) per capita"].min())) * 1

BRICS_HDI_and_Components["Life expectancy at birth norm"] = ((BRICS_HDI_and_Components["Life expectancy at birth"] -
                                                              BRICS_HDI_and_Components[
                                                                  "Life expectancy at birth"].min()) /
                                                             (BRICS_HDI_and_Components[
                                                                  "Life expectancy at birth"].max() -
                                                              BRICS_HDI_and_Components[
                                                                  "Life expectancy at birth"].min())) * 1

BRICS_HDI_and_Components["Expected years of schooling norm"] = ((BRICS_HDI_and_Components[
                                                                     "Expected years of schooling (years)"] -
                                                                 BRICS_HDI_and_Components[
                                                                     "Expected years of schooling (years)"].min()) /
                                                                (BRICS_HDI_and_Components[
                                                                     "Expected years of schooling (years)"].max() -
                                                                 BRICS_HDI_and_Components[
                                                                     "Expected years of schooling (years)"].min())) * 1

BRICS_HDI_and_Components["Mean years of schooling norm"] = ((BRICS_HDI_and_Components[
                                                                 "Mean years of schooling (years)"] -
                                                             BRICS_HDI_and_Components[
                                                                 "Mean years of schooling (years)"].min()) /
                                                            (BRICS_HDI_and_Components[
                                                                 "Mean years of schooling (years)"].max() -
                                                             BRICS_HDI_and_Components[
                                                                 "Mean years of schooling (years)"].min())) * 1

BRICS_HDI_and_Components_norm = BRICS_HDI_and_Components.drop(
    ['Life expectancy at birth', 'Expected years of schooling (years)',
     'Mean years of schooling (years)', 'Gross national income (GNI) per capita'], axis=1)

# displaying all columns of the Dataframe:
print(BRICS_HDI_and_Components_norm)


## Plotting the chart for HDI values & all its Components for various BRICS members:

# plotting HDI values of each member as a line plot in red:
ax = BRICS_HDI_and_Components_norm[['Country_Code', 'HDI Value']].plot(x='Country_Code', linestyle='-', marker='o',color='black',)

# plotting Normalized values of HDI Components as a bar plots:
BRICS_HDI_and_Components_norm[['Country_Code', 'GNI per capita norm','Life expectancy at birth norm',
                         'Expected years of schooling norm','Mean years of schooling norm']].plot(x='Country_Code', kind='bar',ax=ax,figsize=(8,6))

# adding labels & titles to plots:
plt.title('HDI Values and its Components across BRICS Members in 2019',fontsize=14)
plt.ylabel('HDI values & its components',fontsize=12)
plt.xlabel('BRICS member countries',fontsize=12)

# display legend by renaming the legend labels and adding title to legend:
plt.legend(['HDI Value','GNI per capita','Life expectancy at birth','Expected years of schooling', 'Mean years of schooling'],framealpha=0.5
           ,loc='center right',title='Components for calculating HDI values:-')

plt.show()


## Visualizing Trends in GDP per Capita for BRICS members (2015-19)
import pandas as pd

# Step 1 - Importing Data from HDI API putting it into Pandas DataFrame:
response = requests.get('http://ec2-54-174-131-205.compute-1.amazonaws.com/API/HDRO_API.php/country_code=BRA,RUS,IND,CHN,ZAF/indicator_id=194906/year=2015,2016,2017,2018,2019/structure=ciy')
Life_exp_birth = response.json()
my_dict30 = pd.DataFrame(Life_exp_birth['indicator_value'].items())
my_dict30.rename(columns={0:'Country_Code',1:'Health_Indicator'}, inplace=True)

# Step 2 - Putting it into Pandas DataFrame:
df_a = pd.DataFrame.from_dict((my_dict30['Health_Indicator'][0]),orient='index')
df_b = pd.DataFrame.from_dict((my_dict30['Health_Indicator'][1]),orient='index')
df_c = pd.DataFrame.from_dict((my_dict30['Health_Indicator'][2]),orient='index')
df_d = pd.DataFrame.from_dict((my_dict30['Health_Indicator'][3]),orient='index')
df_e = pd.DataFrame.from_dict((my_dict30['Health_Indicator'][4]),orient='index')

# Step 3 - Combining the individual DataFrames for each Country & Year:
frames = [df_a, df_b, df_c,df_d,df_e]
result = pd.concat(frames)
result['Country_Code'] = ['BRA','CHN','IND','RUS','ZAF']
result2 = result[['Country_Code','2015','2016','2017','2018','2019']]
result2.set_index('Country_Code',inplace=True)

# Step 4 - Transposing the DataFrame to show Country_Code as column headers:
new_result = result2.transpose().round(2)
print(new_result)


# Create a DataFrame for Boxplot Components - Minimum, Maximum, Median, First Quartile or 25% and Third Quartile or 75%:
BRICS_members = ['BRA','CHN','IND','RUS','ZAF']
BRICS_Dict = {'BRA': [],'CHN' : [], 'IND' : [], 'RUS' : [], 'ZAF' : []}

for i in BRICS_members:
    BRICS_Dict[i].append(new_result[i].min())
    BRICS_Dict[i].append(new_result[i].max())
    BRICS_Dict[i].append(new_result[i].median())
    BRICS_Dict[i].append(new_result[i].quantile(.25))
    BRICS_Dict[i].append(new_result[i].quantile(.75))
# Creating the DataFrame:
BRICS_boxplot = pd.DataFrame(BRICS_Dict)
# transposing the DataFrame to show Components as columns:
BRICS_boxplot_df = BRICS_boxplot.transpose()
# Setting the Column names of DataFrame:
BRICS_boxplot_df.columns =['Minimum','Maximum','Median', 'First Quartile','Third Quartile']
print(BRICS_boxplot_df)


# Step 5 - Visualizing Trends in GDP per Capita for BRICS members:

import matplotlib.pyplot as plt


fig, ax = plt.subplots(3,1, figsize=(12,10))

# plotting line plots for each Country_Code:
new_result['BRA'].plot(ax=ax[0], label='Brazil', x=new_result.iloc[0],marker='o',color='red',grid=True)
new_result['CHN'].plot(ax=ax[0], label='China', x=new_result.iloc[1],marker='o',color='green',grid=True)
new_result['IND'].plot(ax=ax[0], label='India', x=new_result.iloc[2],marker='o',color='blue',grid=True)
new_result['RUS'].plot(ax=ax[0], label='Russia', x=new_result.iloc[3],marker='o',color='black',grid=True)
new_result['ZAF'].plot(ax=ax[0], label='South Africa', x=new_result.iloc[4],marker='o',color='orange',grid=True)

ax[0].set_title('Line Plot & Box Plot for Trends in GDP per capita for BRICS Members (2015-19)',fontsize=14)
ax[0].set_ylabel('GDP per capita (2017 PPP) $',fontsize=12)

# customize the display different elements:
boxprops = dict(linestyle='-', linewidth=2, color='black')
medianprops = dict(linestyle='-', linewidth=2, color='black')

new_result.boxplot(column =['BRA','CHN','IND','RUS','ZAF'], grid = True, boxprops=boxprops,medianprops=medianprops,ax=ax[1])

#ax[1].set_ylabel('GDP per capita (2017 PPP) $',fontsize=12)

ax[0].legend(loc='upper right',title='BRICS members',framealpha=0.5)

# adding  a table with Box plot components for BRICS members:
table = BRICS_boxplot_df
cell_text = []
for row in range(len(table)):
    cell_text.append(table.iloc[row])

ax[2].table(cellText=cell_text, colLabels=table.columns, loc='center',rowLabels=BRICS_members,fontsize=13)
ax[2].axis('off')

plt.show()






## Data Source 3 - Excel File from HDI API Website:

import pandas as pd
# library to read xlsx format excel file:
import openpyxl

data = pd.read_excel('2020_Statistical_Annex_Table_2.xlsx',skiprows=3)

# Drop Columns with Missing Values:
data = data.dropna(how='all', axis=1)

# Drop Rows with Missing Values:
data = data.dropna(how='all',axis=0)

# Set Column headers for the DataFrame:
new_header = data.iloc[0]
data = data[2:]

data.columns = new_header
print(data.head(7))


BRICS_members = ['Brazil','China','India','Russian Federation','South Africa']

BRICS_df = data[data['Country'].isin(BRICS_members)]

country_code_list = ['RUS','BRA','CHN','ZAF','IND']
BRICS_df['Country_Code'] = country_code_list
BRICS_df.set_index('Country_Code',inplace = True)

# Deleting column 'a' from DataFrame:
del BRICS_df['a']
print(BRICS_df)

# Above code showing KeyError for year 2014 column:
print(BRICS_df.columns.tolist())

# Renaming column for year 2014 as it was giving KeyError due to incorrect format for column header:
BRICS_df.rename(columns={2014:'2014'},inplace=True)

print(BRICS_df.columns.tolist())


years = ['1990','2000','2010','2014','2015','2017','2018','2019']
Russia_data = BRICS_df.iloc[0,2:10]
Brazil_data = BRICS_df.iloc[1,2:10]
China_data = BRICS_df.iloc[2,2:10]
South_Africa_data = BRICS_df.iloc[3,2:10]
India_data = BRICS_df.iloc[4,2:10]

# Calculating percentage change between the current and a prior year for each country:
Russia_data2 = BRICS_df.iloc[0,2:10].pct_change()
Brazil_data2 = BRICS_df.iloc[1,2:10].pct_change()
China_data2 = BRICS_df.iloc[2,2:10].pct_change()
South_Africa_data2 = BRICS_df.iloc[3,2:10].pct_change()
India_data2 = BRICS_df.iloc[4,2:10].pct_change()


import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Chart 1 - Line Chart for trends in HDI Values of BRICS members during 1990-2019:
fig, ax = plt.subplots(2,1, figsize=(12,10))
ax[0].plot(years,Russia_data,label='RUS',marker='s',linestyle='-.')
ax[0].plot(years,Brazil_data,label='BRA',marker='s',linestyle='-.')
ax[0].plot(years,China_data,label='CHN',marker='s',linestyle='-.')
ax[0].plot(years,South_Africa_data,label='ZAF',marker='s',linestyle='-.')
ax[0].plot(years,India_data,label='IND',marker='s',linestyle='-.')
ax[0].legend(loc='lower right')
ax[0].set_title('Trends in HDI values among BRICS Members (1990-2019)',fontsize=14)
ax[1].set_xlabel('Years',fontsize=13)
ax[0].set_ylabel('HDI values',fontsize=13)

ax[1].plot(years,Russia_data2,label='RUS',marker='s')
ax[1].plot(years,Brazil_data2,label='BRA',marker='s')
ax[1].plot(years,China_data2,label='CHN',marker='s')
ax[1].plot(years,South_Africa_data2,label='ZAF',marker='s')
ax[1].plot(years,India_data2,label='IND',marker='s')
ax[1].legend(loc='upper right')
ax[1].set_title('Trends in HDI values Changes (%) among BRICS Members (1990-2019)',fontsize=14)
ax[1].set_ylabel('Change in HDI values',fontsize=13)

# setting the y-axis ticks in percent format for the second subplot:
vals = ax[1].get_yticks()
ax[1].set_yticklabels(['{:,.1%}'.format(x) for x in vals])


plt.show()



