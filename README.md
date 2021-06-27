## UCDPA_SukritiShukla

# **Understanding the Human Development Index (HDI) among BRICS Members - analysing Real-World Scenarios**

## (Visualizing Insights using Python)

UCD Professional Academy - Professional Certificate in Data Analytics **Final Project**
[(Link to Report)](https://www.linkedin.com/in/sukriti-shukla-3989a819/detail/treasury/education:710951249/?entityUrn=urn%3Ali%3Afsd_profileTreasuryMedia%3A(ACoAAAPiXiQBwXwVR_rmub9C1DokEKlxJ1p6JUA%2C1615292196878)&section=education%3A710951249&treasuryCount=1)


Prepared by - [Sukriti Shukla](https://www.linkedin.com/in/sukriti-shukla-3989a819/)

![pexels-porapak-apichodilok-346885](https://user-images.githubusercontent.com/50409210/110473997-11ce7880-80d7-11eb-942a-1141ac277a61.jpg)

Photo by Porapak Apichodilok from Pexels

## Introduction 

[Human Development Index (HDI)](https://en.wikipedia.org/wiki/Human_Development_Index), was created by the United Nations Development Program (UNDP) to emphasize that people and their capabilities should be the ultimate criteria for assessing the development of a country, not economic growth alone. 

The HDI is a summary measure of average achievement in key dimensions of human development: 
* a long and healthy life
* being knowledgeable and 
* have a decent standard of living

It is an interesting statistic compiled by the United Nations to measure the level of socio-economic development of countries across the globe. 

   [BRICS](https://en.wikipedia.org/wiki/BRICS) is an acronym coined to associate 5 major emerging national economies comprising - the Federative Republic of Brazil, the Russian Federation, the Republic of India, the People’s Republic of China, and the Republic of South Africa. It is an Economic & Political Regional organisation founded in 2009. BRICS countries are influential members of leading international organizations and agencies, including the UN, the G20 and many more. 

Both HDI and BRICS were chosen as the focal points of my analysis because of the growing economic might of BRICS countries, their significance as one of the main driving forces in global economic development, their substantial market share and their abundant natural resources. All these factors form the foundation of their influence on the international scene today and shape up their contributions in future. Therefore, the purpose of this project was to study the Human Development Index (HDI) Values and its component indicators particularly for the BRICS member countries. I identified some key economic indicators from HDI to study for each data source and dived deep into analysing them.

Summarizing the insights drawn about BRICS members through this project helped me illustrate how different their levels of socio-economic development were despite being part of a single regional alliance. It also gave an overview of how these emerging economies are progressing and allowed me to highlight the specific economic indicators on which their  governments should focus to improve the ‘quality of life’ of its residents. This would ultimately help them in improving their HDI Values and rankings in the future.


## Data Sources and Preparation 

I performed factor analysis on my data both for a point in time (like a specific year) and for a time-series (like a duration of five or ten years). I also used quantitative data analysis methods using Python programming. Statistical analysis techniques like normalization were then used to align the scales of different indicators and perform exploratory data analysis on the data. Once the data was pre-processed, a series of viualizations were created from which I drew specific insights about the levels of development across various BRICS members and how they differ. My study also involved comparing Regional vs. BRICS member’s contributions and analysing their performance changes over a period. 

For this project I primarily used two sources of data in my project - 
1. Country-specific statistics from the [Country Statistics UNData](https://www.kaggle.com/sudalairajkumar/undata-country-profiles) dataset shared on Kaggle (used only one of      the csv files - *country_profile_variables.csv*)
2. Human Development Reports’ Datasets [HDRO Portal](http://hdr.undp.org/en/content/human-development-index-hdi) - The Human Development Reports portal which is maintained by      the United Nations Development Program (UNDP) publishes data related to the periodically released Human Development Reports. It offers data in many forms (an API,                downloadable excel or pdf files) and has a dedicated Data Center also.

  For my project, I used data in 2 formats on this portal in my analysis & visualizations: -
  
   i. [Human Development Report Office Statistical Data API](http://hdr.undp.org/en/content/human-development-report-office-statistical-data-api) - 
      The Human Development Report Office (HDRO) offers a REST API for the developers to query human development related data in JSON format. The data can be queried by               indicator id(s), year(s) and country code(s) and grouped by any order. 
   
  ii. [Human Development Reports: Data Center – Download Data](http://hdr.undp.org/en/content/download-data) – 
      This part of the Data Center allowed me to download the data in Excel file format (downloaded the Excel file named Table 2: Trends in the Human Development Index,               1990-2019 [link](http://hdr.undp.org/en/composite/trends) as *‘2020_Statistical_Annex_Table_2.xlsx’* and performed further analysis on it).



## Visualizations & Insights Drawn 

I performed a macro-economic analysis of the above indicators among BRICS member countries’ and compared them with their respective Regions as well. I made a series of visualizations using the two datasets and my objective through these set of visualizations was to get a broader understanding of each of the BRICS member’s economy and where it stands in comparison to others. Below are some of the visualizations created in this project using Python visualization libraries such as Matplotlib and Seaborn.


 **Chart 1 - A Comparative Visualization of BRICS member Data with their respective Regions in Bar Chart Subplots for the following set of indicators** 
  * Population in thousands (2017) 
  * Population density (per km2, 2017) 
  * GDP per capita (current US$) 

![Chart 1](https://user-images.githubusercontent.com/50409210/123551846-75a2ad00-d76b-11eb-9ab7-b5c9612f7d08.png)

The following set of insights can be drawn from the above visualization: -
* Insight 1 – China & India have the highest Total Population in 2017 among BRICS members leading to the highest Regional Total Population for Eastern Asia (China’s region) and   Southern Asia (India’s region) regions, respectively.
* Insight 2 – Although India as a country has the highest Population Density (per sq. km) among BRICS members in 2017, its region Southern Asia does not have extremely high       density of population.
* Insight 3 – Although, Russia has the highest GDP per capita, followed by Brazil among BRICS countries in 2017, but at a Regional level Eastern Asia has the largest share of     GDP per capita compared to other regions. One major reason for this can be the contribution of Chinese economy to its region (Eastern Asia).



**Chart 2 - Visualizing the Contribution of different Sectors to each BRICS member’s economy (2017) as a Stacked Bar Chart**

Sectoral contributions form an important part of every economy and viewing the share of 3 major sectors (Agriculture, Industry & Services (and others)) to the % Gross Value Added (GVA) of their countries was crucial to study their current economic structure and determine its future. All the sectors in an economy were stacked on top of other and each sector represented in a different color for ease of comparison. To determine the values easily I also labeled the bar stacks with respective values.

![Chart 3](https://user-images.githubusercontent.com/50409210/123552051-4d677e00-d76c-11eb-8ffc-04181cc527b7.png)

The following set of insights can be drawn from the above visualization: -
* Insight 1 – Among BRICS members, the Agriculture sector contributes to India (17%) the maximum share of its Gross Value Added (GVA) in 2017.
  China appears to be an Industry-centric economy with largest share (41.1%) among BRICS members in 2017, while Brazil is primarily Services sector-based as it contributes the     largest share (72.0%) to its total Gross Value Added in 2017.
* Insight 2 – In 2017, among all the BRICS member’s Services sector appears to have the largest contribution to Gross Value Added (GVA) which shows growing importance of this     sector across the BRICS economies.


**Chart 3 - Visualizing International Trade Indicators for BRICS members (2017) as a Horizontal Bar Plot**

![Chart 4](https://user-images.githubusercontent.com/50409210/123552092-7daf1c80-d76c-11eb-83dc-29a392042940.png)

The following set of insights can be drawn from the above visualization: -
* Insight 1 – All the 4 trade components - Exports, Imports, Balance of Trade and Balance of Payment (Current) are highest for China in 2017 among BRICS countries.
* Insight 2 – As Imports for India exceed Export it is the only BRICS member which has a Trade Deficit i.e., a negative net balance of trade in 2017. Therefore, this point has been annotated using a black arrow and text specifying ‘Trade Deficit’ as it meant a negative Balance of Trade.


**Chart 4 - Seaborn Visualization of the Distribution of GDP per Capita across Continents as a Boxplot**

To do this, I created 5 buckets of *‘Continents’* (as shown in Bold below), grouped and assigned different regions into each Continent bucket. This helped me group all my country-level data from the data file into each of the 5 Continent bucket & visualize their distribution

![Chart 11](https://user-images.githubusercontent.com/50409210/123552149-bea73100-d76c-11eb-874f-ea9790e3b7f5.png)

The following set of insights can be drawn from the above visualization: -
* Insight 1 – The continent of Europe appears to have the largest distribution of GDP per capita values while the continent of Africa has the least. This can be an important       indicator to gauge the prosperity of nations across continents. As can be seen here, it appears that Europe comprises the most prosperous countries of the world while Africa     the least. 
* Insight 2 – If we study the outliers in each continent what appears here is that African continent has outliers concentrated extremely near to its Maximum value line on the     boxplot. as opposed to the continent of America. This can indicate that most of the countries in Africa have the almost the same level of prosperity and economic growth and     there is very little difference or outliers.


**Chart 5 - Visualizing the level of Trade & Financial Flows among BRICS Members in 2019**

![Chart 7](https://user-images.githubusercontent.com/50409210/123552214-147bd900-d76d-11eb-9d0d-0117c6f01801.png)

The following set of insights can be drawn from the above visualization: - 
* Insight 1 – Brazil has the highest level of Foreign Direct Investment (FDI) net inflow as a share of GDP among all the BRICS members in 2019 while South Africa has the largest   level of Exports & Imports as a share of GDP in 2019. 
* Insight 2 – All the BRICS member countries show a negative Private Capital Flow in 2019. Private capital flows include direct and portfolio investment made by locals of a       country living abroad and foreigners living in the country. Negative Private Capital Flow or its outflow is considered undesirable as it is often the result of political or     economic instability.
* Insight 3 - India is the only country showing some Remittances (inflows) as a share of its GDP in 2019, although its level is extremely low. Remittances are Earnings and         material resources transferred by international migrants or refugees to recipients in their country of origin or countries in which the migrant formerly resided.


**Chart 6 - Visualizing HDI values & all its Dimension Components for various BRICS members in 2019**

This visualization depicts a graphical summary of HDI Values and its components for each of the BRICS members. HDI Value data points have been depicted as a line plot in black with each marker representing the exact HDI value. Various HDI components used in calculating HDI value are shown as colored bars. Through this chart I have compared HDI components among BRICS countries in combination with their HDI Value levels.

![Chart 8](https://user-images.githubusercontent.com/50409210/123552261-455c0e00-d76d-11eb-9cc4-42da35262e4b.png)

The following set of insights can be drawn from the above visualization: - 
* Insight 1 – In the year 2019, Russia has the highest HDI Value 0.824 among the BRICS member countries. This is also because most of HDI components, like GNI per capita and       Mean years of schooling are also among the highest or second highest for Russia.
* Insight 2 – India has the lowest HDI Value, 0.645 among the BRICS members in 2019 and this is also evident from the level of almost all its HDI components being exceptionally   low. This graph here also can show only 1 component for India i.e., Life expectancy at birth), rest all values invisible here (after normalization).


**Chart 7 - Visualizing Trends in GDP per capita (2017 PPP $) for BRICS members during 2015-19**

This is a complex visualization displaying 3 different types of charts as Subplots in one visualization. There is a Line plot for each BRICS country represented by a specific color showing the trends in GDP per capita level during 2015-19. Every member country has a boxplot showing the distribution of GDP per capita as boxplot components (minimum, median, maximum, lower & upper quartiles). The Table below boxplot shows the numerical values of each Boxplot component for every country.

![Chart 9](https://user-images.githubusercontent.com/50409210/123552292-6c1a4480-d76d-11eb-808d-4d358b1a5ce4.png)

The following set of insights can be drawn from the above visualization: - 
* Insight 1 – During 2015-19 Russia had the highest GDP per capita values among the BRICS member countries, while India had the lowest GDP per capita during this period. We can   see this in the line plot, box plot & table chart.
* Insight2 – As can be seen from the Boxplot and the table below the box plot China has the maximum variability in the range of values as can also be seen from its Maximum value   (16116.70) and its Minimum value (12691.94) while South Africa has the least variability in its values as evident from its Maximum value (12840.04) & Minimum values             (12481.81).


**Chart 8 - Trends in HDI Values among BRICS members (1990-2019)**
![Chart 10](https://user-images.githubusercontent.com/50409210/123552381-d03d0880-d76d-11eb-8eac-05c9c4a625f5.png)

The following set of insights can be drawn from the above visualization: - 
* Insight 1 – China had the steepest rise in HDI values during the period 1990-2019 followed by India. This rise in HDI values can be seen for both China & India till the year     2010 as visible in Growth in HDI values chart (subplot 2). But, after 2010, both these countries faced a decline in HDI values during 2012-2014 with China’s fall being sharper   than India’s.
* Insight 2 – Brazil also faced a sharp fall in HDI values in 2010 and after 2015, there was little movement in Brazil’s HDI values.



## Conclusion 

By performing Data Analysis in Python and Visualizing different parts of HDI-related data I was able to get a better understanding about each of the BRICS members. It also helped me get acquainted with the Human Development Index and the various indicators contributing to its calculation. I could draw insights about the emerging economies of BRICS countries and find out where they stand today in terms of their socio-economic development levels and how was their journey over time.





