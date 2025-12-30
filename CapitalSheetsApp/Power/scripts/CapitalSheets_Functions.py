#Imports are declared when needed once

#Load Training Data from pickled files
import pickle
import re
import os



def load_training_data():
        
        balance_sheets, income_sheets, cashflow_sheets, wrong_tables = [],[],[],[]

        for _ in os.listdir('Training Data\\Trained Data'):
            
            if re.findall(r'\w+ Balance .+',_):
                with open(f"F:\\PythonProjects\\Machine Learning\\SEC ML Data\\Training Data\\Trained Data\\{_}","rb") as balance:
                    balance_sheets.append(pickle.load(balance))
                
            if re.findall(r'\w+ Income .+',_):
                with open(f"F:\\PythonProjects\\Machine Learning\\SEC ML Data\\Training Data\\Trained Data\\{_}","rb") as income:
                    income_sheets.append(pickle.load(income))
                    
            if re.findall(r'\w+ Cashflow .+',_):
                with open(f"F:\\PythonProjects\\Machine Learning\\SEC ML Data\\Training Data\\Trained Data\\{_}","rb") as cashflow:
                    cashflow_sheets.append(pickle.load(cashflow))
                    
            if re.findall(r'\w+ Wrong Tables .+',_):
                with open(f"F:\\PythonProjects\\Machine Learning\\SEC ML Data\\Training Data\\Trained Data\\{_}","rb") as wrong:
                    wrong_tables.append(pickle.load(wrong))

        return balance_sheets, income_sheets, cashflow_sheets, wrong_tables

#Class acts as a list holder for the selected company data
class ExtractedData:
    def __init__(self):
        self.balance_sheets = []
        self.income_sheets = []
        self.cashflow_sheets = []


from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import pandas as pd
def sec_table_formatter(website):#Pull HTML data/ May be overkill as SEC Edgar are static HTML reports
    chrome_options = Options()
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--block-new-web-contents")
    driver = webdriver.Chrome()
    driver.get(website)
    tables = pd.read_html(driver.page_source)
    driver.quit() #free up driver

    return tables


def pre_process_1(dataframe): #NA Processing
        pd.set_option("future.no_silent_downcasting", True)
        dropna_1 = dataframe.dropna(how='all', axis=0, inplace=False)#Drop any row that is empty
        dropna_2 = dropna_1.dropna(how='all', axis=1, inplace=False)# Drop any column that is empty
        fillna_1 = dropna_2.bfill(axis=1, inplace=False)#Fill any empty cells backwards, applies mostly to first cell in row
        fillna_2 = fillna_1.ffill(axis=1, inplace=False)#Fill any empty cells forward, applies mostly to the last cell
        #Remove emojis/symbols by trying to convert into ascii, not possible, so ignore
        emoji_clean = fillna_2.apply(lambda x: x.astype(str).str.encode('ascii', 'ignore'))#.str.decode('ascii'))
        white_rep = emoji_clean.infer_objects(copy=False)#infer datatypes and convert them to   
        fillna_3 = white_rep.ffill(axis=1, inplace=False)#Front fill after processing to top it off 
        fillna_4 = fillna_3.bfill(axis=1, inplace=False)#Back fill after processing to top it off
        return fillna_4
    
def pre_process_2(dataframe): #Format Procesing
        lowerc_df = dataframe.apply(lambda x: x.astype(str).str.lower())#lower case all values as strings
        rep_symbl_1 = lowerc_df.apply(lambda x: x.astype(str).str.replace('\\(','-',regex=True)) #replace ( as - to show as negative number
        rep_symbl_2 = rep_symbl_1.apply(lambda x: x.astype(str).str.replace('\\)','',regex=True)) #remove all ), serves no purpose
        reset_index = rep_symbl_2.reset_index(drop=True).T.reset_index(drop=True)#Reset the index for both rows and columns
        column_dict = reset_index.T.T #Transpose twice?
        try: #try column names set
            column_dict.columns =  column_dict.iloc[0]#column names equal to first row
            processed_df = column_dict[1:]#dataframe now is starting from second row and onward to get rid of the first row column names
            processed_df = processed_df.drop_duplicates()[1:]#drop all duplicates from new second row
            return processed_df.drop_duplicates()#return dataframe with droppped duplicates
        except:
            pass #

def prepare_training_data():
    dir_balance_sheets, dir_income_sheets, dir_cashflow_sheets, dir_wrong_tables = [], [], [], []
    import os
    import random
    
    for file in os.listdir('Training Data\\Trained Data'):
        if re.findall(r'\w+ Balance .+', file):
            with open(f"Training Data\\Trained Data\\{file}", "rb") as f:
                dir_balance_sheets.append(pickle.load(f))
        elif re.findall(r'\w+ Income .+', file):
            with open(f"Training Data\\Trained Data\\{file}", "rb") as f:
                dir_income_sheets.append(pickle.load(f))
        elif re.findall(r'\w+ Cashflow .+', file):
            with open(f"Training Data\\Trained Data\\{file}", "rb") as f:
                dir_cashflow_sheets.append(pickle.load(f))
        elif re.findall(r'\w+ Wrong Tables .+', file):
            with open(f"Training Data\\Trained Data\\{file}", "rb") as f:
                dir_wrong_tables.append(pickle.load(f))
    
    # Process tables
    processed1_bal_sheets = [pre_process_1(y) for x in dir_balance_sheets for y in x]
    processed1_cashflow_sheets = [pre_process_1(y) for x in dir_cashflow_sheets for y in x if not isinstance(y, list)]
    processed1_income_sheets = [pre_process_1(y) for x in dir_income_sheets for y in x]
    
    flat_wrong_tables = [y for x in dir_wrong_tables for y in x]
    random.shuffle(flat_wrong_tables)
    equal = (len(processed1_bal_sheets) + len(processed1_income_sheets) + len(processed1_cashflow_sheets)) * 2 + 1
    processed1_wrong_tables = [pre_process_1(pd.DataFrame(x, index=[0])) for x in flat_wrong_tables[:equal]]
    
    processed2_bal_sheets = [pre_process_2(x) for x in processed1_bal_sheets]
    processed2_cashflow_sheets = [pre_process_2(x) for x in processed1_cashflow_sheets]
    processed2_income_sheets = [pre_process_2(x) for x in processed1_income_sheets]
    processed2_wrong_tables = [pre_process_2(x) for x in processed1_wrong_tables]
    
    # Remove empty tables
    processed2_wrong_tables = [x for x in processed2_wrong_tables if not pd.DataFrame(x).empty]
    
    # Convert to table-level texts
    table_texts_bal = [extract_table_text(df) for df in processed2_bal_sheets]
    table_texts_inc = [extract_table_text(df) for df in processed2_income_sheets]
    table_texts_cash = [extract_table_text(df) for df in processed2_cashflow_sheets]
    table_texts_wrong = [extract_table_text(df) for df in processed2_wrong_tables]
    
    # Balance classes
    min_len = min(len(table_texts_bal), len(table_texts_inc), len(table_texts_cash), len(table_texts_wrong))
    table_texts_bal = table_texts_bal[:min_len]
    table_texts_inc = table_texts_inc[:min_len]
    table_texts_cash = table_texts_cash[:min_len]
    table_texts_wrong = table_texts_wrong[:min_len * 2]  # Oversample Wrong_Table
    
    texts = table_texts_bal + table_texts_inc + table_texts_cash + table_texts_wrong
    labels = (['Balance_Sheet'] * len(table_texts_bal) +
              ['Income_Sheet'] * len(table_texts_inc) +
              ['Cashflow_Sheet'] * len(table_texts_cash) +
              ['Wrong_Table'] * len(table_texts_wrong))
    
    return texts, labels
def FindData(url):#Extracts HTML column names of all tables in the url
    words = [] #Words holder list
    data_tables = sec_table_formatter(url) #Find HTML tables
    processed_tables = [pre_process_2(pre_process_1(y)) for y in data_tables] #process html tables

    for table in processed_tables:#loop each table
        word_holder = " " #string buffer that will hold complete column names
        if (table is None): #self explanatory
            pass
        else:
            word_list = [x for x in table.columns]#create a list of column names as strings
            word_holder = word_holder.join(word_list)#join the column names into the string word_holder
            words.append(word_holder)# append the column names into the main words list

    return words, processed_tables #important return is words list as its a string of the column names


def FoundData(results):
    #Take list of results after prediction and seperate into corresponding lists and return the 3 main lists
    counter = -1
    bs_list = []
    inc_list = []
    cf_list = []
    
    for _ in results:
        counter += 1
        if _ == "Balance_Sheet":
            bs_list.append(counter)

        elif _ == "Income_Sheet":
            inc_list.append(counter)

        elif _ == "Cashflow_Sheet":
            cf_list.append(counter)

    return bs_list, inc_list, cf_list  # Modified to return lists directly for easier use


from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
#New extract words function
def extract_table_text(table_df):
    lemmatizer = WordNetLemmatizer()
    #Customize for stopping words for SEC 10-K, filter for meaningless(common) words in reports
    financial_stopwords = set(stopwords.words('english')) | {'total', 'amount', 'period', 'year', 'ended', 'as of'}



    #Extract cleaned text from entire table as one document not word tokens
    text = ' '.join(table_df.astype(str).values.flatten())#Convert to list, flatten all cells
    text = re.sub(r'\d+\.?\d*', '', text) #Remove numbers, no need as indexing is what selects tables by its headers/columns/rows
    text = re.sub(r'[^\w\s]',' ', text) #remove punctuation
    words = [lemmatizer.lemmatize(w.lower()) for w in text.split() if w.lower() not in financial_stopwords and len(w) > 2]

    return ' '.join(set(words))#create unique words to avoid repeats


def Capital_Functions_Called():
    print("Capital Functions Called\n")

if __name__ == '__main__':
    Capital_Functions_Called()