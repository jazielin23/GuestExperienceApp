import pandas as pd
import streamlit as st
import config
from scipy.stats import pointbiserialr
import numpy as np


###############################################################################################################################
def concatenate_columns_and_first_row(df):
   
   """
   This function takes a DataFrame as input and returns 
   a list where each element is a concatenation of 
   column names and their corresponding first-row values, separated by '--'.
   
   Parameters:
       df (pd.DataFrame): The input DataFrame.
       
   Returns:
       col_list (list): A list of concatenated strings.
   """

   # Get column names
   col_names = df.columns
   
   # Get first row values
   first_row_values= df.iloc[0,:].tolist()
   
   # Concatenate elements from both lists with "--" separator  
   combined_elements=[ f"{value}--{column_name}" for (value,column_name) in zip(first_row_values,col_names)]
#    print(combined_elements)

   return combined_elements 


###############################################################################################################################
# below function can be used to get back the original lists from the list created by function 'concatenate_columns_and_first_row'

def split_list(list1):
    """
    Splits elements of list1 by '--' and returns two separate lists.
    
    Args:
        list1 (list): A list of strings where each string contains '--'.
        
    Returns:
        tuple: Two lists containing split parts before and after '--'.
    """
    # Initialize two empty lists for storing results
    list_a = []
    list_b = []

    # Iterate through each item in list1, split by '--', and append to respective lists
    for item in list1:
        part_a, part_b = item.split("--")
        list_a.append(part_a)
        list_b.append(part_b)

    return list_a, list_b



###############################################################################################################################
# do encoding of dataset1 based on dataset2
  
def encode_dataset(dataset1, dataset2):
    # Convert datasets to DataFrames
    df1 = pd.DataFrame(dataset1)
    df2 = pd.DataFrame(dataset2)

    # Create a dictionary for encoding based on dataset2
    encoding_dict = {}
    
    for _, row in df2.iterrows():
        col_name = row['Question Description']
        unique_value = row['Unique Values']
        coded_value = row['Coded Values']
        
        if col_name not in encoding_dict:
            encoding_dict[col_name] = {}
        
        if pd.isna(unique_value) or unique_value == '':
            continue
        
        encoding_dict[col_name][unique_value] = coded_value
    
    # Encode dataset1 using the created dictionary
    for col in df1.columns:
        if col in encoding_dict:
            df1[col] = df1[col].map(encoding_dict[col]).fillna(0).astype(int)
    
    return df1

###############################################################################################################################

# Question Keys and Question Description data set
def transform_sample(df):
    df_t = df.iloc[0:1].T
    df_t.insert(1, 'Column_names', df.columns.values)
    df_t.columns = ["Question Keys", "Question Description"]
    return df_t.reset_index()


def unique_values_df(unique_values_df):
    # Initialize an empty list to store rows
    rows = []

    # Iterate through sorted unique values and append each pair to rows list
    for q_code, values in unique_values_df.items():
        for value in values:
            rows.append({'q_code': q_code, 'Unique Values': value})

    # Create DataFrame from rows list
    df = pd.DataFrame(rows).drop_duplicates()
    
    return df

def map_values(option):
    
     # Check if option is NaN or blank and return zero if true 
     if pd.isna(option) or option == '':
         return 0
      
     # Return mapped value if exists in dictionary else return original value 
     return config.mapping_conditions.get(option,option)

# Function to identify unique options for each categorical column (question)
def get_unique_options(df):
    unique_options = {}
    for col in df.select_dtypes(include=['object']).columns:
        unique_options[col] = df[col].unique().tolist()
    return unique_options

def display_custom_feature_engineering():
    
    st.write("<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> Fill the required details in below widgets to create new variables </h1>", unsafe_allow_html=True)
    # st.subheader("Aggregate Columns")
    columns_to_aggregate = st.multiselect("Select columns to aggregate:", st.session_state.data.columns.tolist())
    aggregation_operation = st.selectbox("Select aggregation operation:", ["Sum", "Mean", "Min", "Max"])
    new_feature_name = st.text_input("Name for the new aggregated feature:")
    new_feature_description = st.text_input("Description for the new feature:")

    if st.button("Create New Feature"):
        if not columns_to_aggregate or not aggregation_operation or not new_feature_name:
            st.write("Fill the required details")
        else:
            if aggregation_operation == "Sum":
                st.session_state.data[new_feature_name] = st.session_state.data[columns_to_aggregate].sum(axis=1)
            elif aggregation_operation == "Mean":
                st.session_state.data[new_feature_name] = st.session_state.data[columns_to_aggregate].mean(axis=1)
            elif aggregation_operation == "Min":
                st.session_state.data[new_feature_name] = st.session_state.data[columns_to_aggregate].min(axis=1)
            elif aggregation_operation == "Max":
                st.session_state.data[new_feature_name] = st.session_state.data[columns_to_aggregate].max(axis=1)

            #### coding the new feature to 1 if its 1, else 0 
            st.session_state.data[new_feature_name] = st.session_state.data[new_feature_name].apply(lambda x: 1 if x == 1 else 0)

            top_columns = [col + "_TOP" for col in columns_to_aggregate if col + "_TOP" in st.session_state.data.columns]
            top2_columns = [col + "_TOP2" for col in columns_to_aggregate if col + "_TOP2" in st.session_state.data.columns]

            if top_columns:
                if aggregation_operation == "Sum":
                    st.session_state.data[new_feature_name + "_TOP"] = st.session_state.data[top_columns].sum(axis=1)
                elif aggregation_operation == "Mean":
                    st.session_state.data[new_feature_name + "_TOP"] = st.session_state.data[top_columns].mean(axis=1)
                elif aggregation_operation == "Min":
                    st.session_state.data[new_feature_name + "_TOP"] = st.session_state.data[top_columns].min(axis=1)
                elif aggregation_operation == "Max":
                    st.session_state.data[new_feature_name + "_TOP"] = st.session_state.data[top_columns].max(axis=1)

                st.session_state.data[new_feature_name + "_TOP"] = st.session_state.data[new_feature_name + "_TOP"].apply(lambda x: 1 if x == 1 else 0)

            if top2_columns:
                if aggregation_operation == "Sum":
                    st.session_state.data[new_feature_name + "_TOP2"] = st.session_state.data[top2_columns].sum(axis=1)
                elif aggregation_operation == "Mean":
                    st.session_state.data[new_feature_name + "_TOP2"] = st.session_state.data[top2_columns].mean(axis=1)
                elif aggregation_operation == "Min":
                    st.session_state.data[new_feature_name + "_TOP2"] = st.session_state.data[top2_columns].min(axis=1)
                elif aggregation_operation == "Max":
                    st.session_state.data[new_feature_name + "_TOP2"] = st.session_state.data[top2_columns].max(axis=1)

                st.session_state.data[new_feature_name + "_TOP2"] = st.session_state.data[new_feature_name + "_TOP2"].apply(lambda x: 1 if x == 1 else 0)
            if 'percent1_table' not in st.session_state:
                st.session_state.percent1_table = pd.DataFrame(columns=["Feature", "Actual %"])
            for col in [new_feature_name, new_feature_name + "_TOP", new_feature_name + "_TOP2"]:
                if col in st.session_state.data.columns:
           
                    percent_1 = (st.session_state.data[col].sum() / len(st.session_state.data)) * 100
                    new_row_1 = pd.DataFrame([{"Feature": col, "Actual %": percent_1}])
                    
                    
                    # Check if feature already exists to avoid duplication
                    if col not in st.session_state.percent1_table["Feature"].values:
                        st.session_state.percent1_table = pd.concat(
                            [st.session_state.percent1_table, new_row_1], ignore_index=True
                        )

                    # st.write(st.session_state.percent1_table)
                    new_row = {
                        'Column Names': col,
                        'Custom Categorization': 'explainer'
                    }
                    if 'final_flags' in st.session_state:
                        if col not in st.session_state.final_flags['Column Names'].values:
                            st.session_state.final_flags = pd.concat([
                                st.session_state.final_flags,
                                pd.DataFrame([new_row])
                            ], ignore_index=True)
                    else:
                        # Create final_flags if it doesn't exist
                        st.session_state.final_flags = pd.DataFrame([new_row])

            # if 'new_feature_description_table' not in st.session_state:
            #     st.session_state.new_feature_description_table = pd.DataFrame(columns=["Independent Variables", "Description"])
            desc_rows = pd.DataFrame([
                        {"Independent Variables": new_feature_name, "Description": new_feature_description},
                        {"Independent Variables": new_feature_name + "_TOP", "Description": new_feature_description},
                        {"Independent Variables": new_feature_name + "_TOP2", "Description": new_feature_description},
                    ])
            existing_features = st.session_state.new_feature_description_table["Independent Variables"].values if "new_feature_description_table" in st.session_state else []

            desc_rows = desc_rows[~desc_rows["Independent Variables"].isin(existing_features)]

            # Initialize table if it doesn't exist
            if 'new_feature_description_table' not in st.session_state:
                st.session_state.new_feature_description_table = pd.DataFrame(columns=["Independent Variables", "Description"])

            # Append new descriptions
            st.session_state.new_feature_description_table = pd.concat([
                st.session_state.new_feature_description_table,
                desc_rows
            ], ignore_index=True)
            # st.write(st.session_state.new_feature_description_table)

            
            st.success(f"New feature '{new_feature_name}' created!")
            


def column_segregator(df):
    # Identify columns matching the descriptions
    cols_part1 = [col for col in df.columns if any(desc in col.lower() for desc in config.col_descs)]
    
    # Create two separate DataFrames
    df_part1 = df[cols_part1]
    df_part2 = df.drop(columns=cols_part1)
    
    return df_part1, df_part2

def transform_data(data1, data2):
    # Convert data1 and data2 into DataFrames
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    # Iterate through each column in df1
    for col in df1.columns:
        if col not in df2['Question Description'].values:
            continue
        
        # Get unique values and coded values from df2 for the current column
        mapping_df = df2[df2['Question Description'] == col]
        
        # Create a dictionary for mapping unique values to coded values
        value_map = dict(zip(mapping_df['Variable Rating Scale'], mapping_df['Coded Values']))
        # Replace values in df1 using the created map
        df1[col] = df1[col].map(value_map).fillna(df1[col])
    return df1

def transform_data1(df):
    # Initialize an empty list to store results
    result = []

    # Iterate over each column in the dataframe
    for col in df.columns:
        # Get unique values and their counts
        value_counts = df[col].value_counts()
        
        # Calculate distribution percentages
        total_count = len(df)
        distribution = (value_counts / total_count) * 100
        
        # Append results to the list
        for value, count in value_counts.items():
            result.append([col, value, count, round(distribution[value], 1)])
    
    # Create a new dataframe from the results list
    transformed_df = pd.DataFrame(result, columns=['Question Description', 'Variable Rating Scale', 'Count', 'Distribution Percentage'])
    
    return transformed_df

def biserial_correlation(df, target_col):
    correlations = {}
    for col in df.columns:
        if col != target_col:
            corr, _ = pointbiserialr(df[target_col], df[col])
            correlations[col] = corr
    return correlations


# Define the sigmoid function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def calculate_biserial_correlation(data, y_column):
    correlations = biserial_correlation(data, y_column)
    correlation_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'BS_Correlation'])
    correlation_df.fillna(0, inplace=True)
    correlation_df['Rank_Corr'] = correlation_df['BS_Correlation'].rank(ascending=False).astype(int)
    sorted_correlation_df = correlation_df.sort_values(by=['BS_Correlation'], ascending=False)    
    return sorted_correlation_df

