import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import pickle
import user_defined_functions as udf
from tempfile import NamedTemporaryFile
import zipfile
import io

def old_data_preparation():
    # st.write("<h1 style='color: #0D3512; font-size: 35px; text-align:left; font-weight: normal;'> Old Data Preparation </h1>", unsafe_allow_html=True)
    st.markdown("""
<div style="background-color: #f9f9f9; padding: 12px; border-radius: 8px; box-shadow: 0 3px 8px rgba(0, 0, 0, 0.1);">
    <h1 style="color: #0D3512; font-size: 28px; text-align: left; font-weight: normal;">Old Data Preparation</h1>
    <p style="font-size: 14px; color: #333333; font-family: Arial, sans-serif; line-height: 1.4;">
        Here, the user should upload the <strong style="color: #0D3512;">old data</strong> and the <strong style="color: #0D3512;">features</strong> used in the model. <br>
        The old data will be filtered based on the selected features.
    </p>
</div>
""", unsafe_allow_html=True)


    
    if 'old_file' not in st.session_state:
        st.session_state.old_file = None 
    # if 'old_df' not  in st.session_state:
    #     st.session_state.old_df = None
    if 'old_df_after_col_drop' not in st.session_state:
        st.session_state.old_df_after_col_drop = None   

    col1, col2 = st.columns(2)
    with col1:    
        old_file = st.file_uploader("Upload compressed CSV (.zip)", type=["zip"])
    with col2:
        old_required_features = st.file_uploader("Upload features for the old model (Excel)", type=["xlsx"])
        if old_required_features is not None:
            st.session_state.old_required_features = old_required_features

    

    st.divider()

    if old_file is not None and st.session_state.new_df_after_col_drop is not None and st.session_state.new_codes_df is not None:
        with zipfile.ZipFile(old_file, "r") as z:
            csv_filename = z.namelist()[0]
            with z.open(csv_filename) as f:
                old_df = pd.read_csv(f)
        # old_df = old_df.drop(0) # for dropping first rows as it contains question numbers ###commenting it, DC, SINCE FIRST ROW DOESNOT HAVE DESCRIPTION
        # st.session_state.old_df = old_df
        old_df_col_names = old_df.columns ##added this DC
        st.session_state.old_df_col_names = old_df_col_names

        old_df_old_col = old_df.copy() ####new_df_new_col is the actual data that we copied from session
        old_df_old_col.columns = st.session_state.old_df_col_names
        old_required_features_df = pd.read_excel(st.session_state.old_required_features)
        old_required_features_df['Required_features'] = old_required_features_df['Feature'] ##added this, DC
        old_df_after_col_drop = old_df_old_col[old_required_features_df['Required_features'].tolist()] ## Required_features.xlsx, will concat the rows of each column, convert to a list and filtering only the reqd features in the actual data which is in new_df_new_col
        st.session_state.old_df_after_col_drop = old_df_after_col_drop
        # st.write(old_df_after_col_drop)   
        if st.session_state.old_df_after_col_drop is not None:
            st.write("<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> Old data is uploaded to the tool </h1>", unsafe_allow_html=True)    
            st.dataframe(st.session_state.old_df_after_col_drop.head(15),use_container_width=True,height=400,hide_index=True)  
            old_df_after_col_drop = st.session_state.old_df_after_col_drop 
            old_csv_data = old_df_after_col_drop.to_csv(index=False)  # Convert dataframe to CSV
            old_csv_buffer = io.StringIO(old_csv_data)  # Create an in-memory buffer

            st.download_button(
                label="Download Selected Columns as CSV",
                data=old_csv_buffer.getvalue(),
                file_name="selected_columns.csv",
                mime="text/csv"
            )     






def evaluate_performance(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true,y_pred , average='weighted')
    recall = recall_score(y_true,y_pred , average='weighted')

    return tn, fp, fn, tp ,accuracy ,precision ,recall

def model_comparison():


    st.write("<h1 style='color: #0D3512; font-size: 35px; text-align:left; font-weight: normal;'> Old Model Comparison </h1>", unsafe_allow_html=True)
    # st.write("""             
    # The features include:
    # - Column Matching : Provides functionality of manually matching columns of old dataset with columns in new dataset if both has different variable name but represent the same feature .  
    # - Comparison: Provides visibility over common questions available in both datasets, newly added questions , and excluded questions from old data.
    # - Filter : Filter questions based on response rate.
    # - Detailed Data Comparison: Compare old and new data across various attributes like valid count, unique values, response rate, and question availability flag.
    # - Download Option: Download the final comparison summary in excel
    # """)
    st.markdown("""
<div style="background-color: #f9f9f9; padding: 12px; border-radius: 8px; box-shadow: 0 3px 8px rgba(0, 0, 0, 0.1);">
    <p style="font-size: 14px; color: #333333; font-family: Arial, sans-serif; line-height: 1.5;">
        <strong style="color: #0D3512;">The features include:</strong><br>
        <strong style="color: #0D3512;">- Column Matching:</strong> Provides functionality for manually matching columns of the old dataset with columns in the new dataset if both have different variable names but represent the same feature.<br>
        <strong style="color: #0D3512;">- Comparison:</strong> Provides visibility over common questions available in both datasets, newly added questions, and excluded questions from the old data.<br>
        <strong style="color: #0D3512;">- Filter:</strong> Filter questions based on response rate.<br>
        <strong style="color: #0D3512;">- Detailed Data Comparison:</strong> Compare old and new data across various attributes like valid count, unique values, response rate, and question availability flag.<br>
        <strong style="color: #0D3512;">- Download Option:</strong> Download the final comparison summary in Excel.
    </p>
</div>
""", unsafe_allow_html=True)


         ###################################
    # st.write("<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> Upload old data files </h1>", unsafe_allow_html=True)

    # # # Check whether base files are already uploaded or not
    # if 'old_file' not in st.session_state:
    #     st.session_state.old_file = None

    

    # # old_file = st.file_uploader("Upload old data (csv)", type=["csv"])  
    # old_file = st.file_uploader("Upload old data (csv)", type=["zip"])
    # if 'old_df' not  in st.session_state:
    #     st.session_state.old_df = None

    

    st.divider()

    if st.session_state.old_df_after_col_drop is not None and st.session_state.new_df_after_col_drop is not None and st.session_state.new_codes_df is not None:
        import zipfile
        # import pandas as pd
        # import streamlit as st

        
       
        # with zipfile.ZipFile(old_file, "r") as z:
        #     csv_filename = z.namelist()[0]
        #     with z.open(csv_filename) as f:
        #         old_df = pd.read_csv(f)
        # # old_df = old_df.drop(0) # for dropping first rows as it contains question numbers ###commenting it, DC, SINCE FIRST ROW DOESNOT HAVE DESCRIPTION
        # st.session_state.old_df = old_df
        # st.write("<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> Old data is uploaded to the tool </h1>", unsafe_allow_html=True) 
        
        
        # display_df = st.session_state.old_df_after_col_drop.copy()
        # limited_cols = display_df.columns[:50]  # Display only first 50
        # subset_df = display_df[limited_cols].copy()
        # for col in subset_df.columns:
        #     subset_df[col] = subset_df[col].astype(str)
        # st.dataframe(subset_df)
        
        
        new_df = st.session_state.new_df_after_col_drop
        old_df = st.session_state.old_df_after_col_drop

        if old_df is not None and new_df is not None:
            # Find common columns while preserving order from new_df
            common_columns = [col for col in new_df.columns if col in old_df.columns]

            # Find uncommon columns
            uncommon_in_old_df = list(set(old_df.columns) - set(new_df.columns))
            uncommon_in_new_df = list(set(new_df.columns) - set(old_df.columns))

            # Convert lists into dataframes for better display with Streamlit   
            common_columns_df = pd.DataFrame(common_columns, columns=["Column Name"])
            uncommon_columns_old_df = pd.DataFrame(uncommon_in_old_df, columns=["Column Name"])
            uncommon_columns_new_df = pd.DataFrame(uncommon_in_new_df, columns=["Column Name"]) 





            merged_df_new= pd.concat([common_columns_df.rename(columns={'Column Name': 'Common columns'}), uncommon_columns_old_df.rename(columns={'Column Name': 'Unique_columns_old_data'}), uncommon_columns_new_df.rename(columns={'Column Name': 'Unique_columns_new_data'})], axis=1)

            # Display tables
            # if common_columns_df.empty:
            #     st.write("<h1 style='color: #FF0000; font-size: 20px; text-align:center; font-weight: normal;'> **** No common columns are available in both data **** </h1>", unsafe_allow_html=True)
            # else:

            st.write("<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> Common Columns and unique columns in both data</h1>", unsafe_allow_html=True)    
            st.dataframe(merged_df_new,use_container_width=True,height=400,hide_index=True)
            
            st.write("<h1 style='color: #0D3512; font-size: 20px; text-align:left; font-weight: normal;'> If want to modify then download template and match the columns from old and new data , then upload in excel format </h1>", unsafe_allow_html=True)
            with st.expander("Download the template"):
                    template_df = pd.read_excel("old_new_variable_matching.xlsx")
                    st.dataframe(template_df)
                    st.download_button("Download Template",template_df.to_csv(index = False),'variable_matching_template.csv')
            variable_matching = st.file_uploader("Upload modified variable matching file (Excel)", type=["xlsx"])
            if 'variable_matching_df' not in st.session_state:
                st.session_state.variable_matching_df = None
            if variable_matching is not None:
                variable_matching_df = pd.read_excel(variable_matching)
                st.session_state.variable_matching_df = variable_matching_df

            if st.session_state.variable_matching_df is not None :  
                variable_matching_df = st.session_state.variable_matching_df
                st.write("Variable matching file uploaded")
                replacement_dict = dict(zip(variable_matching_df['old_model_variable'], variable_matching_df['matched_new_model_variable']))
                old_df.columns = [replacement_dict.get(item, item) for item in old_df.columns]
            else:
                st.write("Please download the  variable matching template file and upload if required")
            # if st.button('Continue'):

            if old_df is not None and new_df is not None :

            # Find common columns while preserving order from new_df
                common_columns = [col for col in new_df.columns if col in old_df.columns]

                # Find uncommon columns
                uncommon_in_old_df = list(set(old_df.columns) - set(new_df.columns))
                uncommon_in_new_df = list(set(new_df.columns) - set(old_df.columns))

                # Convert lists into dataframes for better display with Streamlit   
                common_columns_df = pd.DataFrame(common_columns, columns=["Column Name"])
                uncommon_columns_old_df = pd.DataFrame(uncommon_in_old_df, columns=["Column Name"])
                uncommon_columns_new_df = pd.DataFrame(uncommon_in_new_df, columns=["Column Name"]) 

                
            if uncommon_columns_old_df.empty:
                st.write("<h1 style='color: #FF0000; font-size: 20px; text-align:center; font-weight: normal;'> **** No columns are removed from Old Data **** </h1>", unsafe_allow_html=True)
            else:
                st.write("### Removed Columns from Old Data")
                st.dataframe(uncommon_columns_old_df,use_container_width=True,height=400,hide_index=True)

            if uncommon_columns_new_df.empty:
                st.write("<h1 style='color: #FF0000; font-size: 20px; text-align:center; font-weight: normal;'> **** No new columns are added to New Data **** </h1>", unsafe_allow_html=True)
            else:
                st.write("### New Columns in New Data")
                st.dataframe(uncommon_columns_new_df,use_container_width=True,height=400,hide_index=True)

            if old_df is not None and new_df is not None:
                # Combine the column names of both datasets into a unique list
                all_columns = list(set(old_df.columns).union(set(new_df.columns)))
            
                # Create an empty list to store rows for the comparison table
                comparison_data = []
            
                # Compare columns and create flags
                for col in all_columns:
                    # Check if the column is in old_df and new_df
                    in_old_df = 'Yes' if col in old_df.columns else 'No'
                    in_new_df = 'Yes' if col in new_df.columns else 'No'
            
                    # Count valid (non-null) values in the column for both datasets
                    count_old_df = old_df[col].notna().sum() if col in old_df.columns else 0
                    resp_rate_old_df = count_old_df/len(old_df)*100 if len(old_df) > 0 else 0
                    count_new_df = new_df[col].notna().sum() if col in new_df.columns else 0
                    resp_rate_new_df = count_new_df/len(new_df)*100 if len(new_df) > 0 else 0

                    # Count unique values in the column for both datasets
                    unique_old_df = old_df[col].nunique() if col in old_df.columns else 0
                    unique_new_df = new_df[col].nunique() if col in new_df.columns else 0
                    
                    # Append the comparison result
                    comparison_data.append([col, in_old_df, count_old_df, resp_rate_old_df, unique_old_df, in_new_df, count_new_df, resp_rate_new_df, unique_new_df])
            
                # Create a DataFrame from the comparison data
                comparison_df = pd.DataFrame(comparison_data, columns=["Column Name", "In Dataset Old", "Valid Count in Dataset Old", "Response Rate Old (%)", "Unique Values in Dataset Old", "In Dataset New", "Valid Count in Dataset New", "Response Rate New (%)", "Unique Values in Dataset New"])
            
                # Ensuring Guest ID will always remain on the top
                guest_id_row = comparison_df[comparison_df["Column Name"] == "responseid"]

                comparison_df = comparison_df[comparison_df["Column Name"] != "responseid"]

                # Sort the dataframe by "Valid Count in Dataset 1" in descending order
                comparison_df = comparison_df.sort_values(by="Valid Count in Dataset Old", ascending=False)
            
                comparison_df = pd.concat([guest_id_row,comparison_df], ignore_index = True)
                
                st.write("<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> Sliders to filter and review columns based on response rate</h1>", unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    min_value_old = st.slider ('Select Minimum Response Rate of Old Data:' , min_value=0,max_value=100)
                with col2:
                    min_value_new = st.slider ('Select Minimum Response Rate of New Data:' , min_value=0,max_value=100)

                comparison_df = comparison_df[(comparison_df['Response Rate Old (%)']>=min_value_old)&(comparison_df['Response Rate New (%)']>=min_value_new)]
                st.write("<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> Detailed comparison of both datasets  - sorted based on valid count in old data and keeping 'resonseid' on the top </h1>", unsafe_allow_html=True)
                st.dataframe(comparison_df,use_container_width=True,height=400,hide_index=True)
        else:
            # st.write("#### **** Upload Old and New data files ****")
            st.write('<span style="color:red; font-size:25px; font-weight:normal;"> **** Please upload Old data  **** </span >', unsafe_allow_html=True)

        










            









































def old_model_measurement():    
    st.write("<h1 style='color: #0D3512; font-size: 35px; text-align:left; font-weight: normal;'> Measurement of Old Model  </h1>", unsafe_allow_html=True)
    # st.write("""
    # Here, users will get the visibility of old model performance over new 
    
    # """)
#     st.write("""
#     Here, users will get the visibility of old model performance over new.  
#     If there is no existing model to evaluate, you may skip this step and proceed directly to the ML Engine.
# """)
    st.markdown("""
    <div style='background-color: #f0f9f0; padding: 15px; border-radius: 10px; border-left: 5px solid #0D3512;'>
        <p style='font-size: 16px; color: #0D3512;'>
            <strong>Note:</strong> Here, users will get visibility into the <strong>old model's performance</strong> over the new one.<br>
            If there is <em>no existing model</em> to evaluate, you can skip this step and proceed directly to the <strong>ML Engine</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)


    st.divider()
    
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
        # st.write('tester 1----------------------------------------------------------------------------------------------------------------')

    if 'model_old' or 'dependent_variable_old' or 'independent_variable_names_old' or 'performance_metrics_old' or 'log_intercept_df_old' not in st.session_state:
        st.session_state.model_old = None
        st.session_state.dependent_variable_old = None
        st.session_state.independent_variable_names_old = None
        st.session_state.performance_metrics_old = None
        st.session_state.log_intercept_df_old = None
    
    # Initialize session state if not already done
    if 'uploaded_file_path' not in st.session_state:
        st.session_state.uploaded_file_path = None
        # st.write('tester 2----------------------------------------------------------------------------------------------------------------')

    st.write("<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> Load your earlier model in pickled format </h1>", unsafe_allow_html=True)
    # st.write("<h1 style='color: #0D3512; font-size: 20px; text-align:left; font-weight: normal;'> If want to modify then download template and match the columns from old and new data , then upload in excel format </h1>", unsafe_allow_html=True)
    # with st.expander("Understand the template for pickle file "):
    #     st.write(""" 
        
    #     Here users should create a pickle file out of old model outputs .
        
    #     - It should have 4 items packed .
    #       - Model
    #         - The model trained on old data 
    #       - Dependant variable
    #       - Independant variable
    #       - Performance metrics 
    #         - 
    #       - Log intercept 
    #      """)
    #         st.download_button("Download Template",template_df.to_csv(index = False),'variable_matching_template.csv')
    uploaded_pkl = st.file_uploader("Choose a pickle file", type="pkl")
    ##########################################################################################
    
    if uploaded_pkl is not None:
        # Save the uploaded file temporarily on disk
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_pkl.getvalue())
            temp_filepath = temp_file.name
            st.write(temp_filepath)
    
    
        # Store path in session state for persistence across reruns
        st.session_state.uploaded_file_path = temp_filepath

    # Load and display content from saved path if available
    if st.session_state.uploaded_file_path is not None:
        with open(st.session_state.uploaded_file_path, "rb") as f:
            st.session_state.model_old, st.session_state.dependent_variable_old, st.session_state.independent_variable_names_old, st.session_state.performance_metrics_old, st.session_state.log_intercept_df_old = pickle.load(f)
            st.session_state.log_intercept_df_old.rename(columns={'Feature Name': 'Feature Name Training','Log Coefficients': 'Log Coefficients Training'}, inplace=True)


#################################################################################################################################
    # st.write("<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> Fill the required details to check old model performance over new data </h1>", unsafe_allow_html=True)
    if "data" in st.session_state and st.session_state.model_old is not None and st.session_state.dependent_variable_old is not None and st.session_state.independent_variable_names_old is not None and st.session_state.performance_metrics_old is not None and st.session_state.log_intercept_df_old is not None :

        model_data = st.session_state.data.copy()
        # q_no, q_desc = udf.split_list(model_data.columns)
        q_desc = model_data.columns
        model_data_col_reset = st.session_state.data.copy()
        # model_data_col_reset.columns = q_desc
        if st.session_state.variable_matching_df is not None :  
            variable_matching_df = st.session_state.variable_matching_df
            replacement_dict = dict(zip(variable_matching_df['matched_new_model_variable'], variable_matching_df['old_model_variable']))
            model_data_col_reset.columns = [replacement_dict.get(item, item) for item in q_desc]
        else :
            model_data_col_reset.columns = q_desc
        
        # old_variables = st.session_state.independent_variable_names_old
        # st.write(11111,list(old_variables))

        
        # st.write(model_data_col_reset)
        # st.write('tester ----------------------------------------------------------------------------------------------------------------')
        if  len(set([st.session_state.dependent_variable_old])-set(list(model_data_col_reset.columns))) == 0:
            st.write("<h1 style='color: blue; font-size: 25px; text-align:left; font-weight: normal;'> Dependent variable is present in new data </h1>", unsafe_allow_html=True)                    
            # st.write('dependent variable is present')
        else:
            st.write("<h1 style='color: #FF0000; font-size: 20px; text-align:center; font-weight: normal;'> Dependent variable is missing in new data </h1>", unsafe_allow_html=True)
            # st.write('Dependent variable is missing in new data')

        if len(set(list(st.session_state.independent_variable_names_old))-set(list(model_data_col_reset.columns))) == 0:
            st.write("<h1 style='color: blue; font-size: 25px; text-align:left; font-weight: normal;'> All Independent variables are present in new data </h1>", unsafe_allow_html=True)    
            # st.write('all independent variables are present')
        else:
            # st.write('below features are missing')
            st.write("<h1 style='color: #FF0000; font-size: 20px; text-align:center; font-weight: normal;'> Below Independent variables are missing in new data </h1>", unsafe_allow_html=True)
            st.write(pd.DataFrame(set(list(st.session_state.independent_variable_names_old))-set(list(model_data_col_reset.columns)),columns=['Missing Features']))

        # st.write("test--------------------------------")
        st.divider()
        model_data_col_reset = model_data_col_reset.loc[:,[st.session_state.dependent_variable_old]+list(st.session_state.independent_variable_names_old)]
        # st.write(data)

        model_data_col_reset.dropna(inplace=True)
        # st.write(data)

        y_test_new = model_data_col_reset.loc[:,[st.session_state.dependent_variable_old]]
        # st.write(y_test_new)
        X_test_new = model_data_col_reset.loc[:,list(st.session_state.independent_variable_names_old)]
        y_test_pred_new = st.session_state.model_old.predict(X_test_new)
        y_test_pred_prob = st.session_state.model_old.predict_proba(X_test_new)
        # st.write(y_test_pred_prob)
        # st.write("2--------------------------------")
        test_tn, test_fp, test_fn, test_tp, test_accuracy,test_precision,test_recall = evaluate_performance(y_test_new,y_test_pred_new)
        # st.write("3--------------------------------")
        list_prf_mtx =['True Negative','False Positive','False Negative','True Positive','Accuracy','Precision','Recall']
        performance_metrics = pd.DataFrame({'Performance Metrics' : list_prf_mtx, 'New Data':[test_tn, test_fp, test_fn, test_tp, test_accuracy,test_precision,test_recall]},)
        # st.write("<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> Performance metrics based on new data by leveraging old model </h1>", unsafe_allow_html=True)
        performance_metrics_inc_old = pd.concat([st.session_state.performance_metrics_old,performance_metrics],axis=1).loc[:,['Performance Metrics','Training','Testing','New Data']]
        performance_metrics_inc_old.columns = ['Performance Metrics', 'Old Model - Training', 'Old Model - Testing','Old Model - New Data']
        st.session_state.performance_metrics_inc_old = performance_metrics_inc_old
        # st.dataframe(performance_metrics_inc_old, use_container_width=True, hide_index=True)

        st.write("<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> Accuracy metrics based on new data by leveraging old model </h1>", unsafe_allow_html=True)
        # for plotting 
        accuracy_data = performance_metrics_inc_old[performance_metrics_inc_old['Performance Metrics'] == "Accuracy"]
        
        # Plotting the bar chart using Matplotlib
        fig, ax = plt.subplots()
        colors = ['lightgreen', 'green', 'Blue']
        bars = ax.bar(accuracy_data.columns[1:], accuracy_data.iloc[0][1:], color=colors)
        # ax.bar_label(bars)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2%}', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), # Offset label position above the top of the bar
                        textcoords="offset points",
                        ha='center', va='bottom')
        ax.set_title('Accuracy Comparison')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Model Results')
        ax.set_ylim([0,1])

        # Displaying the plot in Streamlit app
        # st.pyplot(fig)
        
        # create 2 columns for layout
        col1, col2 = st.columns(2)
        
        # Upload CSV files
        with col1:
            st.pyplot(fig)

        with col2:
            st.write("")
            st.write("")
            st.write("")
            accuracy_gap = abs(round((performance_metrics_inc_old.iloc[4,3]-performance_metrics_inc_old.iloc[4,2])*100,2))
            if accuracy_gap <= 10 :
                st.write(f"<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> Absolute accuracy drops less than 10% (ie {accuracy_gap}%) so model retaining is not needed </h1>", unsafe_allow_html=True)
            elif accuracy_gap > 10 :
                st.write(f"<h1 style='color: #FF0000; font-size: 25px; text-align:left; font-weight: normal;'> Absolute accuracy drops more than 10% (ie {accuracy_gap}%) so model retaining is needed </h1>", unsafe_allow_html=True)


        # st.write("4--------------------------------")
        # fitting logistic regression on new data by considering same columns to get coefficients
        model_new = LogisticRegression()
        model_new.fit(X_test_new, y_test_new)
        coefficients_log=np.log(np.abs(model_new.coef_[0]))
        coefficients_log_df=pd.DataFrame(coefficients_log,index=X_test_new.columns,columns=['Log Coefficients (New Data)'])
        log_intercept_df=pd.DataFrame({"Log Coefficients (New Data)":[np.log(np.abs(model_new.intercept_[0]))]},index=['Intercept'])
        log_intercept_df = pd.concat([log_intercept_df,coefficients_log_df])
        log_intercept_df = log_intercept_df.reset_index().rename(columns={'index': 'Feature Name'})
        # st.write("<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> Log Odds and intercept based on new data by leveraging old model features </h1>", unsafe_allow_html=True)
        log_intercept_df_inc_old = pd.concat([st.session_state.log_intercept_df_old,log_intercept_df],axis=1).loc[:,['Feature Name','Log Coefficients Training','Log Coefficients (New Data)']]
        # st.dataframe(log_intercept_df_inc_old, use_container_width=True, hide_index=True)
        # st.write("5--------------------------------")

        df_normalized=log_intercept_df_inc_old.iloc[1:,]
        # Normalize columns to sum up to 100%
        df_normalized['Log Coefficients Training'] = (df_normalized['Log Coefficients Training'] / df_normalized['Log Coefficients Training'].sum()) * 100
        df_normalized['Log Coefficients (New Data)'] = (df_normalized['Log Coefficients (New Data)'] / df_normalized['Log Coefficients (New Data)'].sum()) * 100
        df_normalized['Delta'] = (df_normalized['Log Coefficients Training'] - df_normalized['Log Coefficients (New Data)'])
        st.write("<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> Log Odds coefficients allocated to 100% </h1>", unsafe_allow_html=True)
        st.dataframe(df_normalized, use_container_width=True, hide_index=True)


    else:
        st.write('<span style="color:red; font-size:25px; font-weight:normal;"> ****  **** </span >', unsafe_allow_html=True)


def run():
    st.divider()
    
    
    # if st.session_state.new_file is not None and st.session_state.new_codes is not None:
    sub_page = st.sidebar.radio(
        "Choose a sub task:",
        ["Old Data Preparation", "Data comparison","Old model mesaurement"]  # Added "Binning/Bucketing" to the list
    )
    if sub_page == "Old Data Preparation":
        old_data_preparation()


    if sub_page == "Data comparison":
        model_comparison()
    elif sub_page == "Old model mesaurement":
        old_model_measurement()

