import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import textwrap
import user_defined_functions as udf
from scipy.stats import pointbiserialr
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RidgeClassifier
import numpy as np
# from glmnet_py import LogitNet
# from glmnet_python import Lo

import io

# def run_model_and_download(x_columns, y_column, data, top_label="Overall"):
#     ## Function called inside Overall new Model, 
#     ## This Function is called after selection of Stage1 features as well as if Stage2 Features are selected in Overall New Model.
#     st.session_state.x_columns = x_columns
#     if y_column and x_columns:
#         # Prepare training and testing datasets based on user selection
#         X = data[x_columns]
#         y = data[y_column]
#     counts = X.apply(pd.Series.value_counts).fillna(0).astype(int)

#     features = []
#     calculations = []

#     # Iterate over each column in df_a
#     for col in counts.columns:
#         features.append(col)
#         # Calculate second row / first row for each column
#         calculation = counts[col].iloc[1] / counts[col].iloc[0] if counts[col].iloc[0] != 0 else np.nan
#         calculations.append(calculation)
#     # Create dataframe b with two columns: feature and calculation
#     EXperience_ratio_df = pd.DataFrame({
#         'Feature': features,
#         'Experience_perc': calculations
#     })
    
    
#     correlation_df = udf.calculate_biserial_correlation(data, y_column)
#     # Task - Fit Logistic Regression with Crossvalidation and create dataframe of feature names coefficients and rank column.
#     log_reg_cv=LogisticRegressionCV(cv=5,max_iter=10000).fit(X,y)
#     coefficients = log_reg_cv.coef_
#     if coefficients.shape[0] == 1:
#         # Binary case
#         coef_series = pd.Series(coefficients[0], index=X.columns)
#         logistic_regression_df = pd.DataFrame({'Feature': coef_series.index, 'Coefficient_Logistic': coef_series.values})
#     else:
#         # Multiclass — average coefficients across classes
#         avg_coefficients = coefficients.mean(axis=0)
#         coef_series = pd.Series(avg_coefficients, index=X.columns)
#         logistic_regression_df = pd.DataFrame({'Feature': coef_series.index, 'Coefficient_Logistic': coef_series.values})

#     coefficients = coef_series
#     logistic_regression_df=pd.DataFrame({'Feature':coefficients.index,'Coefficient_Logistic':coefficients.values})
#     logistic_regression_df['Rank_logistic']=logistic_regression_df['Coefficient_Logistic'].rank(ascending=False).astype(int)
#     logistic_regression_df= logistic_regression_df.sort_values(by = ['Coefficient_Logistic'], ascending=False)
    
    
#     # Task - Fit Ridge Regression for different values of alpha.
#     alphas=[ 0.0001, 0.001, 0.01, 0.1, 1 ,10, 20, 50, 100, 1000]
#     prefixed_alphas = [f'alpha_{alpha}' for alpha in alphas]
#     # ridge_cv=RidgeCV(alphas=alphas).fit(X,y)
#     ridge_dfs = {}
#     counter = 1
    

#     for alpha in alphas:
#         ridge_classifier = Ridge(alpha=alpha)
        
#         # Perform cross-validation and calculate mean score across folds
#         cv_scores = cross_val_score(ridge_classifier, X, y, cv=5) # Using 5-fold CV
        
#         # Fit the model on entire dataset after evaluating with CV scores 
#         ridge_classifier.fit(X,y)
#         ridge_coeffs=pd.Series(ridge_classifier.coef_,index=X.columns)
#         ridge_regression_df=pd.DataFrame({'Feature':ridge_coeffs.index,f'Coefficient_{counter}':ridge_coeffs.values})
#         ridge_regression_df[f'Rank_df{counter}']=ridge_regression_df[f'Coefficient_{counter}'].rank(ascending=False).astype(int)
#         ridge_regression_df['CV_Score']=[cv_scores.mean()]*len(ridge_coeffs) # Adding CV Score column
#         ridge_dfs[f'ridge_regression_df_{counter}'] = ridge_regression_df.sort_values(by=[f'Coefficient_{counter}'], ascending=False)
#         # st.write(111,ridge_regression_df,111)
#         counter += 1
    
#     ridge_regression_df_1 =  ridge_dfs['ridge_regression_df_1'].loc[:,['Feature','Rank_df1','Coefficient_1']]   
#     ridge_regression_df_2 =  ridge_dfs['ridge_regression_df_2'].loc[:,['Feature','Rank_df2','Coefficient_2']]
#     ridge_regression_df_3 =  ridge_dfs['ridge_regression_df_3'].loc[:,['Feature','Rank_df3','Coefficient_3']]
#     ridge_regression_df_4 =  ridge_dfs['ridge_regression_df_4'].loc[:,['Feature','Rank_df4','Coefficient_4']]
#     ridge_regression_df_5 =  ridge_dfs['ridge_regression_df_5'].loc[:,['Feature','Rank_df5','Coefficient_5']]
#     ridge_regression_df_6 =  ridge_dfs['ridge_regression_df_6'].loc[:,['Feature','Rank_df6','Coefficient_6']]   
#     ridge_regression_df_7 =  ridge_dfs['ridge_regression_df_7'].loc[:,['Feature','Rank_df7','Coefficient_7']]
#     ridge_regression_df_8 =  ridge_dfs['ridge_regression_df_8'].loc[:,['Feature','Rank_df8','Coefficient_8']]
#     ridge_regression_df_9 =  ridge_dfs['ridge_regression_df_9'].loc[:,['Feature','Rank_df9','Coefficient_9']]
#     ridge_regression_df_10 =  ridge_dfs['ridge_regression_df_10'].loc[:,['Feature','Rank_df10','Coefficient_10']]


#     merged_df= ridge_regression_df_1.merge(ridge_regression_df_2, on=['Feature'], how='outer')\
#         .merge(ridge_regression_df_3, on=['Feature'], how='outer')\
#             .merge(ridge_regression_df_4, on=['Feature'], how='outer')\
#                 .merge(ridge_regression_df_5, on=['Feature'], how='outer')\
#                     .merge(ridge_regression_df_6, on=['Feature'], how='outer')\
#                         .merge(ridge_regression_df_7, on=['Feature'], how='outer')\
#                             .merge(ridge_regression_df_8 ,on=['Feature'], how='outer')\
#                                 .merge(ridge_regression_df_9 ,on=['Feature'], how='outer')\
#                                     .merge(ridge_regression_df_10,on=['Feature'], how='outer')\
#                                         .merge(correlation_df,on=['Feature'], how='outer')\
#                                             .merge(logistic_regression_df,on=['Feature'], how='outer')
    
#     merged_df['Rank_Corr_df1'] = abs(merged_df['Rank_Corr'] - merged_df['Rank_df1'])
#     merged_df['Rank_Corr_df2'] = abs(merged_df['Rank_Corr'] - merged_df['Rank_df2'])
#     merged_df['Rank_Corr_df3'] = abs(merged_df['Rank_Corr'] - merged_df['Rank_df3'])
#     merged_df['Rank_Corr_df4'] = abs(merged_df['Rank_Corr'] - merged_df['Rank_df4'])
#     merged_df['Rank_Corr_df5'] = abs(merged_df['Rank_Corr'] - merged_df['Rank_df5'])
#     merged_df['Rank_Corr_df6'] = abs(merged_df['Rank_Corr'] - merged_df['Rank_df6'])
#     merged_df['Rank_Corr_df7'] = abs(merged_df['Rank_Corr'] - merged_df['Rank_df7'])
#     merged_df['Rank_Corr_df8'] = abs(merged_df['Rank_Corr'] - merged_df['Rank_df8'])
#     merged_df['Rank_Corr_df9'] = abs(merged_df['Rank_Corr'] - merged_df['Rank_df9'])
#     merged_df['Rank_Corr_df10'] = abs(merged_df['Rank_Corr'] - merged_df['Rank_df10'])

#     merged_df['Coefficient_Logistic_df1'] = abs(merged_df['Coefficient_Logistic'] - merged_df['Coefficient_1'])
#     merged_df['Coefficient_Logistic_df2'] = abs(merged_df['Coefficient_Logistic'] - merged_df['Coefficient_2'])
#     merged_df['Coefficient_Logistic_df3'] = abs(merged_df['Coefficient_Logistic'] - merged_df['Coefficient_3'])
#     merged_df['Coefficient_Logistic_df4'] = abs(merged_df['Coefficient_Logistic'] - merged_df['Coefficient_4'])
#     merged_df['Coefficient_Logistic_df5'] = abs(merged_df['Coefficient_Logistic'] - merged_df['Coefficient_5'])
#     merged_df['Coefficient_Logistic_df6'] = abs(merged_df['Coefficient_Logistic'] - merged_df['Coefficient_6'])
#     merged_df['Coefficient_Logistic_df7'] = abs(merged_df['Coefficient_Logistic'] - merged_df['Coefficient_7'])
#     merged_df['Coefficient_Logistic_df8'] = abs(merged_df['Coefficient_Logistic'] - merged_df['Coefficient_8'])
#     merged_df['Coefficient_Logistic_df9'] = abs(merged_df['Coefficient_Logistic'] - merged_df['Coefficient_9'])
#     merged_df['Coefficient_Logistic_df10'] = abs(merged_df['Coefficient_Logistic'] - merged_df['Coefficient_10'])

#     diff_sum = merged_df.sum()
#     diff_sum = diff_sum.reset_index().rename(columns={'index': 'Column Names', 0: 'Sum'})
#     delta_rank = diff_sum[diff_sum['Column Names'].isin (['Rank_Corr_df1', 'Rank_Corr_df2', 'Rank_Corr_df3', 'Rank_Corr_df4', 
#                                                         'Rank_Corr_df5','Rank_Corr_df6', 'Rank_Corr_df7', 'Rank_Corr_df8', 
#                                                         'Rank_Corr_df9', 'Rank_Corr_df10'])] #.sort_values(by='Sum', ascending=True)
#     delta_rank = delta_rank.reset_index().iloc[:,2]
#     delta_Coeff = diff_sum[diff_sum['Column Names'].isin (['Coefficient_Logistic_df1', 'Coefficient_Logistic_df2', 'Coefficient_Logistic_df3', 'Coefficient_Logistic_df4', 
#                                                         'Coefficient_Logistic_df5','Coefficient_Logistic_df6', 'Coefficient_Logistic_df7', 'Coefficient_Logistic_df8', 
#                                                         'Coefficient_Logistic_df9', 'Coefficient_Logistic_df10'])] #.sort_values(by='Sum', ascending=True)
#     delta_Coeff = delta_Coeff.reset_index().iloc[:,2]
#     decision_data = {
#         'prefixed_alphas': prefixed_alphas,
#         'delta_rank': delta_rank,
#         'delta_Coeff': delta_Coeff,
#         'alphas': alphas
#     }
#     decision_df = pd.DataFrame(decision_data)
#     # Sort the DataFrame by delta_rank, delta_Coeff, and alphas in ascending order
#     decision_df.sort_values(by=['delta_rank', 'delta_Coeff', 'alphas'], ascending=[True, True, True], inplace=True)
#     final_alpha = decision_df.iloc[0,-1]
#     #######
#     # alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 20, 50, 100, 1000]
#     # ridge_cv_model = RidgeCV(alphas=alphas, cv=5)
#     # ridge_cv_model.fit(X, y)
#     # final_alpha = ridge_cv_model.alpha_
# #########

#     test_size_selector = st.slider ('Test size selector (default 30):' , min_value=0,max_value=100,value = 30,key=f"test_size_selector_{top_label}")
#     st.write(f"<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> {top_label} selected model : </h1>", unsafe_allow_html=True)
#     st.session_state.test_size_selector = test_size_selector


#     # Split data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_selector, random_state=42)
#     ridge_classifier_f = Ridge(alpha=final_alpha)
#     # Perform cross-validation and calculate mean score across folds
#     cv_scores = cross_val_score(ridge_classifier_f, X_train, y_train, cv=5) # Using 5-fold CV
#     # Fit the model on entire dataset after evaluating with CV scores 
#     ridge_classifier_f.fit(X_train, y_train)
#     ridge_coeffs_f=pd.Series(ridge_classifier_f.coef_,index=X_train.columns)
#     ridge_regression_df_f = pd.DataFrame({
#     'Feature': ridge_coeffs_f.index,
#     'Model Value': ridge_coeffs_f.values
# })
#     # Compute Odds-Ratio: exp(coef) if coef >= 0 else 1/exp(abs(coef))
#     ridge_regression_df_f['Odds-Ratio'] = ridge_regression_df_f['Model Value'].apply(
#         lambda coef: np.exp(coef) if coef >= 0 else 1 / np.exp(abs(coef))
#     )
#     # Sort by absolute model value (optional, or use Odds-Ratio)
#     ridge_regression_df_f.sort_values(by='Model Value', ascending=False, inplace=True)
#     ridge_regression_df_f = pd.merge(ridge_regression_df_f,st.session_state.sum_summary1_copy,on='Feature',how='left')
#     y_pred_tr=ridge_classifier_f.predict(X_train)
#     y_pred_tr = [1 if val >= 0.6 else 0 for val in [udf.sigmoid(value) for value in y_pred_tr]]
#     accuracy_tr=accuracy_score(y_train,y_pred_tr)
#     y_pred_te=ridge_classifier_f.predict(X_test)
#     y_pred_te = [1 if val >= 0.6 else 0 for val in [udf.sigmoid(value) for value in y_pred_te]]
#     accuracy_te=accuracy_score(y_test,y_pred_te)

#     accuracy_gap = abs(round((accuracy_tr - accuracy_te)*100,2))
    
#     ############## for the download option ##############
#     ridge_regression_df_f_copy = ridge_regression_df_f
#     ridge_regression_df_f_copy['Segment'] = 'Overall'
#     ridge_regression_df_f_copy['Type'] = 'Overall'
#     ridge_regression_df_f_copy['Dependent Variable'] = y_column
#     ridge_regression_df_f_copy = ridge_regression_df_f_copy.rename(columns={
#             'Feature': 'Independent Variables',
#             'Model Value': 'Coefficient'
#         }).reset_index(drop=True)
#     # Fill NA Actual % for Intercept or missing features with empty string or 0
#     ridge_regression_df_f_copy['Actual %'] = ridge_regression_df_f_copy['Actual %'].fillna('')
#     feature_description = st.session_state.required_features_df
#     feature_description = feature_description.rename(columns={'Feature':'Independent Variables'}).reset_index(drop=True)
#     st.session_state.feature_description = feature_description
#     ridge_regression_df_f_copy = pd.merge(ridge_regression_df_f_copy,feature_description, on= 'Independent Variables', how='left')
#     ridge_regression_df_f_copy['Model Accuracy'] = accuracy_gap

#     ################ overall % ###########################
#     # Get the Overall % value for the dependent variable
#     overall_actual_pct_row = st.session_state.sum_summary1_copy[
#         st.session_state.sum_summary1_copy['Feature'] == st.session_state.y_column
#     ]
#     # Extract the Actual % value, use .values[0] safely
#     overall_actual_pct = ''
#     if not overall_actual_pct_row.empty and 'Actual %' in overall_actual_pct_row.columns:
#         overall_actual_pct = overall_actual_pct_row['Actual %'].values[0]
#     ridge_regression_df_f_copy['Overall %'] = overall_actual_pct
#     ################################################################################
#     desired_order = [
#     "Dependent Variable", "Segment", "Type", "Independent Variables", "Coefficient",
#     "Odds-Ratio", "Actual %", "Overall %", "Model Accuracy", "Description"
# ]
#     ridge_regression_df_f_copy = ridge_regression_df_f_copy[desired_order]  
#     ridge_regression_df_f_copy = ridge_regression_df_f_copy.drop_duplicates()
#     st.session_state.ridge_regression_df_f_copy = ridge_regression_df_f_copy
#         ##############################################
#     st.dataframe(ridge_regression_df_f[['Feature', 'Odds-Ratio']], hide_index=True, use_container_width=True) ### this is getting displayed
#     st.session_state.ridge_regression_df_f = ridge_regression_df_f
#     st.session_state.EXperience_ratio_df = EXperience_ratio_df

#     labels = ['Training', 'Testing']
#     values = [accuracy_tr,accuracy_te]

#     # Plotting the bar chart using Matplotlib
#     fig, ax = plt.subplots()
#     colors = ['lightgreen', 'blue']
#     bars = ax.bar(labels, values, color=colors)
#     # Annotate bars with their respective heights in percentage format
#     for bar in bars:
#         height = bar.get_height()
#         ax.annotate(f'{height * 100:.0f}%', 
#                     xy=(bar.get_x() + bar.get_width() / 2, height),
#                     xytext=(0, 3), # Offset label position above the top of the bar
#                     textcoords="offset points",
#                     ha='center', va='bottom')

#     # Set titles and labels
#     ax.set_title('Accuracy Comparison')
#     ax.set_ylabel('Accuracy')
#     ax.set_xlabel('Model Results')
#     ax.set_ylim([0, 1])

#     col1, col2 = st.columns(2)
#     with col1:
#         st.pyplot(fig)
#     with col2:
#         st.write("")
#         st.write("")
#         st.write("")
#         accuracy_gap = abs(round((accuracy_tr - accuracy_te)*100,2))
#         if accuracy_gap <= 10 :
#             st.write(f"<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> Absolute accuracy difference is less than 10% (ie {accuracy_gap}%) so model is not overfitting </h1>", unsafe_allow_html=True)
#         elif accuracy_gap > 10 :
#             st.write(f"<h1 style='color: #FF0000; font-size: 25px; text-align:left; font-weight: normal;'> Absolute accuracy difference is more than 10% (ie {accuracy_gap}%) so model is overfitting </h1>", unsafe_allow_html=True)

#     excel_buffer = io.BytesIO()
   

#     with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
#         ridge_regression_df_f_copy.to_excel(writer, index=False)
#     excel_buffer.seek(0)
#     # return excel_buffer
#     # Create a download button
#     st.download_button(
#     label=f"📥 Download {top_label} Model Results (Excel)",
#     data=excel_buffer,
#     file_name=f"{top_label}_Model_Results.xlsx",
#     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#     key=f"download_button_{top_label}"
# )

def run_model_and_download(x_columns, y_column, data, top_label="Overall"):
    ## Function called inside Overall new Model, 
    ## This Function is called after selection of Stage1 features as well as if Stage2 Features are selected in Overall New Model.
    st.session_state.x_columns = x_columns
    if y_column and x_columns:
        # Prepare training and testing datasets based on user selection
        X = data[x_columns]
        y = data[y_column]
    counts = X.apply(pd.Series.value_counts).fillna(0).astype(int)

    features = []
    calculations = []

    # Iterate over each column in df_a
    for col in counts.columns:
        features.append(col)
        # Calculate second row / first row for each column
        calculation = counts[col].iloc[1] / counts[col].iloc[0] if counts[col].iloc[0] != 0 else np.nan
        calculations.append(calculation)
    # Create dataframe b with two columns: feature and calculation
    EXperience_ratio_df = pd.DataFrame({
        'Feature': features,
        'Experience_perc': calculations
    })
    
    
    correlation_df = udf.calculate_biserial_correlation(data, y_column)
    
    ############################# matching with R #################
    import statsmodels.api as sm
    X_with_const = sm.add_constant(X, has_constant='add')
    glm_model = sm.GLM(y, X_with_const, family=sm.families.Binomial())
    glm_result = glm_model.fit()

    
    glm_coeff_df = glm_result.summary2().tables[1].reset_index()
    glm_coeff_df.rename(columns={'index': 'Feature', 'Coef.': 'Coefficient_Logistic'}, inplace=True)
    glm_coeff_df = glm_coeff_df[['Feature', 'Coefficient_Logistic']]

    # Optional: Drop intercept
    # glm_coeff_df = glm_coeff_df[glm_coeff_df['Feature'] != 'const']
    logistic_regression_df = glm_coeff_df
    st.write(glm_coeff_df)
    ############################# upto here matching with R #################
    logistic_regression_df['Rank_logistic'] = logistic_regression_df['Coefficient_Logistic'].rank(ascending=False).astype(int)
    logistic_regression_df = logistic_regression_df.sort_values(by=['Coefficient_Logistic'], ascending=False)

    
    # Task - Fit Ridge Regression for different values of alpha.
    
    X_train = X
    # st.write(X_train)
    y_train = y
    

    ################### test ######################
    ################## commented portion is to generate lambdas in python ####################
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    # Assuming x_train and y_train are numpy arrays or pandas DataFrame/Series
    # Standardize features (glmnet standardizes by default)
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)

#     # Initialize an empty DataFrame to collect coefficients and other info
#     coef_final = pd.DataFrame(columns=['rowname', 'coefficients', 'lambda', 'alpha', 'misclassification'])

#     # alpha in glmnet controls mixing between ridge and lasso:
#     # alpha=0 ridge, alpha=1 lasso, 0<alpha<1 elastic net.
#     # LogisticRegressionCV in sklearn uses l1_ratio similarly.

#     alphas = [0]  # Just like your R code, could be extended to multiple values
#     for alpha in alphas:
#         # LogisticRegressionCV uses Cs (inverse of lambda), so we create an array of Cs.
#         # We'll use 'l1' or 'elasticnet' penalty depending on alpha.
#         # For alpha=0, only 'l2' penalty (ridge)
#         if alpha == 0:
#             penalty = 'l2'
#             solver = 'lbfgs'  # solver that supports l2
#             l1_ratio = None
#         else:
#             penalty = 'elasticnet'
#             solver = 'saga'  # supports elasticnet
#             l1_ratio = alpha
        
#         # Setup LogisticRegressionCV with 5-fold CV and scoring by misclassification error
#         # Note: 'scoring' in sklearn uses accuracy, so misclassification = 1 - accuracy
        # clf = LogisticRegressionCV(
        #     Cs=98,  # number of lambda (inverse) values to test
        #     # Cs = 1,
        #     penalty=penalty,
        #     solver=solver,
        #     l1_ratios=[l1_ratio] if l1_ratio is not None else None,
        #     scoring='accuracy',
        #     cv=5,
        #     max_iter=10000,
        #     refit=True,
        #     random_state=42
        # )
        
        # clf.fit(X_train_scaled, y_train)

        # # Get lambdas (inverse of Cs)
        # lambdas = 1 / clf.Cs_
#         # lambdas = [0.009986952]
# #         lambdas = [82.913430198994,
# # 75.5476295791045,
# # 68.8361863852889,
# # 62.7209693072994,
# # 57.1490112602739,
# # 52.0720506091863,
# # 47.4461131496514,
# # 43.2311312244038,
# # 39.3905966764181,
# # 35.8912446327188,
# # 32.7027653799633,
# # 29.7975418362056,
# # 27.1504103449436,
# # 24.7384427195653,
# # 22.5407476503643,
# # 20.5382897540097,
# # 18.713724698162,
# # 17.0512489731638,
# # 15.5364630095993,
# # 14.1562464561129,
# # 12.8986445372019,
# # 11.7527645066709,
# # 10.7086812998746,
# # 9.7573515675561,
# # 8.8905353466822,
# # 8.10072468982513,
# # 7.38107863491386,
# # 6.72536395209327,
# # 6.12790115446906,
# # 5.58351530510932,
# # 5.08749119421642,
# # 4.63553249823552,
# # 4.22372456715519,
# # 3.84850051768183,
# # 3.50661033860286,
# # 3.19509274074445,
# # 2.91124950770131,
# # 2.65262212517698,
# # 2.41697048650919,
# # 2.20225348993751,
# # 2.00661135955641,
# # 1.82834953682607,
# # 1.66592400311692,
# # 1.51792790615896,
# # 1.38307937456041,
# # 1.26021041485095,
# # 1.14825679488104,
# # 1.04624882595214,
# # 0.953302963837151,
# # 0.868614155942923,
# # 0.791448868329883,
# # 0.721138732191941,
# # 0.657074754765639,
# # 0.598702044526169,
# # 0.545515004982442,
# # 0.497052955442157,
# # 0.452896140816025,
# # 0.412662095899998,
# # 0.376002332644637,
# # 0.34259932171834,
# # 0.31216374222018,
# # 0.284431975720664,
# # 0.259163822925014,
# # 0.236140423181783,
# # 0.215162358816591,
# # 0.196047927871635,
# # 0.17863157029025,
# # 0.162762433914902,
# # 0.148303067877967,
# # 0.135128232068071,
# # 0.123123812360157,
# # 0.112185832213527,
# # 0.102219552076787,
# # 0.093138648799176,
# # 0.0848644679407298,
# # 0.0773253425051499,
# # 0.0704559721945666,
# # 0.0641968578095976,
# # 0.0584937858957471,
# # 0.0532973591723983,
# # 0.0485625686771995,
# # 0.0442484039199649,
# # 0.0403174976694266,
# # 0.0367358022961546,
# # 0.0334722948682779,
# # 0.0304987084456904,
# # 0.027789287245338,
# # 0.0253205635569472,
# # 0.0230711544769455,
# # 0.0210215766999798,
# # 0.0191540777638468,
# # 0.017452482286157,
# # 0.0159020518609116,
# # 0.0144893574014797,
# # 0.0132021628242746,
# # 0.0120293190656515,
# # 0.0109606675140516,
# # 0.00998695202096876,

# # ]
#         # st.write(lambdas)

#         # misclassification = 1 - mean test scores per fold
#         # LogisticRegressionCV stores scores_ per fold for each Cs in shape (n_folds, n_Cs)
#         mean_scores = np.mean(clf.scores_[1], axis=0)  # assuming binary class '1' in y_train
#         misclassification = 1 - mean_scores

#         for i, lam in enumerate(lambdas):
#             coefs = clf.coef_[0]  # shape (n_features,), coefficients at best C after refit
            
#             # But coef_ is for the best model refit on full data,
#             # To get coef per lambda, need to use path, which LogisticRegressionCV does not expose easily.
#             # Alternatively, retrain with each lambda:
            
#             # Retrain model at each lambda to get coefficients for that lambda
#             C = 1 / lam
#             model = LogisticRegression(
#                 penalty=penalty,
#                 solver=solver,
#                 l1_ratio=l1_ratio,
#                 C=C,
#                 max_iter=10000,
#                 random_state=42
#             )
#             model.fit(X_train_scaled, y_train)
            
#             # Create DataFrame with coefficients, including intercept as row 0
#             coefs_with_intercept = np.append(model.intercept_, model.coef_.flatten())
#             # rownames = ['(Intercept)'] + [f'X{i}' for i in range(1, X_train_scaled.shape[1]+1)]
#             feature_names = list(X_train.columns)

#             # Include intercept as the first "rowname"
#             rownames = ['(Intercept)'] + feature_names
#             df_coefs = pd.DataFrame({
#                 'rowname': rownames,
#                 'coefficients': coefs_with_intercept,
#                 'lambda': lam,
#                 'alpha': alpha,
#                 'misclassification': misclassification[i]
#             })

#             coef_final = pd.concat([coef_final, df_coefs], ignore_index=True)

#     st.write(coef_final)
    # scaler = StandardScaler()



    ############################## closest match with R results ######################
    scaler = StandardScaler(with_mean=True, with_std=True)

    X_train_scaled = scaler.fit_transform(X_train)
    


    ####### lambda set from R code ##############
    lambdas = [82.913430198994,
75.5476295791045,
68.8361863852889,
62.7209693072994,
57.1490112602739,
52.0720506091863,
47.4461131496514,
43.2311312244038,
39.3905966764181,
35.8912446327188,
32.7027653799633,
29.7975418362056,
27.1504103449436,
24.7384427195653,
22.5407476503643,
20.5382897540097,
18.713724698162,
17.0512489731638,
15.5364630095993,
14.1562464561129,
12.8986445372019,
11.7527645066709,
10.7086812998746,
9.7573515675561,
8.8905353466822,
8.10072468982513,
7.38107863491386,
6.72536395209327,
6.12790115446906,
5.58351530510932,
5.08749119421642,
4.63553249823552,
4.22372456715519,
3.84850051768183,
3.50661033860286,
3.19509274074445,
2.91124950770131,
2.65262212517698,
2.41697048650919,
2.20225348993751,
2.00661135955641,
1.82834953682607,
1.66592400311692,
1.51792790615896,
1.38307937456041,
1.26021041485095,
1.14825679488104,
1.04624882595214,
0.953302963837151,
0.868614155942923,
0.791448868329883,
0.721138732191941,
0.657074754765639,
0.598702044526169,
0.545515004982442,
0.497052955442157,
0.452896140816025,
0.412662095899998,
0.376002332644637,
0.34259932171834,
0.31216374222018,
0.284431975720664,
0.259163822925014,
0.236140423181783,
0.215162358816591,
0.196047927871635,
0.17863157029025,
0.162762433914902,
0.148303067877967,
0.135128232068071,
0.123123812360157,
0.112185832213527,
0.102219552076787,
0.093138648799176,
0.0848644679407298,
0.0773253425051499,
0.0704559721945666,
0.0641968578095976,
0.0584937858957471,
0.0532973591723983,
0.0485625686771995,
0.0442484039199649,
0.0403174976694266,
0.0367358022961546,
0.0334722948682779,
0.0304987084456904,
0.027789287245338,
0.0253205635569472,
0.0230711544769455,
0.0210215766999798,
0.0191540777638468,
0.017452482286157,
0.0159020518609116,
0.0144893574014797,
0.0132021628242746,
0.0120293190656515,
0.0109606675140516,
0.00998695202096876,

]
    n_samples = X_train_scaled.shape[0]

    # --- Initialize empty DataFrame
    coef_final = pd.DataFrame(columns=['rowname', 'coefficients', 'lambda', 'alpha', 'misclassification'])

    alpha = 0  # ridge only
    penalty = 'l2'
    solver = 'lbfgs'
    l1_ratio = None

    for lam in lambdas:
        # C = 1 / lam
        C = 2 / (lam * n_samples)
        # C1 = 2 / (lam * n_samples)
        # C = C1 / 100

        model = LogisticRegression(
            penalty=penalty,
            solver=solver,
            l1_ratio=l1_ratio,
            C=C,
            max_iter=1000000,
            tol=1e-7,
            # tol=1e-15,
            fit_intercept=True,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)

        # --- Get standardized coefs and back-transform
        scaled_coefs = model.coef_.flatten()
        unscaled_coefs = scaled_coefs / scaler.scale_

        intercept = model.intercept_[0]
        unscaled_intercept = intercept - np.sum(unscaled_coefs * scaler.mean_)

        # --- Combine intercept and coefficients
        coefs_with_intercept = np.insert(unscaled_coefs, 0, unscaled_intercept)
        rownames = ['(Intercept)'] + list(X_train.columns)

        # --- Predict and compute misclassification error
        y_pred = model.predict(X_train_scaled)
        misclassification = np.mean(y_pred != y_train)

        df_coefs = pd.DataFrame({
            'rowname': rownames,
            'coefficients': coefs_with_intercept,
            'lambda': lam,
            'alpha': alpha,
            'misclassification': misclassification
        })

        coef_final = pd.concat([coef_final, df_coefs], ignore_index=True)
    st.write(coef_final)

    
# 1. Merge on variable names (rowname from coef_final with varnam from Lmodel_coeff)
    joined_table = pd.merge(
        coef_final,
        glm_coeff_df,
        left_on='rowname',
        right_on='Feature',
        how='left'
    )

    # 2. Calculate absolute and relative differences between coefficients
    joined_table['Coefficient_Logistic'] = pd.to_numeric(joined_table['Coefficient_Logistic'], errors='coerce')
    joined_table['diff'] = (joined_table['Coefficient_Logistic'] - joined_table['coefficients']).abs()
    joined_table['diff_per'] = joined_table['diff'] / joined_table['Coefficient_Logistic'].abs()

    # 3. Replace NA/NaN values with 0
    joined_table.fillna(0, inplace=True)

    # 4. Group by lambda, alpha, misclassification and sum up diff
    joined_table_group = (
        joined_table.groupby(['lambda', 'alpha', 'misclassification'], as_index=False)
        .agg(sum_diff=('diff', 'sum'))
    )

    # 5. Merge summary back to joined_table
    joined_table = pd.merge(
        joined_table,
        joined_table_group,
        on=['lambda', 'alpha', 'misclassification'],
        how='left'
    )

    # 6. Filter rows to keep only best models based on conditions:
    min_sum_diff = joined_table['sum_diff'].min()
    min_misclassification = joined_table[joined_table['sum_diff'] == min_sum_diff]['misclassification'].min()
    min_lambda = joined_table[
        (joined_table['sum_diff'] == min_sum_diff) &
        (joined_table['misclassification'] == min_misclassification)
    ]['lambda'].min()

    filtered = joined_table[
        (joined_table['sum_diff'] == min_sum_diff) &
        (joined_table['misclassification'] == min_misclassification) &
        (joined_table['lambda'] == min_lambda)
    ]

    # 7. Add a column for type (assuming Data_level_list[i] is defined)
    # For example:
    Data_level_list = ['YourLabel']  # replace with your list
    i = 0  # index for your current data level
    filtered['Type'] = Data_level_list[i]

    # 8. Print the final filtered DataFrame
    st.write(filtered)
    joined_table_new = filtered
    df1 = joined_table_new[['rowname', 'alpha', 'coefficients', 'misclassification', 'Type']]

    # Copy input to output (just like in R)
    Overall_trans_model = df1.copy()

    # Append to Model_output (initialize Model_output if it doesn't exist yet)
    Model_output = pd.DataFrame()
    try:
        Model_output = pd.concat([Model_output, Overall_trans_model], ignore_index=True)
    except NameError:
        Model_output = Overall_trans_model.copy()

    
    Model_output['Odds-Ratio'] = np.where(Model_output['coefficients'] >= 0,
                            np.exp(Model_output['coefficients']),
                            1 / np.exp(Model_output['coefficients']))
    Model_output['Accuracy'] = 1 - Model_output['misclassification']
    st.write(Model_output)



    

    #####################################etest ######





    

def evaluate_performance(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = "N/A (multiclass)"

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    return tn, fp, fn, tp, accuracy, precision, recall




def overall_new_model():
  
    if 'final_flags' not in st.session_state:
        st.write('review data prep steps')
    else:
        temp_list = st.session_state.final_flags.loc[st.session_state.final_flags['Custom Categorization'] == "explainer", 'Column Names'].to_list()
        extra_top_columns = []

        # for col in temp_list:
        #     top_col = col + "_TOP"
        #     top2_col = col + "_TOP2"
        #     if top_col in st.session_state.data.columns:
        #         extra_top_columns.append(top_col)
        #     if top2_col in st.session_state.data.columns:
        #         extra_top_columns.append(top2_col)

        # temp_list += extra_top_columns
        # temp_list = list(set(temp_list))

        classfier_list = st.session_state.final_flags.loc[st.session_state.final_flags['Custom Categorization'] == "segment", 'Column Names'].to_list()
        st.session_state.classifier_list = classfier_list
    st.write("<h1 style='color: #0D3512; font-size: 35px; text-align:left; font-weight: normal;'> Overall New Model Creation </h1>", unsafe_allow_html=True)

    st.markdown("""
<div style="background-color: #f0f9f0; padding: 12px; border-radius: 8px; box-shadow: 0 3px 8px rgba(0, 0, 0, 0.1);">
<p style="font-size: 14px; color: #333333; font-family: Arial, sans-serif; line-height: 1.5;">
    Here, users can create overall new model by selecting features as per the requirements.
</p>
</div>
""", unsafe_allow_html=True)

    if "data" in st.session_state:
        # st.write(st.session_state.data)

        data = st.session_state.data.loc[:,temp_list]
        # st.write(data)
        
    # st.write(data)  
    columns = data.columns.tolist()
    data.fillna(0, inplace=True)
    columns_new = data.columns.tolist()
    # st.write(columns_new)
    st.write("<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> Select target column (y) </h1>", unsafe_allow_html=True)
    y_column = st.selectbox("Dependent variable (y):", columns_new)
    st.session_state.y_column = y_column
    st.write("<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> Selected explainer columns (X) </h1>", unsafe_allow_html=True)
    # x_columns = st.multiselect("Independent features (X):", options = [col for col in columns_new if col != y_column], default = [col for col in columns_new if col != y_column])
    # st.session_state.x_columns = x_columns
    # Filter _TOP and _TOP2 columns (excluding y_column)
    top_columns = [col for col in columns_new if col.endswith('_TOP') and col != y_column]
    # st.write(top_columns)
    # top_columns = [col for col in columns_new if col.endswith('_TOP') and not col.startswith(y_column) and col!=y_column]
    top2_columns = [col for col in columns_new if col.endswith('_TOP2') and col != y_column]


    st.markdown("""
    <div style="
        background: linear-gradient(to right, #e0f7fa, #b2ebf2);
        padding: 16px 24px;
        border-left: 6px solid #00796B;
        border-radius: 10px;
        color: black;  
        font-family: 'Segoe UI', sans-serif;
        font-size: 18px;
        font-weight: 500;
        margin-top: 20px;
    ">
        <strong>Stage 1:</strong> Feature Selection Based on <code>_TOP</code> Features
    </div>
    """, unsafe_allow_html=True) #004D40

    stage1_feature_options = [col for col in top_columns if col != f"{y_column}_TOP"] #####excluding the dependent variable _TOP from independent features
    # stage1_selected = st.multiselect("Select Stage 1 Features (_TOP):", options=top_columns, default=stage1_feature_options)
    stage1_selected = [
    # "rmexp_TOP",
    # "pv_TOP",
    # "stfexp_TOP",
    # "cknexp_TOP",
    # "rmcln_TOP",
    # "publicarea1_lobby_TOP",
    # "conosat_TOP",
    # "publicarea1_storeexp_TOP",
    # "digease_1_TOP",
    # "publicarea1_poolexp_TOP",
    # "spaexp_TOP",
    # "publicarea1_pubareaexp_TOP",
    # "Valet_Parking_avg_TOP",
    # "rmcln2_TOP",
    # "Bell_Services_avg_TOP",
    # "character_gchdine_mtastorytel_TOP",
    # "publicarea3_publicareaattb2_TOP",
    # "depart_checkout_dep_TOP",
    # "digitalrate_7_TOP",
    # "digitalrate_6_TOP",
    # "character_gchdine_pbanaparose_TOP",
    # "convgrid_2_TOP",
    # "character_dlhdine_goofys_TOP",
    # "bcosat_TOP",
    # "Win_TOP",
    # "loss_TOP",
    # "digitalrate_5_TOP",
    # "arrival_selfpark_TOP",
    # "digitalrate_3_TOP",
    # "dtr2_TOP",
    # "digitalrate_2_TOP",
    # "convgrid_1_TOP",
    # "publicarea1_fitness_TOP",
    # "entexpjoegardner_TOP",
    # "entexp_joegardner_TOP",
    # "losshandle_TOP",
    # "publicarea3_publicareaattb7_TOP",
    # "charexpbingbong_TOP"

#     "Bell_Services_avg",
# "Valet_Parking_avg",
"arrival_selfpark",
"bcosat"
# "character_dlhdine_goofys",
# "character_gchdine_mtastorytel",
# "character_gchdine_pbanaparose",
# "charexpbingbong",
# "cknexp",
# "conosat",
# "convgrid_1",
# "convgrid_2",
# "depart_checkout_dep",
# "digease_1",
# "entexpjoegardner",
# "losshandle",
# "pv",
# "rmcln",
# "rmcln2",
# "rmexp",
# "spaexp",
# "stfexp",
# # "Win",
# # "loss",
# "publicarea1_pubareaexp",
# "publicarea1_storeexp",
# "publicarea1_poolexp",
# "publicarea1_fitness",
# "publicarea3_publicareaattb2",
# "publicarea3_publicareaattb7",
# "digitalrate_2",
# "digitalrate_3",
# "digitalrate_6",
# "digitalrate_7",
# "dtr2",

]


    stage1_removed = list(set(top_columns) - set(stage1_selected))     # Determine features removed in Stage 1
    run_model_and_download(stage1_selected, y_column, data,top_label="Stage 1")  # Run model with Stage 1 features
    
    def run_stage2_feature_selection(stage1_removed, top2_columns):
        ### features(_TOP) that are removed from stage 1, are replaced with _TOP2 and kept for Stage2
        removed_base_names = [col.replace("_TOP", "") for col in stage1_removed]
        return [f"{base}_TOP2" for base in removed_base_names if f"{base}_TOP2" in top2_columns]

    stage2_candidates = run_stage2_feature_selection(stage1_removed, top2_columns)
    if stage2_candidates:
        # st.write("")
        st.markdown("""
    <div style="
        background: linear-gradient(to right, #e0f7fa, #b2ebf2);
        padding: 16px 24px;
        border-left: 6px solid #00796B;
        border-radius: 10px;
        color: black;  
        font-family: 'Segoe UI', sans-serif;
        font-size: 18px;
        font-weight: 500;
        margin-top: 20px;
    ">
        <strong>Stage 2:</strong> Select additional <code>_TOP2</code> Features
    </div>
    """, unsafe_allow_html=True) 
        stage2_selected = st.multiselect("Stage 2 Features (_TOP2):", options=stage2_candidates, default=[])
        if stage2_selected:
            final_x_columns = stage1_selected + stage2_selected
            run_model_and_download(final_x_columns, y_column, data,top_label="Stage 2")  #Run model again with stage 2 features
        #     excel_buffer = run_model_and_download(final_x_columns, y_column, data,top_label="Stage 2")  #Run model again with stage 2 features
        #     st.download_button(
        #     label="📥 Download Stage 2 Model Results (Excel)",
        #     data=excel_buffer,
        #     file_name="Stage2_Model_Results.xlsx",
        #     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        #     key="download_button_stage2"
        # )
    # x_columns = stage1_selected
    # st.session_state.x_columns = x_columns


def quadrant_graph():
    import plotly.graph_objects as go
    import streamlit as st
    import pandas as pd
    import numpy as np

    # Load and prepare data
    merged_df = st.session_state.merged_overall_segment.copy()

    # Rename for clarity
    merged_df.rename(columns={'Odds-Ratio': 'Odds Ratio', 'Actual %': 'Actual Perc'}, inplace=True)

    # Ensure correct types
    merged_df['Odds Ratio'] = pd.to_numeric(merged_df['Odds Ratio'], errors='coerce')
    merged_df['Actual Perc'] = pd.to_numeric(merged_df['Actual Perc'], errors='coerce')
    merged_df.dropna(subset=['Odds Ratio', 'Actual Perc', 'Independent Variables'], inplace=True)

    # Dropdown to select Segment
    segment_list = merged_df['Segment'].dropna().unique().tolist()
    selected_segment = st.selectbox("Select a Segment to view Quadrant Graphs for each Type:", options=segment_list)

    # Filter data for selected Segment
    segment_df = merged_df[merged_df['Segment'] == selected_segment]

    # Get all Types under this Segment
    type_list = segment_df['Type'].dropna().unique().tolist()

    if not type_list:
        st.warning("No Types found for the selected Segment.")
        return

    for current_type in type_list:
        type_df = segment_df[segment_df['Type'] == current_type]

        if type_df.empty:
            st.warning(f"No data available for {current_type}")
            continue

        # Centroids
        centroid_x = type_df['Odds Ratio'].mean()
        centroid_y = type_df['Actual Perc'].mean()

        fig = go.Figure()

        for i in range(len(type_df)):
            fig.add_trace(go.Scatter(
                x=[type_df['Odds Ratio'].iloc[i]],
                y=[type_df['Actual Perc'].iloc[i]],
                mode='markers',
                text=[type_df['Independent Variables'].iloc[i]],
                textposition='top center',
                marker=dict(size=10)
            ))

        # Axis ranges
        x_min, x_max = type_df['Odds Ratio'].min(), type_df['Odds Ratio'].max()
        y_min, y_max = type_df['Actual Perc'].min(), type_df['Actual Perc'].max()

        # Add quadrant lines
        fig.add_shape(type="line",
                      x0=x_min, x1=x_max,
                      y0=centroid_y, y1=centroid_y,
                      line=dict(color="Black", width=2))
        fig.add_shape(type="line",
                      x0=centroid_x, x1=centroid_x,
                      y0=y_min, y1=y_max,
                      line=dict(color="Black", width=2))

        # Axis titles and layout
        fig.update_xaxes(title='Odds Ratio')
        fig.update_yaxes(title='Actual %')

        fig.update_layout(
            title=f'Quadrant Graph: {selected_segment} | {current_type}',
            showlegend=False,
            height=600
        )

        st.plotly_chart(fig)



def category_level_model1():
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    import streamlit as st

    st.write("<h1 style='color: #0D3512; font-size: 35px; text-align:left; font-weight: normal;'> Segment Level Model Creation </h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color: #f0f9f0; padding: 12px; border-radius: 8px; box-shadow: 0 3px 8px rgba(0, 0, 0, 0.1);">
    <p style="font-size: 14px; color: #333333; font-family: Arial, sans-serif; line-height: 1.5;">
        Here, users can create new model based on the unique options available in selected column.
    </p>
    </div>
    """, unsafe_allow_html=True)

    if "data" in st.session_state:
        data = st.session_state.data.copy()
        data.fillna(0, inplace=True)
    else:
        st.error("Data not found. Please complete data preparation steps.")
        return

    test_size_cat = st.session_state.get("test_size_selector", 30)
    y_column_cat = st.session_state.get("y_column", None)
    x_columns_cat = st.session_state.get("x_columns", [])
    classifier_list = st.session_state.get("classifier_list", [])

    if not y_column_cat or not x_columns_cat or not classifier_list:
        st.error("Missing required session state variables. Please complete the overall model setup first.")
        return

    if st.session_state.new_df_after_col_drop is not None:
        st.write("<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> Select column(s) to create segment level models: </h1>", unsafe_allow_html=True)
        selected_columns = st.multiselect("", options=st.session_state.classifier_list, default=st.session_state.classifier_list)

        results_dict = {}
        all_log_odds_dfs = []

        for selected_column in selected_columns:
            unique_values = data[selected_column].unique()
            col_results = {}

            for category in unique_values:
                data1 = data[data[selected_column] == category]
                X = data1[x_columns_cat]
                y = data1[y_column_cat]

                if len(X.index) < 10:
                    col_results[category] = {"error": f"{category} - sample data has less than 10 records"}
                    continue
                # Calculate Actual % here
                no_of_records = len(data1)
                actual_pct = data1[x_columns_cat].sum() / no_of_records * 100
                actual_pct_df = actual_pct.reset_index()
                actual_pct_df.columns = ['Feature Name', 'Actual %'] 
                # st.write(actual_pct_df)  
                # 
                ##### overall % ##################################
                # # Calculate Overall % (proportion of y_column_cat == 1 in this segment)
                overall_pct = (data1[y_column_cat].sum() / len(data1)) * 100  # Percentage
                # overall_pct = round(overall_pct, 2)  # Optional: round to 2 decimals  


                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_cat/100, random_state=42)
                model = LogisticRegression()
                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Assume evaluate_performance function is defined elsewhere and imported
                train_tn, train_fp, train_fn, train_tp, train_accuracy, train_precision, train_recall = evaluate_performance(y_train, y_train_pred)
                test_tn, test_fp, test_fn, test_tp, test_accuracy, test_precision, test_recall = evaluate_performance(y_test, y_test_pred)

                performance_df = pd.DataFrame({
                    'Performance Metrics': ['True Negative','False Positive','False Negative','True Positive','Accuracy','Precision','Recall'],
                    'Training': [train_tn, train_fp, train_fn, train_tp ,train_accuracy ,train_precision ,train_recall],
                    'Testing': [test_tn, test_fp, test_fn, test_tp, test_accuracy, test_precision, test_recall]
                })

                # Calculate coefficients and odds ratios properly
                coef_series = pd.Series(model.coef_[0], index=X.columns)
                odds_ratio_series = coef_series.apply(np.exp)
                intercept_odds_ratio = np.exp(model.intercept_[0])


                y_pred_tr = model.predict(X_train)
                y_pred_tr = [1 if val >= 0.6 else 0 for val in [udf.sigmoid(value) for value in y_pred_tr]]
                accuracy_tr = accuracy_score(y_train, y_pred_tr)

                y_pred_te = model.predict(X_test)
                y_pred_te = [1 if val >= 0.6 else 0 for val in [udf.sigmoid(value) for value in y_pred_te]]
                accuracy_te = accuracy_score(y_test, y_pred_te)

                accuracy_gap = abs(round((accuracy_tr - accuracy_te) * 100, 2))


                log_odds_df = pd.DataFrame({
                    'Feature Name': list(coef_series.index) + ['Intercept'],
                    'Coefficient': list(coef_series.values) + [model.intercept_[0]],
                    'Odds-Ratio': list(odds_ratio_series.values) + [intercept_odds_ratio]
                })
                log_odds_df = log_odds_df.merge(actual_pct_df, on='Feature Name', how='left')

                # Fill NA Actual % for Intercept or missing features with empty string or 0
                log_odds_df['Actual %'] = log_odds_df['Actual %'].fillna('')
                
                ############## for the download option ##############
                # log_odds_df_copy = log_odds_df
                log_odds_df['Segment'] = selected_column
                log_odds_df['Type'] = category
                log_odds_df['Dependent Variable'] = y_column_cat

                log_odds_df['Overall %'] = overall_pct
                log_odds_df = log_odds_df.rename(columns={'Feature Name':'Independent Variables'}).reset_index(drop=True)
                
                log_odds_df = pd.merge(log_odds_df,st.session_state.feature_description, on= 'Independent Variables', how='left')
                log_odds_df['Model Accuracy'] = accuracy_gap
                desired_order = [
                    "Dependent Variable", "Segment", "Type", "Independent Variables",
                    "Coefficient", "Odds-Ratio", "Actual %", "Overall %", "Model Accuracy", "Description"
                ]
                log_odds_df = log_odds_df[desired_order]

                all_log_odds_dfs.append(log_odds_df)

                log_odds_df_copy = log_odds_df
                log_odds_df_copy= log_odds_df_copy[["Independent Variables","Odds-Ratio"]]
                col_results[category] = {
                    "sample_size": len(X.index),
                    "performance_df": performance_df,
                    "accuracy_df": performance_df[performance_df['Performance Metrics'] == 'Accuracy'],
                    "log_odds_df": log_odds_df_copy
                }

            results_dict[selected_column] = col_results
            combined_log_odds_df = pd.concat(all_log_odds_dfs, ignore_index=True)
            # st.write(combined_log_odds_df)

        if results_dict:
            st.write("<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> View Results For Column: </h1>", unsafe_allow_html=True)

            selectbox_options = [""] + list(results_dict.keys())
            selected_col_display = st.selectbox("", options=selectbox_options)

            if selected_col_display != "":
                st.write(f"<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> Result for '{selected_col_display}' </h1>", unsafe_allow_html=True)
                st.dataframe(combined_log_odds_df[combined_log_odds_df['Segment'] == selected_col_display], use_container_width=True)

                for cat, result in results_dict[selected_col_display].items():
                    if "error" in result:
                        st.write(f"<h1 style='color: #FF0000; font-size: 20px; text-align:left; font-weight: normal;'>{result['error']}</h1>", unsafe_allow_html=True)
                        continue

        merged_overall_segment = pd.concat([st.session_state.ridge_regression_df_f_copy, combined_log_odds_df], ignore_index=True)
        st.session_state.merged_overall_segment = merged_overall_segment
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                combined_log_odds_df.to_excel(writer, index=False)

            # Rewind the buffer to the beginning
            excel_buffer.seek(0)

            # Create a download button
            st.download_button(
                label="📥 Download Segment Level Model Results (Excel)",
                data=excel_buffer,
                file_name="Segment_Model_Results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        with col2:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                merged_overall_segment.to_excel(writer, index=False)

            # Rewind the buffer to the beginning
            excel_buffer.seek(0)

            # Create a download button
            st.download_button(
                label="📥 Download Overall+Segment Level Model Results (Excel)",
                data=excel_buffer,
                file_name="Combined_Model_Results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


    else:
        st.write('<span style="color:red; font-size:25px; font-weight:normal;"> **** Please upload New data, New data keys files on Data Preparation page **** </span >', unsafe_allow_html=True)



def run():
    st.divider()
    
    if st.session_state.new_df_after_col_drop is not None: # is not None and st.session_state.new_file_keys is not None:
        sub_page = st.sidebar.radio(
            "Choose a sub task:",
            ["Overall new model", "Segment level model","Quadrant Graph" ]  # Added "Binning/Bucketing" to the list
        )
        if sub_page == "Overall new model":
            overall_new_model()
        elif sub_page == "Segment level model":
            category_level_model1()
        elif sub_page == "Quadrant Graph":
            quadrant_graph()
    else:
        st.write('<span style="color:red; font-size:25px; font-weight:normal;"> **** Please upload Old data, New data and New Data Codes files on Data Integration page **** </span >', unsafe_allow_html=True)


