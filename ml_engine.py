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
import io
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

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
    
    X_with_const = sm.add_constant(X, has_constant='add')
    glm_model = sm.GLM(y, X_with_const, family=sm.families.Binomial())
    glm_result = glm_model.fit()

    glm_coeff_df = glm_result.summary2().tables[1].reset_index()
    glm_coeff_df.rename(columns={'index': 'Feature', 'Coef.': 'Coefficient_Logistic'}, inplace=True)
    glm_coeff_df = glm_coeff_df[['Feature', 'Coefficient_Logistic']]

    logistic_regression_df = glm_coeff_df    
    logistic_regression_df['Rank_logistic'] = logistic_regression_df['Coefficient_Logistic'].rank(ascending=False).astype(int)
    logistic_regression_df = logistic_regression_df.sort_values(by=['Coefficient_Logistic'], ascending=False)

    
    # Task - Fit Ridge Regression for different values of alpha.
    X_train = X
    y_train = y
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_scaled = scaler.fit_transform(X_train)
    n_samples = X_train_scaled.shape[0]
    alphas = [0]  
    for alpha in alphas:
        # LogisticRegressionCV uses Cs (inverse of lambda), so we create an array of Cs.
        # We'll use 'l1' or 'elasticnet' penalty depending on alpha.
        # For alpha=0, only 'l2' penalty (ridge)
        if alpha == 0:
            penalty = 'l2'
            solver = 'lbfgs'  # solver that supports l2
            l1_ratio = None
        else:
            penalty = 'elasticnet'
            solver = 'saga'  # supports elasticnet
            l1_ratio = alpha
        
    clf = LogisticRegressionCV(
            Cs=98,  # number of lambda (inverse) values to test
            # Cs = 1,
            penalty=penalty,
            solver=solver,
            l1_ratios=[l1_ratio] if l1_ratio is not None else None,
            scoring='accuracy',
            cv=5,
            max_iter=10000,
            refit=True,
            random_state=42
        )
        
    clf.fit(X_train_scaled, y_train)

    # Get lambdas (inverse of Cs)
    lambdas = 1 / clf.Cs_
    # --- Initialize empty DataFrame
    coef_final = pd.DataFrame(columns=['rowname', 'coefficients', 'lambda', 'alpha', 'misclassification'])

    alpha = 0  # ridge only
    penalty = 'l2'
    solver = 'lbfgs'
    l1_ratio = None

    for lam in lambdas:
        # C = 1 / lam #### this was for when python generated lambdas were there
        C1 = 2 / (lam * n_samples) 
        C = C1 / 100

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
    # st.write(coef_final) ##was printing


    # Step 1: Sort coef_final by lambda ascending
    coef_final['lambda'] = pd.to_numeric(coef_final['lambda'], errors='coerce')
    coef_final_sorted = coef_final.sort_values(by='lambda')

    # Step 2: Create 5 buckets of 20 lambdas each
    unique_lambdas = sorted(coef_final_sorted['lambda'].unique())
    n_buckets = 5
    bucket_size = len(unique_lambdas) // n_buckets
    lambda_buckets = [unique_lambdas[i * bucket_size:(i + 1) * bucket_size] for i in range(n_buckets)]

    # Optional: handle any leftover lambdas in the last bucket
    if len(unique_lambdas) % n_buckets != 0:
        lambda_buckets[-1] += unique_lambdas[n_buckets * bucket_size:]

    # Prepare output
    Model_output = pd.DataFrame()
    Data_level_list = ['Overall']  # Update this if you have multiple labels
    found_match = False

    # Step 3: Iterate over buckets, break if match found
    for i, lambda_group in enumerate(lambda_buckets):
        if found_match:
            break

        # Filter coef_final for current lambda bucket
        coef_bucket = coef_final_sorted[coef_final_sorted['lambda'].isin(lambda_group)].copy()

        # --- Merge with glm coefficients
        joined_table = pd.merge(
            coef_bucket,
            glm_coeff_df,
            left_on='rowname',
            right_on='Feature',
            how='left'
        )

        # --- Calculate diffs
        joined_table['Coefficient_Logistic'] = pd.to_numeric(joined_table['Coefficient_Logistic'], errors='coerce')
        joined_table['diff'] = (joined_table['Coefficient_Logistic'] - joined_table['coefficients']).abs()
        joined_table['diff_per'] = joined_table['diff'] / joined_table['Coefficient_Logistic'].abs()

        # --- Fill NAs
        joined_table.fillna(0, inplace=True)

        # --- Group and summarize
        joined_table_group = (
            joined_table.groupby(['lambda', 'alpha', 'misclassification'], as_index=False)
            .agg(sum_diff=('diff', 'sum'))
        )

        joined_table = pd.merge(
            joined_table,
            joined_table_group,
            on=['lambda', 'alpha', 'misclassification'],
            how='left'
        )

        # --- Get best match in current bucket
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
        # st.write(filtered) ##was printing
        # If we found a good match, set flag and store results
        if not filtered.empty:
            found_match = True
            joined_table_new = filtered
            df1 = joined_table_new[['rowname', 'alpha', 'coefficients', 'misclassification']].copy()

            # Copy input to output 
            Overall_trans_model = df1.copy()

            # Append to Model_output
            Model_output = pd.concat([Model_output, Overall_trans_model], ignore_index=True)

            # Calculate odds-ratios and accuracy
            Model_output['Odds-Ratio'] = np.where(
                Model_output['coefficients'] >= 0,
                np.exp(Model_output['coefficients']),
                1 / np.exp((Model_output['coefficients'])) ############# checked ###
            )
            Model_output['Model Accuracy'] = 1 - Model_output['misclassification']

            # Output final model
            # st.write(Model_output) ##was printing



            Model_output.rename(columns={'rowname': 'Feature'}, inplace=True)
            # sum_summary1_copy = pd.merge(st.session_state.sum_summary1_copy,st.session_state.percent1_table,on ='Feature',how='left')
            if 'percent1_table' in st.session_state:
                st.session_state.sum_summary1_copy = pd.concat(
                    [st.session_state.sum_summary1_copy, st.session_state.percent1_table], 
                    ignore_index=True
                )

            Model_output = pd.merge(Model_output,st.session_state.sum_summary1_copy, on = 'Feature',how='left')
            Model_output['Segment'] = 'Overall'
            Model_output['Type'] = 'Overall'
            Model_output['Dependent Variable'] = y_column
            Model_output = Model_output.rename(columns={
                    'Feature': 'Independent Variables',
                    'coefficients':'Coefficient'
                    # 'Model Value': 'Coefficient'
                }).reset_index(drop=True)
            # Fill NA Actual % for Intercept or missing features with empty string or 0
            Model_output['Actual %'] = Model_output['Actual %'].fillna('')
            feature_description = st.session_state.required_features_df
            feature_description = feature_description.rename(columns={'Feature':'Independent Variables'}).reset_index(drop=True)
            
            if 'new_feature_description_table' in st.session_state:
                feature_description = pd.concat(
                    [feature_description, st.session_state.new_feature_description_table], 
                    ignore_index=True
                )
            # st.write(feature_description)


            st.session_state.feature_description = feature_description
            Model_output = pd.merge(Model_output,feature_description, on= 'Independent Variables', how='left')
            # Model_output['Model Accuracy'] = Model_output

            ################ overall % ###########################
            # Get the Overall % value for the dependent variable
            overall_actual_pct_row = st.session_state.sum_summary1_copy[
                st.session_state.sum_summary1_copy['Feature'] == st.session_state.y_column
            ]
            # Extract the Actual % value, use .values[0] safely
            overall_actual_pct = ''
            if not overall_actual_pct_row.empty and 'Actual %' in overall_actual_pct_row.columns:
                overall_actual_pct = overall_actual_pct_row['Actual %'].values[0]
            Model_output['Overall %'] = overall_actual_pct
            ################################################################################
            desired_order = [
            "Dependent Variable", "Segment", "Type", "Independent Variables", "Coefficient",
            "Odds-Ratio", "Actual %", "Overall %", "Model Accuracy", "Description"
        ]
            Model_output = Model_output[desired_order]  
            Model_output = Model_output.drop_duplicates()

            st.write(f"<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> {top_label} selected model : </h1>", unsafe_allow_html=True)
            st.dataframe(Model_output[['Independent Variables', 'Odds-Ratio']], hide_index=True, use_container_width=True) ### this is getting displayed
            st.session_state.Model_output = Model_output
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                Model_output.to_excel(writer, index=False)
            excel_buffer.seek(0)
            # return excel_buffer
            # Create a download button
            st.download_button(
            label=f"📥 Download {top_label} Model Results (Excel)",
            data=excel_buffer,
            file_name=f"{top_label}_Model_Results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"download_button_{top_label}"
    )
            






    

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
        data = st.session_state.data.loc[:,temp_list]
    columns = data.columns.tolist()
    data.fillna(0, inplace=True)
    columns_new = data.columns.tolist()
    # st.write(columns_new)
    st.write("<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> Select target column (y) </h1>", unsafe_allow_html=True)
    y_column = st.selectbox("Dependent variable (y):", columns_new)
    st.session_state.y_column = y_column
    st.write("<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> Selected explainer columns (X) </h1>", unsafe_allow_html=True)
    # Filter _TOP and _TOP2 columns (excluding y_column)
    top_columns = [col for col in columns_new if col.endswith('_TOP') and col != y_column]
    # st.write(top_columns)
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
    stage1_selected = st.multiselect("Select Stage 1 Features (_TOP):", options=top_columns, default=stage1_feature_options)

    stage1_removed = list(set(top_columns) - set(stage1_selected))     # Determine features removed in Stage 1
    run_model_and_download(stage1_selected, y_column, data,top_label="Stage 1")  # Run model with Stage 1 features
    
    def run_stage2_feature_selection(stage1_removed, top2_columns):
        ### features(_TOP) that are removed from stage 1, are replaced with _TOP2 and kept for Stage2
        removed_base_names = [col.replace("_TOP", "") for col in stage1_removed]
        return [f"{base}_TOP2" for base in removed_base_names if f"{base}_TOP2" in top2_columns]

    stage2_candidates = run_stage2_feature_selection(stage1_removed, top2_columns)
    if stage2_candidates:
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
       

def quadrant_graph():

    import plotly.graph_objects as go
    import streamlit as st
    import pandas as pd
    import numpy as np
    
    if 'merged_overall_segment' in st.session_state:
        merged_df = st.session_state.merged_overall_segment.copy()
    else:
        merged_df = st.session_state.Model_output.copy()

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
        centroid_x = type_df['Actual Perc'].mean()
        centroid_y = type_df['Odds Ratio'].mean()

        # Determine quadrant and color
        def get_quadrant_color(x, y):
            if x >= centroid_x and y >= centroid_y:
                return 'blue'    # Quadrant I
            elif x < centroid_x and y >= centroid_y:
                return 'green'   # Quadrant II
            elif x < centroid_x and y < centroid_y:
                return 'orange'  # Quadrant III
            else:
                return 'red'     # Quadrant IV

        type_df['Color'] = type_df.apply(lambda row: get_quadrant_color(row['Actual Perc'], row['Odds Ratio']), axis=1)

        # Create scatter plot with color-coded points
        fig = go.Figure()

        for color in type_df['Color'].unique():
            df_color = type_df[type_df['Color'] == color]
            fig.add_trace(go.Scatter(
                x=df_color['Actual Perc'],
                y=df_color['Odds Ratio'],
                mode='markers',
                text=df_color['Independent Variables'],
                textposition='top center',
                marker=dict(size=10, color=color),
                name=f'Quadrant - {color.capitalize()}'
            ))

        # Axis ranges
        x_min, x_max = type_df['Actual Perc'].min(), type_df['Actual Perc'].max()
        y_min, y_max = type_df['Odds Ratio'].min(), type_df['Odds Ratio'].max()

        # Add quadrant lines
        fig.add_shape(type="line",
                      x0=x_min, x1=x_max,
                      y0=centroid_y, y1=centroid_y,
                      line=dict(color="black", width=2))

        fig.add_shape(type="line",
                      x0=centroid_x, x1=centroid_x,
                      y0=y_min, y1=y_max,
                      line=dict(color="black", width=2))

        # Axis titles
        fig.update_xaxes(title='Actual %')
        fig.update_yaxes(title='Odds Ratio')

        fig.update_layout(
            title=f'Quadrant Graph: {selected_segment} | {current_type}',
            showlegend=False,
            height=600
        )

        # st.plotly_chart(fig)
        # --- BAR CHART BELOW: Color by Quadrant ---
        sorted_df = type_df.sort_values(by='Odds Ratio', ascending=True)

        bar_fig = go.Figure(go.Bar(
        x=sorted_df['Odds Ratio'],                     
        y=sorted_df['Independent Variables'],          
        orientation='h',                               
        marker_color=sorted_df['Color']
    ))

        bar_fig.update_layout(
            title=f'Odds Ratio Bar Chart: {selected_segment} | {current_type}',
            xaxis_title='Independent Variables',
            yaxis_title='Odds Ratio',
            height=500
        )
        bar_fig.update_xaxes(tickangle=45)
        st.plotly_chart(bar_fig) ### bar chart 
        st.plotly_chart(fig) #### quadrant graph

        # col1, col2 = st.columns(2)

        # with col1:
        #     st.plotly_chart(bar_fig, use_container_width=True)

        # with col2:
        #     st.plotly_chart(fig, use_container_width=True)


# def category_level_model1():
#     import numpy as np
#     import pandas as pd
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.model_selection import train_test_split
#     import streamlit as st

#     st.write("<h1 style='color: #0D3512; font-size: 35px; text-align:left; font-weight: normal;'> Segment Level Model Creation </h1>", unsafe_allow_html=True)
#     st.markdown("""
#     <div style="background-color: #f0f9f0; padding: 12px; border-radius: 8px; box-shadow: 0 3px 8px rgba(0, 0, 0, 0.1);">
#     <p style="font-size: 14px; color: #333333; font-family: Arial, sans-serif; line-height: 1.5;">
#         Here, users can create new model based on the unique options available in selected column.
#     </p>
#     </div>
#     """, unsafe_allow_html=True)

#     if "data" in st.session_state:
#         data = st.session_state.data.copy()
#         data.fillna(0, inplace=True)
#     else:
#         st.error("Data not found. Please complete data preparation steps.")
#         return

#     test_size_cat = st.session_state.get("test_size_selector", 30)
#     #####################################
    
#     columns = data.columns.tolist()
#     data.fillna(0, inplace=True)
#     columns_new = data.columns.tolist()
#     # st.write(columns_new)
#     st.write("<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> Select target column (y) </h1>", unsafe_allow_html=True)
#     y_column_cat = st.selectbox("Dependent variable (y):", columns_new)
#     # st.session_state.y_column = y_column
#     ###########################
#     # y_column_cat = st.session_state.get("y_column", None)
#     x_columns_cat = st.session_state.get("x_columns", [])
#     classifier_list = st.session_state.get("classifier_list", [])

#     if not y_column_cat or not x_columns_cat or not classifier_list:
#         st.error("Missing required session state variables. No segments are specified. Please complete the overall model setup first.")
#         return

#     if st.session_state.new_df_after_col_drop is not None:
#         st.write("<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> Select column(s) to create segment level models: </h1>", unsafe_allow_html=True)
#         selected_columns = st.multiselect("", options=st.session_state.classifier_list, default=st.session_state.classifier_list)

#         results_dict = {}
#         all_log_odds_dfs = []

#         for selected_column in selected_columns:
#             unique_values = data[selected_column].unique()
#             col_results = {}

#             for category in unique_values:
#                 data1 = data[data[selected_column] == category]
#                 X = data1[x_columns_cat]
#                 y = data1[y_column_cat]

#                 if len(X.index) < 10:
#                     col_results[category] = {"error": f"{category} - sample data has less than 10 records"}
#                     continue
#                 # Calculate Actual % here
#                 no_of_records = len(data1)
#                 actual_pct = data1[x_columns_cat].sum() / no_of_records * 100
#                 actual_pct_df = actual_pct.reset_index()
#                 actual_pct_df.columns = ['Feature Name', 'Actual %'] 
#                 # st.write(actual_pct_df)  
#                 # 
#                 ##### overall % ##################################
#                 # # Calculate Overall % (proportion of y_column_cat == 1 in this segment)
#                 overall_pct = (data1[y_column_cat].sum() / len(data1)) * 100  # Percentage
#                 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_cat/100, random_state=42)
#                 model = LogisticRegression()
#                 model.fit(X_train, y_train)

#                 y_train_pred = model.predict(X_train)
#                 y_test_pred = model.predict(X_test)

#                 # Assume evaluate_performance function is defined elsewhere and imported
#                 train_tn, train_fp, train_fn, train_tp, train_accuracy, train_precision, train_recall = evaluate_performance(y_train, y_train_pred)
#                 test_tn, test_fp, test_fn, test_tp, test_accuracy, test_precision, test_recall = evaluate_performance(y_test, y_test_pred)

#                 performance_df = pd.DataFrame({
#                     'Performance Metrics': ['True Negative','False Positive','False Negative','True Positive','Accuracy','Precision','Recall'],
#                     'Training': [train_tn, train_fp, train_fn, train_tp ,train_accuracy ,train_precision ,train_recall],
#                     'Testing': [test_tn, test_fp, test_fn, test_tp, test_accuracy, test_precision, test_recall]
#                 })

#                 # Calculate coefficients and odds ratios properly
#                 coef_series = pd.Series(model.coef_[0], index=X.columns)
#                 odds_ratio_series = coef_series.apply(np.exp)
#                 intercept_odds_ratio = np.exp(model.intercept_[0])


#                 y_pred_tr = model.predict(X_train)
#                 y_pred_tr = [1 if val >= 0.6 else 0 for val in [udf.sigmoid(value) for value in y_pred_tr]]
#                 accuracy_tr = accuracy_score(y_train, y_pred_tr)

#                 y_pred_te = model.predict(X_test)
#                 y_pred_te = [1 if val >= 0.6 else 0 for val in [udf.sigmoid(value) for value in y_pred_te]]
#                 accuracy_te = accuracy_score(y_test, y_pred_te)

#                 accuracy_gap = abs(round((accuracy_tr - accuracy_te) * 100, 2))


#                 log_odds_df = pd.DataFrame({
#                     'Feature Name': list(coef_series.index) + ['Intercept'],
#                     'Coefficient': list(coef_series.values) + [model.intercept_[0]],
#                     'Odds-Ratio': list(odds_ratio_series.values) + [intercept_odds_ratio]
#                 })
#                 log_odds_df = log_odds_df.merge(actual_pct_df, on='Feature Name', how='left')

#                 # Fill NA Actual % for Intercept or missing features with empty string or 0
#                 log_odds_df['Actual %'] = log_odds_df['Actual %'].fillna('')
                
#                 ############## for the download option ##############
#                 log_odds_df['Segment'] = selected_column
#                 log_odds_df['Type'] = category
#                 log_odds_df['Dependent Variable'] = y_column_cat

#                 log_odds_df['Overall %'] = overall_pct
#                 log_odds_df = log_odds_df.rename(columns={'Feature Name':'Independent Variables'}).reset_index(drop=True)
                
#                 log_odds_df = pd.merge(log_odds_df,st.session_state.feature_description, on= 'Independent Variables', how='left')
#                 log_odds_df['Model Accuracy'] = accuracy_gap
#                 desired_order = [
#                     "Dependent Variable", "Segment", "Type", "Independent Variables",
#                     "Coefficient", "Odds-Ratio", "Actual %", "Overall %", "Model Accuracy", "Description"
#                 ]
#                 log_odds_df = log_odds_df[desired_order]

#                 all_log_odds_dfs.append(log_odds_df)

#                 log_odds_df_copy = log_odds_df
#                 log_odds_df_copy= log_odds_df_copy[["Independent Variables","Odds-Ratio"]]
#                 col_results[category] = {
#                     "sample_size": len(X.index),
#                     "performance_df": performance_df,
#                     "accuracy_df": performance_df[performance_df['Performance Metrics'] == 'Accuracy'],
#                     "log_odds_df": log_odds_df_copy
#                 }

#             results_dict[selected_column] = col_results
#             combined_log_odds_df = pd.concat(all_log_odds_dfs, ignore_index=True)
#         if results_dict:
#             st.write("<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> View Results For Column: </h1>", unsafe_allow_html=True)

#             selectbox_options = [""] + list(results_dict.keys())
#             selected_col_display = st.selectbox("", options=selectbox_options)

#             if selected_col_display != "":
#                 st.write(f"<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> Result for '{selected_col_display}' </h1>", unsafe_allow_html=True)
#                 st.dataframe(combined_log_odds_df[combined_log_odds_df['Segment'] == selected_col_display], use_container_width=True)

#                 for cat, result in results_dict[selected_col_display].items():
#                     if "error" in result:
#                         st.write(f"<h1 style='color: #FF0000; font-size: 20px; text-align:left; font-weight: normal;'>{result['error']}</h1>", unsafe_allow_html=True)
#                         continue

#         merged_overall_segment = pd.concat([st.session_state.Model_output, combined_log_odds_df], ignore_index=True)
#         st.session_state.merged_overall_segment = merged_overall_segment
#         st.divider()
#         col1, col2 = st.columns(2)
#         with col1:
#             excel_buffer = io.BytesIO()
#             with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
#                 combined_log_odds_df.to_excel(writer, index=False)

#             # Rewind the buffer to the beginning
#             excel_buffer.seek(0)

#             # Create a download button
#             st.download_button(
#                 label="📥 Download Segment Level Model Results (Excel)",
#                 data=excel_buffer,
#                 file_name="Segment_Model_Results.xlsx",
#                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#             )

#         with col2:
#             excel_buffer = io.BytesIO()
#             with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
#                 merged_overall_segment.to_excel(writer, index=False)

#             # Rewind the buffer to the beginning
#             excel_buffer.seek(0)

#             # Create a download button
#             st.download_button(
#                 label="📥 Download Overall+Segment Level Model Results (Excel)",
#                 data=excel_buffer,
#                 file_name="Combined_Model_Results.xlsx",
#                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#             )


#     else:
#         st.write('<span style="color:red; font-size:25px; font-weight:normal;"> **** Please upload New data, New data keys files on Data Preparation page **** </span >', unsafe_allow_html=True)

def run_model_segment_and_download(x_columns, y_column, data, top_label="Overall"):
    ## Function called inside Overall new Model, 
    ## This Function is called after selection of Stage1 features as well as if Stage2 Features are selected in Overall New Model.
    # st.session_state.x_columns = x_columns
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
    
    X_with_const = sm.add_constant(X, has_constant='add')
    glm_model = sm.GLM(y, X_with_const, family=sm.families.Binomial())
    glm_result = glm_model.fit()

    glm_coeff_df = glm_result.summary2().tables[1].reset_index()
    glm_coeff_df.rename(columns={'index': 'Feature', 'Coef.': 'Coefficient_Logistic'}, inplace=True)
    glm_coeff_df = glm_coeff_df[['Feature', 'Coefficient_Logistic']]

    logistic_regression_df = glm_coeff_df    
    logistic_regression_df['Rank_logistic'] = logistic_regression_df['Coefficient_Logistic'].rank(ascending=False).astype(int)
    logistic_regression_df = logistic_regression_df.sort_values(by=['Coefficient_Logistic'], ascending=False)

    
    # Task - Fit Ridge Regression for different values of alpha.
    X_train = X
    y_train = y
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_scaled = scaler.fit_transform(X_train)
    n_samples = X_train_scaled.shape[0]
    alphas = [0]  
    for alpha in alphas:
        # LogisticRegressionCV uses Cs (inverse of lambda), so we create an array of Cs.
        # We'll use 'l1' or 'elasticnet' penalty depending on alpha.
        # For alpha=0, only 'l2' penalty (ridge)
        if alpha == 0:
            penalty = 'l2'
            solver = 'lbfgs'  # solver that supports l2
            l1_ratio = None
        else:
            penalty = 'elasticnet'
            solver = 'saga'  # supports elasticnet
            l1_ratio = alpha
        
    clf = LogisticRegressionCV(
            Cs=98,  # number of lambda (inverse) values to test
            # Cs = 1,
            penalty=penalty,
            solver=solver,
            l1_ratios=[l1_ratio] if l1_ratio is not None else None,
            scoring='accuracy',
            cv=5,
            max_iter=10000,
            refit=True,
            random_state=42
        )
        
    clf.fit(X_train_scaled, y_train)

    # Get lambdas (inverse of Cs)
    lambdas = 1 / clf.Cs_
    # --- Initialize empty DataFrame
    coef_final = pd.DataFrame(columns=['rowname', 'coefficients', 'lambda', 'alpha', 'misclassification'])

    alpha = 0  # ridge only
    penalty = 'l2'
    solver = 'lbfgs'
    l1_ratio = None

    for lam in lambdas:
        # C = 1 / lam #### this was for when python generated lambdas were there
        C1 = 2 / (lam * n_samples) 
        C = C1 / 100

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
    # st.write(coef_final) ##was printing


    # Step 1: Sort coef_final by lambda ascending
    coef_final['lambda'] = pd.to_numeric(coef_final['lambda'], errors='coerce')
    coef_final_sorted = coef_final.sort_values(by='lambda')

    # Step 2: Create 5 buckets of 20 lambdas each
    unique_lambdas = sorted(coef_final_sorted['lambda'].unique())
    n_buckets = 5
    bucket_size = len(unique_lambdas) // n_buckets
    lambda_buckets = [unique_lambdas[i * bucket_size:(i + 1) * bucket_size] for i in range(n_buckets)]

    # Optional: handle any leftover lambdas in the last bucket
    if len(unique_lambdas) % n_buckets != 0:
        lambda_buckets[-1] += unique_lambdas[n_buckets * bucket_size:]

    # Prepare output
    Model_output = pd.DataFrame()
    Data_level_list = ['Overall']  # Update this if you have multiple labels
    found_match = False

    # Step 3: Iterate over buckets, break if match found
    for i, lambda_group in enumerate(lambda_buckets):
        if found_match:
            break

        # Filter coef_final for current lambda bucket
        coef_bucket = coef_final_sorted[coef_final_sorted['lambda'].isin(lambda_group)].copy()

        # --- Merge with glm coefficients
        joined_table = pd.merge(
            coef_bucket,
            glm_coeff_df,
            left_on='rowname',
            right_on='Feature',
            how='left'
        )

        # --- Calculate diffs
        joined_table['Coefficient_Logistic'] = pd.to_numeric(joined_table['Coefficient_Logistic'], errors='coerce')
        joined_table['diff'] = (joined_table['Coefficient_Logistic'] - joined_table['coefficients']).abs()
        joined_table['diff_per'] = joined_table['diff'] / joined_table['Coefficient_Logistic'].abs()

        # --- Fill NAs
        joined_table.fillna(0, inplace=True)

        # --- Group and summarize
        joined_table_group = (
            joined_table.groupby(['lambda', 'alpha', 'misclassification'], as_index=False)
            .agg(sum_diff=('diff', 'sum'))
        )

        joined_table = pd.merge(
            joined_table,
            joined_table_group,
            on=['lambda', 'alpha', 'misclassification'],
            how='left'
        )

        # --- Get best match in current bucket
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
        # st.write(filtered) ##was printing
        # If we found a good match, set flag and store results
        if not filtered.empty:
            found_match = True
            joined_table_new = filtered
            df1 = joined_table_new[['rowname', 'alpha', 'coefficients', 'misclassification']].copy()

            # Copy input to output 
            Overall_trans_model = df1.copy()

            # Append to Model_output
            Model_output = pd.concat([Model_output, Overall_trans_model], ignore_index=True)

            # Calculate odds-ratios and accuracy
            Model_output['Odds-Ratio'] = np.where(
                Model_output['coefficients'] >= 0,
                np.exp(Model_output['coefficients']),
                1 / np.exp((Model_output['coefficients'])) ############# checked ###
            )
            Model_output['Model Accuracy'] = 1 - Model_output['misclassification']

            # Output final model
            # st.write(Model_output) ##was printing



            Model_output.rename(columns={'rowname': 'Feature'}, inplace=True)
            # sum_summary1_copy = pd.merge(st.session_state.sum_summary1_copy,st.session_state.percent1_table,on ='Feature',how='left')
            if 'percent1_table' in st.session_state:
                st.session_state.sum_summary1_copy = pd.concat(
                    [st.session_state.sum_summary1_copy, st.session_state.percent1_table], 
                    ignore_index=True
                )

            Model_output = pd.merge(Model_output,st.session_state.sum_summary1_copy, on = 'Feature',how='left')
            # Model_output['Segment'] = 'Overall'
            # Model_output['Type'] = 'Overall'
            Model_output['Dependent Variable'] = y_column
            Model_output = Model_output.rename(columns={
                    'Feature': 'Independent Variables',
                    'coefficients':'Coefficient'
                    # 'Model Value': 'Coefficient'
                }).reset_index(drop=True)
            # Fill NA Actual % for Intercept or missing features with empty string or 0
            Model_output['Actual %'] = Model_output['Actual %'].fillna('')
            feature_description = st.session_state.required_features_df
            feature_description = feature_description.rename(columns={'Feature':'Independent Variables'}).reset_index(drop=True)
            
            if 'new_feature_description_table' in st.session_state:
                feature_description = pd.concat(
                    [feature_description, st.session_state.new_feature_description_table], 
                    ignore_index=True
                )
            # st.write(feature_description)


            st.session_state.feature_description = feature_description
            Model_output = pd.merge(Model_output,feature_description, on= 'Independent Variables', how='left')
            # Model_output['Model Accuracy'] = Model_output

            ################ overall % ###########################
            # Get the Overall % value for the dependent variable
            overall_actual_pct_row = st.session_state.sum_summary1_copy[
                st.session_state.sum_summary1_copy['Feature'] == st.session_state.y_column
            ]
            # Extract the Actual % value, use .values[0] safely
            overall_actual_pct = ''
            if not overall_actual_pct_row.empty and 'Actual %' in overall_actual_pct_row.columns:
                overall_actual_pct = overall_actual_pct_row['Actual %'].values[0]
            Model_output['Overall %'] = overall_actual_pct
            ################################################################################
        #     desired_order = [
        #     "Dependent Variable", "Segment", "Type", "Independent Variables", "Coefficient",
        #     "Odds-Ratio", "Actual %", "Overall %", "Model Accuracy", "Description"
        # ]
            desired_order = [
            "Dependent Variable", "Independent Variables", "Coefficient",
            "Odds-Ratio", "Actual %", "Overall %", "Model Accuracy", "Description"
        ]
            Model_output = Model_output[desired_order]  
            Model_output = Model_output.drop_duplicates()
            # st.write(Model_output)

            # st.write(f"<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> {top_label} selected model : </h1>", unsafe_allow_html=True)
            # st.dataframe(Model_output[['Independent Variables', 'Odds-Ratio']], hide_index=True, use_container_width=True) ### this is getting displayed
            st.session_state.Model_output_segment = Model_output
    #         excel_buffer = io.BytesIO()
    #         with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
    #             Model_output.to_excel(writer, index=False)
    #         excel_buffer.seek(0)
    #         # return excel_buffer
    #         # Create a download button
    #         st.download_button(
    #         label=f"📥 Download {top_label} Model Results (Excel)",
    #         data=excel_buffer,
    #         file_name=f"{top_label}_Model_Results.xlsx",
    #         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    #         key=f"download_button_{top_label}"
    # )
            





def category_level_model1():
    st.write("<h1 style='color: #0D3512; font-size: 35px; text-align:left; font-weight: normal;'> Segment Level Model Creation </h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color: #f0f9f0; padding: 12px; border-radius: 8px; box-shadow: 0 3px 8px rgba(0, 0, 0, 0.1);">
    <p style="font-size: 14px; color: #333333; font-family: Arial, sans-serif; line-height: 1.5;">
        Here, users can create a segment-level model based on selected segment columns and target variable.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # Preconditions
    if "data" not in st.session_state or "classifier_list" not in st.session_state or "Model_output" not in st.session_state or "x_columns" not in st.session_state:
        st.error("Required data or model results not found. Make sure you have run the overall model and have classifier_list, x_columns, Model_output in session state.")
        return

    data = st.session_state.data.copy()
    data.fillna(0, inplace=True)
    classifier_list = st.session_state.classifier_list

    # Step 1: Select Segment Columns
    st.write("<h3 style='color:#0D3512;'>1. Select Segment Columns:</h3>", unsafe_allow_html=True)
    selected_segments = st.multiselect("Select segment-defining column(s):", options=classifier_list, default=classifier_list)
    # selected_columns = st.multiselect("", options=st.session_state.classifier_list, default=st.session_state.classifier_list)

    if not selected_segments:
        st.stop()

    # Step 2: Select Target Column
    st.write("<h3 style='color:#0D3512;'>2. Select Target Column (Dependent Variable):</h3>", unsafe_allow_html=True)
    # Exclude nothing or maybe exclude the classifier columns? But keeping simple:
    y_column = st.selectbox("Select target column (y):", options=[col for col in data.columns])
    if not y_column:
        st.stop()

    # Step 3: Choose Feature Set Strategy
    st.write("<h3 style='color:#0D3512;'>3. Select Independent Variable Strategy:</h3>", unsafe_allow_html=True)
    # feature_strategy = st.radio(
    #     "Do you want to use all independent variables from the overall model, or select custom ones?",
    #     ["Use features from Overall Model", "Select custom features (_TOP / _TOP2)"]
    # )
    feature_strategy = st.radio(
    "Do you want to use all independent variables from the overall model, or select custom ones?",
    options=["", "Use features from Overall Model", "Select custom features (_TOP / _TOP2)"],
    index=0,
    format_func=lambda x: "Please select an option" if x == "" else x
)

    if feature_strategy == "":
        st.warning("Please select a feature selection strategy to continue.")
        st.stop()


    if feature_strategy == "Use features from Overall Model":
        x_cols_for_segments = st.session_state.x_columns.copy()
        if not x_cols_for_segments:
            st.error("No independent variables found in session_state.x_columns. Make sure the overall model has been run.")
            return
    else:
        columns = data.columns.tolist()
        top_columns = [c for c in columns if c.endswith("_TOP")]
        top2_columns = [c for c in columns if c.endswith("_TOP2")]

        st.markdown("<h4 style='margin-top: 20px;'>Select _TOP Features:</h4>", unsafe_allow_html=True)
        top_selected = st.multiselect("Select features from _TOP", options=top_columns, default=[])
        st.markdown("<h4 style='margin-top: 20px;'>Select _TOP2 Features:</h4>", unsafe_allow_html=True)
        top2_selected = st.multiselect("Select features from _TOP2", options=top2_columns, default=[])

        x_cols_for_segments = top_selected + top2_selected
        if not x_cols_for_segments:
            st.warning("You must select at least one independent variable. Please select from _TOP or _TOP2.")
            return

    # Store in session if needed
    st.session_state.y_column_cat = y_column
    st.session_state.x_columns_cat = x_cols_for_segments
    st.session_state.selected_segments_cat = selected_segments

    # Perform segment-level modeling using run_model_and_download
    # but we won't display all segment results; only allow user to view via dropdown

    all_segment_dfs = []
    for segment_col in selected_segments:
        unique_values = data[segment_col].unique()
        for cat in unique_values:
            subset = data[data[segment_col] == cat]
            if subset.shape[0] < 10:
                # skip small groups
                continue

            top_label = f"{segment_col}_{cat}"
            run_model_segment_and_download(
                x_columns=x_cols_for_segments,
                y_column=y_column,
                data=subset,
                top_label=top_label
            )
            # After run_model_and_download, st.session_state.Model_output holds the result for this segment
            seg_df = st.session_state.Model_output_segment.copy()
            seg_df["Segment"] = segment_col
            seg_df["Type"] = cat
            desired_order = [
            "Dependent Variable", "Segment", "Type", "Independent Variables", "Coefficient",
            "Odds-Ratio", "Actual %", "Overall %", "Model Accuracy", "Description"
        ]
          
            seg_df = seg_df[desired_order]  
            all_segment_dfs.append(seg_df)

    if not all_segment_dfs:
        st.warning("No valid segments (with enough data) to model.")
        return

    # Build segment-level concatenated DataFrame
    final_segment_df = pd.concat(all_segment_dfs, ignore_index=True)
    # st.write(final_segment_df)

    # Also get overall model df (from before), keep as is
    overall_df = st.session_state.Model_output_overall if "Model_output_overall" in st.session_state else st.session_state.Model_output
    # st.write(overall_df)
   
    combined = pd.concat([overall_df, final_segment_df], ignore_index=True)
    # Reindex so that missing columns are present (with blank or NaN)
    all_cols = list(set(overall_df.columns).union(set(final_segment_df.columns)))
    overall_df_uniform = overall_df.reindex(columns=all_cols, fill_value="")
    # final_segment_df_uniform = final_segment_df.reindex(columns=all_cols, fill_value="")
    # combined_uniform = pd.concat([overall_df_uniform, final_segment_df_uniform], ignore_index=True)

    # Store
    # st.session_state.segment_model_output = final_segment_df_uniform
    st.session_state.merged_overall_segment = combined

    # Dropdown to select which segment column to view (no immediate display of table)

    st.write("<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> View Results For Column: </h1>", unsafe_allow_html=True)
    selected_segment_for_view = st.selectbox("", options=[""] + selected_segments, key="segment_view_dropdown")
    if selected_segment_for_view != "":
        st.write(f"<h1 style='color: #0D3512; font-size: 25px; text-align:left; font-weight: normal;'> Result for '{selected_segment_for_view}' </h1>", unsafe_allow_html=True)
        st.dataframe(final_segment_df[final_segment_df['Segment'] == selected_segment_for_view], use_container_width=True)


    # Download buttons
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        buf1 = io.BytesIO()
        with pd.ExcelWriter(buf1, engine="xlsxwriter") as writer:
            final_segment_df.to_excel(writer, index=False)
        buf1.seek(0)
        st.download_button(
            label="📥 Download Segment Level Model Results (Excel)",
            data=buf1,
            file_name="Segment_Model_Results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    with col2:
        buf2 = io.BytesIO()
        with pd.ExcelWriter(buf2, engine="xlsxwriter") as writer:
            combined.to_excel(writer, index=False)
        buf2.seek(0)
        st.download_button(
            label="📥 Download Combined (Overall + Segment) Results (Excel)",
            data=buf2,
            file_name="Combined_Model_Results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

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


