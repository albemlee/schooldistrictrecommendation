import zipfile
import pandas as pd
import chardet
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, Normalizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def load_data(data_type, year='2010', location='data'):
    """
    Return pandas dataframe of raw data

    param string data_type: Type of data being read
        options: 'finance', 'dropout', 'universe', 'membership'
    param string location: Folder where data is located
    """
    file_in = location + '/' + data_type + '_fiscal' + year + '.txt.zip'
    archive = zipfile.ZipFile(file_in, 'r')
    data_table = None
    while data_table is None:
        try:
            encoding_type = find_encoding(file_in)
            data_table = pd.read_table(archive.open(data_type + '_fiscal' + year + '.txt'), dtype='str', delimiter='\t', encoding=encoding_type)
        except:
            pass

    return data_table

def find_encoding(file, is_zipped=True):
    """
    Return encoding of file

    param string file: Location of file to read (include folder name)
    param bool is_zipped: True if file is zipped, otherwise False
    """
    if is_zipped:
        file_name = file.split('/')[1]
        with zipfile.ZipFile(file, 'r') as archive:
            with archive.open(file_name[:-4]) as raw_data:
                lines = raw_data.readlines()
                sample_lines = np.random.choice(lines, 1000)
                sample_text = b'\n'.join(sample_lines)
                encoding_prediction = chardet.detect(sample_text)
                raw_data.close()
            archive.close()
        encoding_type = encoding_prediction['encoding']
    else:
        with open(file, 'rb') as raw_data:
            lines = raw_data.readlines()
            sample_lines = np.random.choice(lines, 1000)
            sample_text = b'\n'.join(sample_lines)
            encoding_prediction = chardet.detect(sample_text)
            raw_data.close()
        encoding_type = encoding_prediction['encoding']

    return encoding_type

def aggregate_school_data(school_columns, year='2010'):
    """
    Return dataframe of numerical columns within school_universe grouped by (sum) LEAID

    param dataframe school_columns: list of columns in the school universe file
    """
    school_universe_data = load_data('school_universe', year=year)
    school_universe_data = school_universe_data[school_columns]
    school_universe_data = encode_missing_values(school_universe_data)
    for column in school_columns:
        if column != 'LEAID':
            school_universe_data[column] = pd.to_numeric(school_universe_data[column])
    school_universe_data = school_universe_data.groupby('LEAID').sum(skipna=True, min_count=1).reset_index()

    return school_universe_data

def load_columns(column_type):
    """
    Return list of columns within column type

    param string column_type: Type of column
        options: 'categorical', 'dropout', 'finance', 'identifier', 'numerical', 'ordinal', 'remove', 'school', 'universe'
    """
    file_name = 'data/columns/{0}_columns.txt'.format(column_type)
    with open(file_name, 'r') as column_file:
        column_text = column_file.read()
        column_text = column_text.replace('\n', '')
        column_text = column_text.replace(' ', '')
        column_list = column_text.split(',')
        column_file.close()

    return column_list

def merge_data(*args):
    """
    Return pandas dataframe of merged dataframe

    param dataframe *args: dataframes or raw data to be joined (typically finance, dropout, and universe)
    """
    data_frames = list(args)
    merged_data = reduce(lambda d1, d2: pd.merge(d1, d2, how='outer', left_on='LEAID', right_on='LEAID'), data_frames)

    return merged_data

def wrangle_data(merged_data, directory_data):
    """
    Return pandas dataframe of wrangled dataframe

    param dataframe merged_data: dataframe of merged data
    param dataframe directory_data: dataframe with all school districts in 2015
    """
    wrangled_data = merged_data.copy()

    # remove flag columns
    wrangled_data = remove_flag_columns(wrangled_data)

    # Identify and replace missing and low quality values with NA
    # Replace "non applicable" values with 0
    wrangled_data = encode_missing_values(wrangled_data)

    # Create column to represent whether school district is still operational in five years
    wrangled_data = calc_five_years_operational(wrangled_data, directory_data)

    return wrangled_data

def remove_flag_columns(merged_data):
    """
    Return dataframe with flag columns removed

    param dataframe merged_data: merged_data
    """
    flag_columns = [column for column in merged_data.columns.tolist() if column.split('_')[0] == 'FL']
    merged_data.drop(columns=flag_columns, inplace=True)
    return merged_data

def encode_missing_values(raw_data):
    """
    Return dataframe with missing and non applicable values re-encoded as NA and 0

    param DataFrame raw_data: dataframe without re-encoded missing and non applicable values
    """
    reencoded_data = raw_data.copy()
    for column in reencoded_data.columns.tolist():
        missing_values = reencoded_data[column].apply(lambda x: x in ['-1', '-1.0', '-1.00', 'M'] or pd.isnull(x))
        non_applicable_values = reencoded_data[column].apply(lambda x: x in ['-2', '-2.0', '-2.00', 'N'])
        low_quality_values = reencoded_data[column].apply(lambda x: x in ['-9', '-9.0', '-9.00', '-3', '-3.0', '-3.00', '-4', '-4.0', '-4.00'])

        reencoded_data.loc[missing_values, column] = np.nan
        reencoded_data.loc[non_applicable_values, column] = '0'
        reencoded_data.loc[low_quality_values, column] = np.nan

    return reencoded_data

def calc_five_years_operational(merged_data, directory_data):
    """
    Return pandas dataframe of wrangled dataframe

    param dataframe merged_data: dataframe of merged data
    param dataframe directory_data: dataframe with all school districts in 2015
    """
    def condition1_generate(x):
        """
        Return boolean of whether school district exists in 5 years

        param string x: LEAID of school district to check
        """
        condition1 = x in directory_data['LEAID'].values
        if condition1:
            condition1 = directory_data.loc[directory_data['LEAID']==x, 'SY_STATUS'].values != '2'
        return condition1

    condition1 = merged_data['LEAID'].apply(condition1_generate)
    condition2 = merged_data['BOUND09'].apply(lambda x: x != '2')
    merged_data['exist_five_years'] = condition1 & condition2
    return merged_data

def wrangle_data_ii(wrangled_data):
    """
    Return pandas dataframe of wrangled dataframe with columns to keep renamed

    param dataframe wrangled_data: dataframe of wrangled_data data (before column renaming)
    """
    # create dictionaries mapping old column names to new column names
    few_missing_values_columns = {'STNAME': 'state_name',
                                  'GSLO09': 'lowest_grade',
                                  'GSHI09': 'highest_grade',
                                  'MEMBERSCH': 'total_students',
                                  'METMIC09': 'metro_micro',
                                  'BIEA09': 'bureau_indian_education',
                                  'AGCHRT09': 'charter_status',
                                  'SCH09': 'total_schools',
                                  'ELL09': 'english_language_learners'}

    normal_missing_value_columns = {'T02': 'local_revenue_parent_government_contributions',
                                    'T06': 'local_revenue_property_tax',
                                    'T09': 'local_revenue_sales_tax',
                                    'T15': 'local_revenue_utilities_tax',
                                    'T40': 'local_revenue_income_tax',
                                    'T99': 'local_revenue_other_tax',
                                    'ELMTCH09': 'teachers_elementary',
                                    'SECTCH09': 'teachers_secondary',
                                    'UGTCH09': 'teachers_ungraded',
                                    'TOTTCH09': 'teachers_total',
                                    'LEAADM09': 'administrators_district',
                                    'SCHADM09': 'administrators_school',
                                    'STUSUP09': 'staff_student_support'}

    many_missing_value_columns = {'TOTALREV': 'total_revenue',
                                  'TFEDREV': 'total_federal_revenue',
                                  'C14': 'federal_revenue_state_title_i',
                                  'C15': 'federal_revenue_ideas',
                                  'C16': 'federal_revenue_math_science_quality',
                                  'C17': 'federal_revenue_drug_free',
                                  'C19': 'federal_revenue_vocational_tech_training',
                                  'B10': 'federal_revenue_impact_aid',
                                  'B12': 'federal_revenue_indian_education',
                                  'B13': 'federal_revenue_other',
                                  'TSTREV': 'total_state_revenue',
                                  'C01': 'state_revenue_general_formula_assistance',
                                  'C04': 'state_revenue_staff_improvement',
                                  'C05': 'state_revenue_special_education',
                                  'C06': 'state_revenue_compensatory_basic_training',
                                  'C07': 'state_revenue_bilingual_education',
                                  'C08': 'state_revenue_gifted_talented',
                                  'C09': 'state_revenue_vocational_education',
                                  'C10': 'state_revenue_school_lunch',
                                  'C11': 'state_revenue_capital_outlay_debt_services',
                                  'C12': 'state_revenue_transportation',
                                  'C13': 'state_revenue_other',
                                  'C35': 'state_revenue_nonspecified',
                                  'C38': 'state_revenue_employee_benefits',
                                  'C39': 'state_revenue_not_employee_benefits',
                                  'TLOCREV': 'total_local_revenue',
                                  'D11': 'local_revenue_other_school_systems',
                                  'D23': 'local_revenue_cities_counties',
                                  'A07': 'local_revenue_tuition_fee_pupils_parents',
                                  'A08': 'local_revenue_transportation_fee_pupil_parents',
                                  'A09': 'local_revenue_school_lunch',
                                  'A11': 'local_revenue_textbook_sale_rental',
                                  'A13': 'local_revenue_district_activity_receipt',
                                  'A15': 'local_revenue_student_fee_nonspecified',
                                  'A20': 'local_revenue_other_sales_services',
                                  'A40': 'local_revenue_rent_royalties',
                                  'U11': 'local_revenue_property_sales',
                                  'U22': 'local_revenue_interest_earnings',
                                  'U30': 'local_revenue_fines_forfeits',
                                  'U50': 'local_revenue_private_contributions',
                                  'U97': 'local_revenue_miscellaneous',
                                  'C24': 'local_revenue_NCES',
                                  'TOTALEXP': 'total_expenditure',
                                  'TCURELSC': 'total_expenditure_elementary_secondary',
                                  'TCURINST': 'total_expenditure_instruction',
                                  'E13': 'total_expenditure_instruction_public',
                                  'V91': 'expenditure_private_school',
                                  'V92': 'expenditure_charter_school',
                                  'TCURSSVC': 'total_expenditure_support_services',
                                  'E17': 'expenditure_support_services_pupils',
                                  'E07': 'expenditure_support_services_instructional_staff',
                                  'E08': 'expenditure_support_services_general_administration',
                                  'E09': 'expenditure_support_services_school_administration',
                                  'V40': 'expenditure_support_services_maintenance',
                                  'V45': 'expenditure_support_services_transportation',
                                  'V90': 'expenditure_support_services_business',
                                  'V85': 'expenditure_support_services_nonspecified',
                                  'TCUROTH': 'total_expenditure_other_elementary_secondary',
                                  'E11': 'expenditure_food_service',
                                  'V60': 'expenditure_enterprise',
                                  'V65': 'expenditure_other_elementary_secondary',
                                  'TNONELSE': 'total_expenditure_non_elementary_secondary',
                                  'V70': 'expenditure_non_elementary_secondary_community_service',
                                  'V75': 'expenditure_non_elementary_secondary_adult_education',
                                  'V80': 'expenditure_non_elementary_secondary_other',
                                  'TCAPOUT': 'total_expenditure_capital_outlay',
                                  'F12': 'expenditure_capital_outlay_construction',
                                  'G15': 'expenditure_capital_outlay_land_existing_structures',
                                  'K09': 'expenditure_capital_outlay_instruction_equipment',
                                  'K10': 'expenditure_capital_outlay_other_equipment',
                                  'K11': 'expenditure_capital_outlay_nonspecified',
                                  'L12': 'payments_state_government',
                                  'M12': 'payments_local_government',
                                  'Q11': 'payments_other_school_systems',
                                  'I86': 'interest_on_debt',
                                  'Z32': 'total_salaries',
                                  'Z33': 'salaries_instruction',
                                  'Z35': 'salaries_regular_education',
                                  'Z36': 'salaries_special_education',
                                  'Z37': 'salaries_vocationall_education',
                                  'Z38': 'salaries_other_education',
                                  'V11': 'salaries_support_services_pupils',
                                  'V13': 'salaries_support_services_instructional_staff',
                                  'V15': 'salaries_support_services_general_administration',
                                  'V17': 'salaries_support_services_school_administration',
                                  'V21': 'salaries_support_services_maintenance',
                                  'V23': 'salaries_support_transportation',
                                  'V37': 'salaries_support_services_business',
                                  'V29': 'salaries_food_service',
                                  'V10': 'employee_benefits_instruction',
                                  'V12': 'employee_benefits_support_services_pupil',
                                  'V14': 'employee_benefits_support_services_instructional_staff',
                                  'V16': 'employee_benefits_support_services_general_administration',
                                  'V18': 'employee_benefits_support_services_school_administration',
                                  'V22': 'employee_benefits_support_services_maintenance',
                                  'V24': 'employee_benefits_support_transportation',
                                  'V38': 'employee_benefits_support_services_business',
                                  'V30': 'employee_benefits_food_service',
                                  'V32': 'employee_benefits_enterprise',
                                  'V93': 'textbooks',
                                  '_19H': 'long_term_debt_outstanding_beginning_fiscal_year',
                                  '_21F': 'long_term_debt_issued_during_fiscal_year',
                                  '_31F': 'long_term_debt_retired_during_fiscal_year',
                                  '_41F': 'long_term_debt_outstanding_end_fiscal_year',
                                  '_61V': 'short_term_debt_outstanding_beginning_fiscal_year',
                                  '_66V': 'short_term_debt_outstanding_end_fiscal_year',
                                  'W01': 'assets_sinking_fund',
                                  'W31': 'assets_bond_fund',
                                  'W61': 'assets_other_funds',
                                  'HR1': 'ARRA_revenue',
                                  'HE1': 'ARRA_current_expenditures',
                                  'HE2': 'ARRA_capital_outlay_expenditures',
                                  'LIBSPE09': 'librarian_media_specialists',
                                  'AM09': 'american_indian_alaskan_native_students',
                                  'AMALM09': 'american_indian_alaskan_native_male_students',
                                  'AMALF09': 'american_indian_alaskan_native_female_students',
                                  'ASIAN09': 'asian_hawaiian_native_pacific_islander_students',
                                  'ASALM09': 'asian_hawaiian_native_pacific_islander_male_students',
                                  'ASALF09': 'asian_hawaiian_native_pacific_islander_female_students',
                                  'HISP09': 'hispanic_students',
                                  'HIALM09': 'hispanic_male_students',
                                  'HIALF09': 'hispanic_female_students',
                                  'BLACK09': 'black_non_hispanic_students',
                                  'BLALM09': 'black_non_hispanic_male_students',
                                  'BLALF09': 'black_non_hispanic_female_students',
                                  'WHITE09': 'white_students',
                                  'WHALM09': 'white_male_students',
                                  'WHALF09': 'white_female_students',
                                  'PACIFIC09': 'hawaiian_native_pacific_islander_students',
                                  'HPALM09': 'hawaiian_native_pacific_islander_male_students',
                                  'HPALF09': 'hawaiian_native_pacific_islander_female_students',
                                  'TR09': 'mixed_race_students',
                                  'TRALM09': 'mixed_race_male_students',
                                  'TRALF09': 'mixed_race_female_students'}

    # create list of columns to keep
    columns_to_keep = ['LEAID', 'NAME', 'exist_five_years']
    for column in few_missing_values_columns.keys():
        columns_to_keep.append(column)
    for column in normal_missing_value_columns.keys():
        columns_to_keep.append(column)
    for column in many_missing_value_columns.keys():
        columns_to_keep.append(column)

    # remove unnecessary columns
    wrangled_data_ii = wrangled_data[columns_to_keep].copy()

    # rename columns
    wrangled_data_ii.rename(columns=few_missing_values_columns, inplace=True)
    wrangled_data_ii.rename(columns=normal_missing_value_columns, inplace=True)
    wrangled_data_ii.rename(columns=many_missing_value_columns, inplace=True)

    return wrangled_data_ii

def prep_columns_cluster(wrangled_data_ii):
    """
    Return pandas dataframe of data prepped for clustering

    param dataframe wrangled_data_ii: dataframe of wrangled dataframe (after rename/reduce columns)
    """
    
    prepped_cluster_data = wrangled_data_ii.copy()
    
    # Identify column types
    identifying_columns = ['LEAID', 'NAME']
    prediction_columns = ['exist_five_years']
    categorical_columns = ['state_name', 'lowest_grade', 'highest_grade', 'metro_micro', 'charter_status']
    boolean_columns = ['bureau_indian_education']
    numerical_columns = []
    keep_features = ['total_students',
                     'total_schools',
                     'teachers_total',
                     'total_revenue',
                     'total_federal_revenue',
                     'total_state_revenue',
                     'total_local_revenue',
                     'total_expenditure',
                     'total_salaries',
                     'white_students',
                     'lowest_grade',
                     'highest_grade',
                     'metro_micro',
                     'charter_status',
                     'state_name',
                     'LEAID',
                     'NAME',
                     'exist_five_years']
    
    prepped_cluster_data = prepped_cluster_data[keep_features]
    
    # Identify numerical columns
    for column in prepped_cluster_data.columns:
        if column in identifying_columns or column in categorical_columns or column in boolean_columns or column in prediction_columns:
            pass
        elif len(prepped_cluster_data[column].unique()) > 100:
            numerical_columns.append(column)
        else:
            categorical_columns.append(column)
            
    # One-hot encode categorical variables
    prepped_cluster_data = pd.get_dummies(prepped_cluster_data, prefix_sep='_', columns=categorical_columns, drop_first=True)
    
    # Convert numerical columns to float
    prepped_cluster_data[numerical_columns] = prepped_cluster_data[numerical_columns].astype(float)

    # Remove identifying
    prepped_cluster_data.drop(columns=identifying_columns, inplace=True)
    y = prepped_cluster_data[prediction_columns]
    prepped_cluster_data.drop(columns=prediction_columns, inplace=True)
    X = prepped_cluster_data
    
    # Split into training, development, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
    X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=21)
    
    return X_train, y_train, X_dev, y_dev, X_test, y_test

def find_clusters(wrangled_data_ii, X, y):
    """
    Return pandas dataframe with cluster and label of all school districts

    param dataframe wrangled_data_ii: dataframe of wrangled dataframe (after rename/reduce columns)
    param dataframe X: dataframe of features
    param dataframe y: dataframe of labels
    """

    # Define pipeline for clustering
    five_cluster_pipeline = Pipeline([
        ('imp', Imputer(strategy='median')),
        ('normalizer', Normalizer()),
        ('cluster', KMeans(n_clusters=5, random_state=22))
    ])

    # Fit pipeline to prepped data
    five_cluster_pipeline.fit(X)

    # Create DataFrame with labels and clusters
    clusters = five_cluster_pipeline.named_steps['cluster'].labels_
    labels = y.apply(lambda x: x=='False').values.ravel()

    cluster_frame = pd.DataFrame({'LEAID': wrangled_data_ii.loc[X.index, 'LEAID'],
                                  'NAME': wrangled_data_ii.loc[X.index, 'NAME'],
                                  'cluster': clusters,
                                  'label': labels})

    return cluster_frame

def prep_columns_classification(wrangled_data_ii, feature_set=None):
    """
    Return X_train, X_dev, X_test, y_train, y_dev, y_test: predictors (X) and label (y) of train, dev, and test sets (0.8, 0.1, 0.1 split)

    param dataframe wrangled_data_ii: dataframe of wrangled dataframe (after rename/reduce columns)
    param list feature_set: list of features to include
    """

    prepped_classification_data = wrangled_data_ii.copy()

    # Specify features to include in model
    if feature_set:
        pass
    else:
        feature_set = wrangled_data_ii.drop(columns=['NAME', 'LEAID', 'exist_five_years']).columns
    X = wrangled_data_ii[feature_set].copy()

    # Identify column types
    identifying_columns = ['NAME', 'LEAID']
    prediction_columns = ['exist_five_years']
    categorical_columns = ['lowest_grade', 'highest_grade', 'charter_status']
    boolean_columns = ['bureau_indian_education']
    numerical_columns = []

    # identify numerical columns
    for column in X.columns:
        if column in identifying_columns or column in categorical_columns or column in boolean_columns or column in prediction_columns:
            pass
        elif len(X[column].unique()) > 100:
            numerical_columns.append(column)
        else:
            categorical_columns.append(column)

    X[numerical_columns] = X[numerical_columns].astype(float)

    # one hot encode categorical variables
    X = pd.get_dummies(X, prefix_sep='_', columns=categorical_columns, drop_first=True)

    # Split into train and test sets
    y = wrangled_data_ii[prediction_columns].apply(lambda x: x=='False')
    y = y.values.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
    X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=21)

    return X_train, y_train, X_dev, y_dev, X_test, y_test

def classify_school_districts(X_train, y_train, X_dev, y_dev, X_test, y_test):
    """
    Return fitted classification pipeline
    Return train_score, development_score, test_score: recall scores on train, development, and test sets

    parameters X_train, y_train, X_dev, y_dev, X_test, y_test: predictors and labels of train, development, and test sets
    """
    clf_pipeline = Pipeline([
        ('imp', Imputer(missing_values='NaN', strategy='median', axis=0)),
        ('scaler', MinMaxScaler()),
        ('clf', XGBClassifier(scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train)))
    ])

    clf_pipeline.fit(X_train, y_train)

    train_predictions = clf_pipeline.predict(X_train)
    dev_predictions = clf_pipeline.predict(X_dev)
    test_predictions = clf_pipeline.predict(X_test)

    train_score = recall_score(y_train, train_predictions)
    development_score = recall_score(y_train, train_predictions)
    test_score = recall_score(y_train, train_predictions)

    return clf_pipeline, train_score, development_score, test_score
