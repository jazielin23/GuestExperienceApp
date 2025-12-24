
col_descs = [
# "responseid",
# "resid",
# "in which country do you live",
# "country_new",
# "county",
# "usregion",
# "other (please specify)",
# "please tell us your age",
# "please tell us the ages of each person",
# "longest you waited for the photopass photos",
# "how many people",
# "keymkts",
# "mapres",
# "who is your cell phone carrier",
# "where did you stay during your most recent visit to the disneyland",
# "resort hotel did you stay",
# "where was this non-disney hotel located",
# "what type of ticket did you primarily use to visit the disneyland",
# "what is your gender",
# "what is your total annual household income",
# "interntl",
# "which of the following ways did you get your fastpass experiences" 
'RESPONSEID', 'RESPID', 'STATUS', 'INTERVIEW_START', 'INTERVIEW_END',
    'PROJECT_ID', 'UNIQUE_ID', 'INSIGHT_ID', 'FACNAME', 'FACNAME2', 'CITY', 'ZIP',
    'COUNTRY', 'RENUM', 'ROOMTYPE', 'PKGNAME', 'GRPCODE', 'GRPNAME', 'CHANNEL',
    'CHANNUM', 'CODE', 'UNIQUE_KEY', 'SWID', 'LANGUAGE', 'DEPDATE', 'DVC',
    'PKGCODE', 'FAC_ID', 'SOURCE', 'RESORT', 'ARRDATE', 'CHKDATE', 'COUNTRYCHECK',
    'ZIPCODE', 'DBOPTIN', 'LDTI', 'SAGE', 'VERIFICATION', 'TRIPREAS', 'TRIPREAS_8_OTHER']

# Define the mapping for new column values based on unique options
mapping_conditions = {
    'Yes': 1,
    'Not Sure': 0,
    'No': 0,
    'Small Impact': 1,
    'No Impact': 0,
    'Large Impact': 1,
    'Excellent': 1,
    'Very Good': 0,
    'Good': 0,
    'Just Okay': 0,
    'Poor': 0,
    'No Answer': 0,
    99: 0,
    98: 0
}