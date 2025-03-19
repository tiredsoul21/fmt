""" Prepares the data for the model """


import pandas as pd


def load_data(dataFile: str, prep: bool = False):
    """ Loads the data from the file """
    data = pd.read_csv(dataFile)
    if prep:
        data = prepare_data(data)
    return data

def prepare_data(data: pd.DataFrame):
    """ Preprocesses the data """
    # Drop columns with all NaN values -- if any
    data = data.dropna(axis=1, how="all")

    # Drop static columns -- if any
    for column in data.columns:
        if len(data[column].unique()) == 1:
            data.drop(column, axis=1, inplace=True)

    # Convert 'semester' column to categorical dtype with custom order
    order = ['F13','Sp15','F15','Sp16','F16','Sp17','F17','Sp18','F18','Sp19','F19']
    data["semester"] = pd.Categorical(data["semester"], categories=order, ordered=True)

    # Drop concent, time (currently single value / little variance)
    data.drop(['consent_pre'], axis=1, inplace=True)
    # Time is automatically dropped because it has only one unique value

    # Drop plan of ('unknown', 11, 22) - Need to determine if "plan" is a distinguishing feature
    data = data[~data["plan"].isin(['unknown', '11', '22'])]
    order = ['1', '2', '3']
    data["plan"] = pd.Categorical(data["plan"], categories=order, ordered=True )

    # Drop blank values
    # data = data[~data["reading_ability"].isna()]
    # data = data[~data["writing_ability"].isna()]
    # data = data[~data["english_first_language"].isna()]

    # english_first_language left Mapped to [1(yes) -> 1, 2(no) -> 0, 3(choose not to answer) -> 2]
    data["english_first_language"] = data["english_first_language"].map({1:1, 2:0, 3:2})

    # cast type on prior_bio as category
    data["prior_bio"] = pd.Categorical(data["prior_bio"], ordered=True)

    # age cast empty to 0 and 'Rather not answer' to 0 -- cast to int
    data["age"] = data["age"].fillna(-1)
    data["age"] = data["age"].replace("Rather not answer", -1)
    data["age"] = data["age"].astype("int64")

    # race_complete cast empty to 'No answer'- move 'race_complete' column to 'race'
    data["race"] = data["race_complete"].fillna("Rather not answer")
    data.drop(["race_complete"], axis=1, inplace=True)
    # Make categorical
    order = ['Asian', 'Black/African American', 'White', 'American Indian/Alaska Native',
                 'Hispanic of any race', 'Native Hawaiian/Other Pacific Island', 'No answer']
    data["race"] = pd.Categorical(data["race"], categories=order, ordered=True)
    data["race_e"] = data["race"].cat.codes

    # 'gender_complete' column to 'gender' (male - 1, non-male - 0)
    data["gender"] = data["gender_complete"]
    data.drop(["gender_complete"], axis=1, inplace=True)

    # 'sbu_admit' order ('New','Transfer', 'Other') - cast to category
    data["sbu_admit"] = data["sbu_admit_type_category"]
    data.drop(["sbu_admit_type_category"], axis=1, inplace=True)
    order = ['New', 'Transfer', 'Other']
    data["sbu_admit"] = pd.Categorical(data["sbu_admit"], categories=order, ordered=True)
    data["sbu_admit_e"] = data["sbu_admit"].cat.codes

    # citizenship - cast to category
    data["citizenship"] = pd.Categorical(data["citizenship"], ordered=True)
    data["citizenship_e"] = data["citizenship"].cat.codes
    # 'country_citizenship' - cast to category rename 'Viet m' to 'Vietnam'
    data["country_citizenship"] = data["country_citizenship"].replace("Viet m", "Vietnam")
    data["country_citizenship"] = pd.Categorical(data["country_citizenship"], ordered=True)
    data["country_citizenship_e"] = data["country_citizenship"].cat.codes

    # 'family_income' - cast to category
    data["family_income"] = data["family_income"].fillna("No Answer")
    # Move 'Family receives payments from a NY county Department of Social Services' > 'State Assistance'
    data["family_income"] = data["family_income"].replace('Family receives payments from a NY county Department of Social Services', 'State Assistance')
    order = ["No Answer", "State Assistance", "$0-14,999", "$15,000-29,999",
                 "$30,000-44,999", "$45,000-64,999", "$65,000-84,999", "$85,000-104,999",
                 "$105,000+", "105,000-124,999", "$125,000-144,999","$145,000-164,999",
                 "$165,000-184,999", "$185,000+"]
    data["family_income"] = pd.Categorical(data["family_income"], categories=order, ordered=True)
    data["family_income_e"] = data["family_income"].cat.codes

    # 'first_generation' - replace empty with 'U' - cast to category (yes - 1, no - 0, unknown - 2)
    data["first_generation"] = data["first_generation"].fillna("U")
    order = ['N', 'Y', 'U']
    data["first_generation"] = pd.Categorical(data["first_generation"], categories=order, ordered=True)
    data["first_generation_e"] = data["first_generation"].cat.codes

    # high_school_gpa_band to 'hs_gpa_band' - cast to category
    data["hs_gpa_band"] = data["high_school_gpa_band"].fillna("Unknown")
    data.drop(["high_school_gpa_band"], axis=1, inplace=True)
    order = ["Unknown", "Below 75", "75 - 79.9", "80 - 84.9", "85 - 89.9", "90 - 94.9", "95 - 100"]
    data["hs_gpa_band"] = pd.Categorical(data["hs_gpa_band"], categories=order, ordered=True)
    data['hs_gpa_band_e'] = data['hs_gpa_band'].cat.codes

    # high_school_gpa to 'hs_gpa' - replace empty with -1
    data["hs_gpa"] = data["high_school_gpa"].fillna(-1)
    data.drop(["high_school_gpa"], axis=1, inplace=True)

    # cum_college_gpa replace empty with -1
    data["cum_college_gpa"] = data["cum_college_gpa"].fillna(-1)

    # PlacementMathScore with math_placement - cast to category
    data["math_placement"] = data["PlacementMathScore"].fillna(-1)
    data.drop(["PlacementMathScore"], axis=1, inplace=True)
    order = [-1, 1, 2, 2.5, 3, 4, 5, 6, 7, 8, 9]
    data["math_placement"] = pd.Categorical(data["math_placement"], categories=order, ordered=True)

    # PlacementWritingScore with writing_placement - cast to category
    data["writing_placement"] = data["PlacementWritingScore"].fillna(-1)
    data.drop(["PlacementWritingScore"], axis=1, inplace=True)
    order = [-1, 1, 1.5, 2, 3, 4]
    data["writing_placement"] = pd.Categorical(data["writing_placement"], categories=order, ordered=True)

    # SAT1600Score with sat_1600_score - replace empty with -1
    data["sat_1600_score"] = data["SAT1600Score"].fillna(-1)
    data.drop(["SAT1600Score"], axis=1, inplace=True)

    # SATMathScore with sat_math_score - replace empty with -1
    data["sat_math_score"] = data["SATMathScore"].fillna(-1)
    data.drop(["SATMathScore"], axis=1, inplace=True)

    # SATVerbalScore with sat_verbal_score - replace empty with -1
    data["sat_verbal_score"] = data["SATVerbalScore"].fillna(-1)
    data.drop(["SATVerbalScore"], axis=1, inplace=True)

    # SATWritingScore with sat_writing_score - replace empty with -1
    data["sat_writing_score"] = data["SATWritingScore"].fillna(-1)
    data.drop(["SATWritingScore"], axis=1, inplace=True)

    # SATEssayScore with sat_essay_score - replace empty with -1
    data["sat_essay_score"] = data["SATEssayScore"].fillna(-1)
    data.drop(["SATEssayScore"], axis=1, inplace=True)

    # SATCompScore with sat_comp_score - replace empty with -1
    data["sat_comp_score"] = data["SATCompScore"].fillna(-1)
    data.drop(["SATCompScore"], axis=1, inplace=True)

    return data
