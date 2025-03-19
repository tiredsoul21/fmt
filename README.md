## FMT
### Setup
```
# Code snippets
sudo apt-get install python3-venv
python3 -m venv venv
source venv/bin/activate
pip3 install datasets==2.17.1         # For dataset managment
```

### Definitions
* CANS - Child and Adolescent Needs and Strengths
* CINS - Children in Need of Support

### Prep - Management
* Migrate the data marked SBU_BIO201_CINS_CANS.xlsx to equity/data/.
* Export SBU_BIO201_CINS_CANS.xlsx to SBU_BIO201_CINS_CANS.csv
* Description of data isolated from data
### Prep - Code / Conditioning
* Semester is categorical with order:

    | F13 | Sp15 | F15 | Sp16 | F16 | Sp17 | F17 | Sp18 | F18 | Sp19 | F19 |
    |-----|------|-----|------|-----|------|-----|------|-----|------|-----|
    |  0  |   1  |  2  |   3  |  4  |   5  |  6  |   7  |  8  |   9  |  10 |
    
* Time is dropped (single value)
* Consent is dropped (~ single value)
* Plan:

    | 1 | 2 | 3 | Other      |
    |---|---|---|------------|
    | 0 | 1 | 2 |Dropped (17)|
* reading_ability - Dropped empty (31)
* writing_ability - Dropped empty (59)
* Race:

    | No answer | Asian | Black / African American | White  | American Indian/Alaska Native |  Hispanic of any race | Native Hawaiian/Other Pacific Island |
    |-----------|-------|--------------------------|--------|-------------------------------|-----------------------|--------------------------------------|
    |      0    |   1   |             2            |    3   |              4                |            5          |                  6                   |
* SBU Admit

    | New | Transfer | Other |
    |-----|----------|-------|
    | 0   |    1     |   2   |

* citizenship
    |  Alien Permanent | Alien Temporary | Native | Naturalized | Undocumented Alien |
    |------------------|-----------------|--------|-------------|--------------------|
    |        0         |       1         |    2   |      3      |         4          |

* country_citizenship - categorical (many)
* family_income:
    | No Answer | State Assistance | $0-14k | $15-30k | $30-45k | $45-65k | $65-85k | $85-105k | $105k+ | $105-125k | $125-145k | $145-165k | $165-185k | $185k+ |
    |-----------|------------------|--------|---------|---------|---------|---------|----------|--------|-----------|-----------|-----------|-----------|--------|
    |     0     |         1        |    2   |    3    |    4    |    5    |    6    |    7     |    8   |     9     |     10    |     11    |     12    |   13   |

* first_generation:
    | N | Y | U |
    |---|---|---|
    | 0 | 1 | 2 |
* high_school_gpa_band -> hs_gpa_band - empty filled 'Unknown'
    | Unknown | Below 75 | 75 - 79.9 | 80 - 84.9 | 85 - 89.9 | 90 - 94.9 | 95 - 100 |
    |---------|----------|-----------|-----------|-----------|-----------|----------|
    |    0    |     1    |      2    |      3    |      4    |     5     |     6    |
* high_school_gpa -> hs_gpa - empty filled with -1
* cum_colllege_gpa - empty filled with -1
* PlacementMathScore -> math_placement (empty -> -1):
    | -1 | 1 | 2 | 2.5 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
    |----|---|---|-----|---|---|---|---|---|---|---|
    |  0 | 1 | 2 |  3  | 4 | 5 | 6 | 7 | 8 | 9 | 10|
* PlacementWritingScore -> writing_placement (empty -> -1):
    | -1 | 1 | 1.5 | 2 | 3 | 4 |
    |----|---|-----|---|---|---|
    |  0 | 1 |  2  | 3 | 4 | 5 |
* english_first_language -> empty -> 'not answered'
    | No | Yes | I prefer not to answer |
    |----|-----|------------------------|
    | 0  |  1  |            2           |
* SAT1600Score -> sat_1600_score (empty -> -1)
* SATMathScore -> sat_math_score (empty -> -1)
* SATVerbalScore -> sat_verbal_score (empty -> -1)
* SATWritingScore -> sat_writing_score (empty -> -1)
* SATEssayScore -> sat_essay_score (empty -> -1)
* SATCompScore -> sat_comp_score (empty -> -1)
* gender_complete -> gender
* PELL_complete - as is
* level         - as is
* prior_bio     - as is
* cins*         - as is
* cans*         - as is

### Feature Elimination
| Feature                | Reason                                                                                      |
|------------------------|---------------------------------------------------------------------------------------------|
| unique_id              | Note a features                                                                             |
| semester               | Should be inconsequential over small periods of time                                        |
| time                   | Constant                                                                                    |
| consent_pre            | Constant                                                                                    |
| SAT1600Score           | Predicetd / calculated by others features                                                   |
| SATCompScore           | Predicetd / calculated by others features                                                   |
| hs_gpa_band            | Rollup value of hs_gpa, loss of precision. Model will most likely account for binning. <br> It should be noted that human binning may have little basis in reality.  There could be no <br> difference between B / A grade (bins).  Let the model determine bins / population breaks <br> This is a calculated value. |


### Setup
```
# Code snippets
sudo apt-get install python3-venv
python3 -m venv venv
source venv/bin/activate
pip3 install torch==2.4.1             # Models
pip3 install numpy==1.26.3            # For data management
pip3 install pandas==2.2.2            # For data management
pip3 install matplotlib==3.9.0        # For plotting
pip3 install scikit-learn==1.4.2      # TSNE plots
pip3 install umap-learn==0.5.6        # Umap
pip3 install seaborn==0.13.2          # For Data Analysis
```

### Execution
```
python -m src.data.cins_cans_eda -d data/SBU_Bio201_CINS_CANS.csv -o output
python -m src.model.fmt.train -o uniform -t toy -d uniform
python -m src.model.fmt.train -o uniform -t toy -d linear
python -m src.model.fmt.train -o uniform -t toy -d uniform-dif
python -m src.model.fmt.train -d data/SBU_Bio201_CINS_CANS.csv -o trial -t cins --train
python -m src.model.fmt.train -d data/SBU_Bio201_CINS_CANS.csv -o trial -t cins --eval -m trial
python -m src.model.fmt.train -d data/SBU_Bio201_CINS_CANS.csv -o trial -t cins --study
clear; python -m src.model.fmt.train2 -d data/SBU_Bio201_CINS_CANS.csv -o trial_t -t cins --train --eval -m trial_t
```