""" Run this script to train the linear model. """
import os
from enum import Enum
import argparse
import seaborn
import matplotlib.pyplot as plt

from src.data.cins_cans_data_prep import load_data

READING_ABILITY_CINS = True
READING_ABILITY_CANS = True
CORRELATION_MAPS = True

class TargetSet(Enum):
    ''' Enum for the target set, either cans or cins '''
    CANS = (['cans_0' + str(i) if i < 10 else 'cans_' + str(i) for i in range(1, 25)])
    CINS = (['cins_0' + str(i) if i < 10 else 'cins_' + str(i) for i in range(1, 21)])

    def __str__(self):
        '''
        Will print either 'cans' or 'cins'
        '''
        return self.name.lower()

    def get_str_array(self):
        '''
        Will return the array of strings for cans or cins
        '''
        return self.value

    def get_q_count(self):
        '''
        Will return the number of questions for cans or cins
        '''
        return len(self.value)

def efl_study(data, out_dir, demograph, target_set: TargetSet):
    '''
    This function will create a series of plots to show the relationship between
    reading ability and english first language on the target set.
    '''
    efl_yes = data[data['english_first_language'] == 1]
    efl_no = data[data['english_first_language'] == 0]
    target = 'cins_total_pre' if target_set == TargetSet.CINS else 'cans_total_pre'

    # Make directory for reading_ability vs target
    out_dir = out_dir + '/'+ target_set.name.lower() + '/' + demograph.lower()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.figure()
    # seaborn.kdeplot(efl_yes[efl_yes[demograph] == 0][target], label='0') // 16
    # seaborn.kdeplot(efl_yes[efl_yes[demograph] == 1][target], label='1') // 7
    seaborn.kdeplot(efl_yes[efl_yes[demograph] == 2][target], label='2')
    seaborn.kdeplot(efl_yes[efl_yes[demograph] == 3][target], label='3')
    seaborn.kdeplot(efl_yes[efl_yes[demograph] == 4][target], label='4')
    seaborn.kdeplot(efl_yes[efl_yes[demograph] == 5][target], label='5')
    plt.xlim(0, target_set.get_q_count())
    plt.legend(title=demograph)
    plt.title('English First Language: Yes')
    plt.savefig(out_dir + '/efl_yes.png')
    plt.close()

    plt.figure()
    # seaborn.kdeplot(efl_yes[efl_yes[demograph] == 0][target], label='0') // 7
    # seaborn.kdeplot(efl_yes[efl_yes[demograph] == 1][target], label='1') // 39
    seaborn.kdeplot(efl_no[efl_no[demograph] == 2][target], label='2')
    seaborn.kdeplot(efl_no[efl_no[demograph] == 3][target], label='3')
    seaborn.kdeplot(efl_no[efl_no[demograph] == 4][target], label='4')
    seaborn.kdeplot(efl_no[efl_no[demograph] == 5][target], label='5')
    plt.xlim(0, target_set.get_q_count())
    plt.legend(title=demograph)
    plt.title('English First Language: No')
    plt.savefig(out_dir + '/elf_no.png')
    plt.close()

    for i in range(2,6):
        plt.figure()
        seaborn.kdeplot(efl_yes[efl_yes[demograph] == i][target], label='yes')
        seaborn.kdeplot(efl_no[efl_no[demograph] == i][target], label='no')
        plt.xlim(0, target_set.get_q_count())
        plt.legend(title='english_first_language')
        plt.title('Reading Ability: ' + str(i))
        plt.savefig(out_dir + '/' + str(i) +  '.png')
        plt.close()

    # Create 8 groups reading_ability * english_first_language and bin target by correct
    for entry in target_set.get_str_array():
        ratio = data.groupby([demograph, 'english_first_language'])[entry].mean()
        ratio = ratio.drop([0, 1], level=0)
        # remove efl 2
        ratio = ratio.drop(2, level=1)

        ratio = ratio.unstack()
        ratio = ratio.T
        ratio.plot(kind='bar')
        plt.title(entry)
        # Legend in bottom center
        plt.legend(loc='lower center', shadow=True, ncol=1, title=demograph)
        plt.savefig(out_dir + '/' + entry + '.png')
        plt.close()

def correlation_map(data, out_file, features, title='Correlation Matrix'):
    '''
    This function will create a correlation matrix of the data
    param data: the data to be used
    param features: the features array (list of strings)
    param target: the target variable (string)
    '''
    plt.figure()
    plt.figure(figsize=(12, 12))
    seaborn.heatmap(data[features].corr(), annot=True, cmap='coolwarm', linewidths=0.7)
    plt.title(title)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    # Make image larger
    plt.tight_layout()

    plt.savefig(out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the linear model.")
    parser.add_argument("-d", "--dataPath", type=str, help="Path to the data file.")
    parser.add_argument("-o", "--outDir", type=str, help="Path to the outputs.")
    args = parser.parse_args()

    # Make output directory if it doesn't exist
    if not os.path.exists(args.outDir):
        os.makedirs(args.outDir)

    data_set = load_data(args.dataPath, prep=True)

    #print columns
    print(data_set.columns)

    ## Assessment of reading ability and native language on cins_total_pre
    ## Categories reading_ability 0 and 1 excluded due to low representation
    if READING_ABILITY_CINS:
        DEMOGRAPH = 'reading_ability'
        efl_study(data_set, args.outDir, DEMOGRAPH, TargetSet.CINS)

    if READING_ABILITY_CANS:
        DEMOGRAPH = 'reading_ability'
        efl_study(data_set, args.outDir, DEMOGRAPH, TargetSet.CANS)

    if CORRELATION_MAPS:
                # Estiated to be highly correlated with each other...but also distinct?
        inputs = ['sat_math_score', 'sat_verbal_score', 'sat_writing_score', 'sat_essay_score',
                # Additional placement tests
                'math_placement', 'writing_placement','reading_ability', 'writing_ability',
                # Cumulative HS GPA - may be correlated with sat scores - may remove
                'hs_gpa',
                # Highly correlated with hs_gpa, but may have impacts on target
                'cum_college_gpa']

        TITLE = ' to Assessment - Correlation Matrix'
        file_out = args.outDir + '/cins/heatmap_assessments.png'
        correlation_map(data_set, file_out, inputs+['cins_total_pre'], TargetSet.CINS.name + TITLE)
        file_out = args.outDir + '/cans/heatmap_assessments.png'
        correlation_map(data_set, file_out, inputs+['cans_total_pre'], TargetSet.CANS.name + TITLE)

        inputs = ['plan', 'level', 'english_first_language', 'prior_bio',
                'age', 'gender', 'PELL_complete','citizenship_e',
                'country_citizenship_e', 'family_income_e', 'first_generation_e',
                'race_e', 'sbu_admit_e']

        TITLE = ' to Demographics - Correlation Matrix'
        file_out = args.outDir + '/cins/heatmap_demograph.png'
        correlation_map(data_set, file_out, inputs+['cins_total_pre'], TargetSet.CINS.name + TITLE)
        file_out = args.outDir + '/cans/heatmap_demograph.png'
        correlation_map(data_set, file_out, inputs+['cans_total_pre'], TargetSet.CANS.name + TITLE)

    print("Done.")
