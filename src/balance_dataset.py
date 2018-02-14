import pandas as pd


def fusionDatasets(filename1, filename2, balanced=True):
    """
    Fusion 2 datasets
    """

    d1 = pd.read_csv(filename1, names=['code', 'text'])
    d2 = pd.read_csv(filename2, names=['code', 'text'])

    fusion = d1.append(d2)
    fusion.reset_index(drop=True, inplace=True)

    if balanced:
        fusion = createBalancedDataset(fusion, num_by_cat=3581)

    fusion['code'] = fusion['code'].replace(1, False).replace(2, False).replace(3, False).replace(4, True)

    return fusion


def splitDataset(filename, prct_train):
    """
    Take a dataset and return 2 new ones :
        - The train dataset containing prct_train % of the full dataset
        - The test dataset containing 100 - prct_train % of the full dataset
    """

    full_dataset = pd.read_csv(filename, index_col=0)
    size = len(full_dataset)

    nb_train = round((prct_train / 100) * size)
    nb_fakes = nb_train // 2
    nb_real = nb_train - nb_fakes

    fakes = full_dataset.loc[full_dataset['code'] == False]
    real = full_dataset.loc[full_dataset['code'] == True]

    train_dataset = fakes[:nb_fakes].append(real[:nb_real])
    test_dataset = fakes[nb_fakes:].append(real[nb_real:])

    train_dataset.reset_index(drop=True, inplace=True)
    test_dataset.reset_index(drop=True, inplace=True)

    return train_dataset, test_dataset


def createBalancedDataset(dataframe, num_by_cat):
    """
    Creates a balanced dataset containing the same number of fake news as real news.
    """
    unbalanced = dataframe
    balanced = pd.DataFrame(columns=['code', 'text'])
    for code in range(1, 4):
        # Keep num_by_cat fake news from each fake category
        selected_texts = unbalanced.loc[unbalanced['code'] == code][:num_by_cat]
        balanced = balanced.append(selected_texts)

    # Add the 3*num_by_cat real news
    balanced = balanced.append(unbalanced.loc[unbalanced['code'] == 4][:3 * num_by_cat])
    balanced.reset_index(drop=True, inplace=True)

    return balanced


def mainFusion():
    fusion_balanced = fusionDatasets("../dataset/train.csv", "../dataset/balancedtest.csv", balanced=True)
    fusion_balanced.to_csv("../dataset/fusion_balanced.csv")

    fusion = fusionDatasets("../dataset/train.csv", "../dataset/balancedtest.csv", balanced=False)
    fusion.to_csv("../dataset/fusion.csv")


def mainSplit():
    train, test = splitDataset("../dataset/fusion_balanced.csv", prct_train=80)

    train.to_csv("../dataset/train_80.csv")
    test.to_csv("../dataset/test_20.csv")


if __name__ == "__main__":
    mainSplit()
