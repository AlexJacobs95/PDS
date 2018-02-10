import pandas as pd


def balance(filename, num_by_cat):
    """
    Return a Dataset containing the same number of fake news
    than real ones.
    """

    unbalanced = pd.read_csv(filename, names=['code', 'text'])
    balanced = pd.DataFrame(columns=['code', 'text'])
    for code in range(1, 4):
        # Keep 250 fake news from each fake category
        selected_texts = unbalanced.loc[unbalanced['code'] == code][:num_by_cat]
        balanced = balanced.append(selected_texts)

    # Add the 750 real news
    balanced = balanced.append(unbalanced.loc[unbalanced['code'] == 4][:3*num_by_cat])
    print(balanced.groupby('code').size())

    return balanced


if __name__ == "__main__":
    balanced_test = balance("../dataset/balancedtest.csv", 250)
    # balanced_train = balance("../dataset/train.csv", 3331)

    balanced_test.to_csv("../dataset/test_OK.csv")
    # balanced_train.to_csv("../dataset/train_OK.csv")
