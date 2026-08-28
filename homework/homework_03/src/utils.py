def get_summary_stats(df):
    df.describe(include="all")
    print("----------------------")
    df.info()
    print("----------------------")
    print('num of missing values: ',df.isnull().sum())
    print("----------------------")
    print("Num of unique values: ",    df.nunique())


