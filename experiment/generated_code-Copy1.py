def main(p_max_depth, p_n_estimators): 
    from google.cloud import bigquery
    import pandas as pd
    import datetime
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    from sklearn.model_selection import train_test_split

    from sklearn.preprocessing import OneHotEncoder

    from sklearn.metrics import accuracy_score
    import pickle as pk
    from google.cloud import storage as gcs
    import google.cloud.aiplatform as aip
    client = bigquery.Client()
    def load_from_bq(query):

        return client.query(query).to_dataframe()
    query = """

【テストテーブル】

    """

    df_test = load_from_bq(query)
    query = """

【トレーニングテーブル】

    """

    df_train = load_from_bq(query)
    train, val = train_test_split(df_train)
    def make_encorder(df):

        x_ = df[["Pclass", "Sex", "SibSp", "Embarked"]]

        

        enc = OneHotEncoder()

        enc.fit(x_)

        

        return enc
    enc = make_encorder(df_train)
    def make_xy(df, enc):

        x_ = df[["Pclass", "Sex", "Age", "SibSp", "Fare", "Embarked"]]

        

        if "Survived" in df.columns:

            y_ = df["Survived"]

        else:

            y_ = 0

        

        #ここから欠損値の補完やらone-hot化やら

        x_float    = x_[["Age", "Fare"]]

        x_category = x_[["Pclass", "Sex", "SibSp", "Embarked"]]

        

        x_float = x_float.fillna(0)

        

        #enc = OneHotEncoder(handle_unknown='ignore')

        #enc.fit(x_category)

        

        

        

        x = enc.transform(x_category).toarray()

    

        li = []

        for col_name, elements in zip(x_category.columns, enc.categories_):

            for e in elements:

                li.append(col_name +"_" +  str(e))

        #print(enc.categories_)

        cols_ = list(x_float.columns) + li

        values_ =  np.concatenate([x_float.values, x], axis = 1)

        

        x_ = pd.DataFrame(values_, columns = cols_)

        

        return x_, y_
    x_train, y_train = make_xy(train, enc)

    x_val, y_val     = make_xy(val, enc)

    x_test, y_test        = make_xy(df_test, enc)
    clf = RandomForestClassifier(max_depth=p_max_depth , n_estimators = p_n_estimators)
    clf.fit(x_train, y_train)
    val_score = accuracy_score(y_val, clf.predict(x_val))
    test_score = accuracy_score(y_test, clf.predict(x_test))
    test_score
    val_score
    # client = gcs.Client("test-hyron")

    # bucket = client.get_bucket("cyberpot-titanic")

    # blob = bucket.blob("model/model.pkl")
    # pk.dump(clf, blob.open('wb'))
    # model = aip.Model.upload(

    #     display_name='my-model',

    #     artifact_uri="gs://cyberpot-titanic/model",

    #     serving_container_image_uri ="gcr.io/deeplearning-platform-release/sklearn-cpu"

    # )
    return test_score, val_score
