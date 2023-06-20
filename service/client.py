import pandas as pd
from sklearn.model_selection import train_test_split

if __name__=="__main__":
    df = pd.read_csv('kafka/dataset_sdn.csv')
    df = df.head(10000)
    X = df[['pktcount', 'bytecount']]
    y = df['label']
    X_norm = (X - X.mean()) / X.std()
    X_norm = X_norm.dropna()
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42)