import seaborn as sns
import pandas as pd

def download():
    print("Downloading penguins dataset...")
    df = sns.load_dataset('penguins')
    df.to_csv("penguins_raw.csv", index=False)
    print("Data saved to penguins_raw.csv")

if __name__ == "__main__":
    download()