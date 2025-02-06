# read_columns.py
import pandas as pd
import sys

def read_columns(file_path):
    try:
        # Read just the header
        df = pd.read_csv(file_path, nrows=0)
        print("\nColumn names:")
        for i, col in enumerate(df.columns):
            print(f"{i}: {col}")
            
        # Read first few rows to see data types
        print("\nFirst few rows sample:")
        sample = pd.read_csv(file_path, nrows=3)
        print(sample)
        
    except Exception as e:
        print(f"Error reading file: {str(e)}")

if __name__ == "__main__":
    file_path = "/home/rhildebr/dim_sum/benchmark_code/DIM_SUM/config/datasets/to/TO_FULL.csv"
    read_columns(file_path)