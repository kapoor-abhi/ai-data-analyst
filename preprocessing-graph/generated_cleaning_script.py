import pandas as pd
import numpy as np

try:
    # Load CSV
    df = pd.read_csv('retail_sales_messy.csv')

    # Column: transaction_id
    # No steps specified, so no action is taken

    # Column: category
    df['category'] = df['category'].apply(lambda x: x.lower())

    # Column: price
    df['price'] = df['price'].replace(['Unknown', 'Not Listed'], np.nan)
    df['price'] = pd.to_numeric(df['price'])
    df['price'] = df['price'].fillna(df['price'].median())

    # Column: transaction_date
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df['transaction_date'] = df['transaction_date'].ffill()
    df['transaction_date'] = df['transaction_date'].bfill()

    # Column: shipping_method
    df['shipping_method'] = df['shipping_method'].fillna('Unknown')

    # Save to 'cleaned_data.csv'
    df.to_csv('cleaned_data.csv', index=False)

except Exception as e:
    print(f"An error occurred: {e}")