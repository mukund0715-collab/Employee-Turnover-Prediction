import pandas as pd
import numpy as np
from faker import Faker

# Initialize Faker
fake = Faker()

# Number of rows per file
n_rows = 10000
# Number of files
n_files = 5

# Function to create one dataset
def create_dataset(n_rows):
    data = {
        "Name": [fake.first_name() + " " + fake.last_name() for _ in range(n_rows)],
        "satisfaction_level": np.round(np.random.uniform(0, 1, n_rows), 2),
        "last_evaluation": np.round(np.random.uniform(0, 1, n_rows), 2),
        "number_project": np.random.randint(2, 7, n_rows),
        "average_montly_hours": np.random.randint(90, 310, n_rows),
        "time_spend_company": np.random.randint(2, 11, n_rows),
        "Work_accident": np.random.randint(0, 2, n_rows),
        "left": np.random.randint(0, 2, n_rows),
        "promotion_last_5years": np.random.randint(0, 2, n_rows),
        "sales": np.random.choice(
            ["sales", "accounting", "hr", "technical", "support", "IT",
             "product_mng", "marketing", "RandD", "management"],
            n_rows
        ),
        "salary": np.random.choice(["low", "medium", "high"], n_rows)
    }
    return pd.DataFrame(data)

# Generate and save multiple files
for i in range(1, n_files + 1):
    df = create_dataset(n_rows)
    filename = f"HR_file_part{i}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {filename}")
