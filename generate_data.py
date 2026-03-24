import pandas as pd
import numpy as np

np.random.seed(42)

data_size = 300

data = {
    "age": np.random.randint(18, 40, data_size),
    "cycle_length": np.random.randint(24, 38, data_size),
    "pain_level": np.random.randint(1, 10, data_size),
    "flow": np.random.randint(1, 4, data_size),
    "fatigue": np.random.randint(1, 10, data_size),
    "mood_swings": np.random.randint(1, 10, data_size),
}
df = pd.DataFrame(data)

df["anemia_risk"] = ((df["flow"] == 3) & (df["fatigue"] > 6)).astype(int)
df["irregular"] = (df["cycle_length"] > 32).astype(int)

df.to_csv("data.csv", index=False)

print("Dataset created!")