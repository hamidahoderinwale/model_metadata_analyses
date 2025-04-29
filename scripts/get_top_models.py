import pandas as pd
import datetime
from huggingface_hub import HfApi
from pathlib import Path

# Initialize API
api = HfApi()
time = datetime.datetime.now().strftime("m%d%y_%H%M%S")  # Fixed time format (removed invalid `%_`)
# Fetch models (sorted by likes, excluding gated by default)
full_model_list = api.list_models(
    sort="likes",
    direction="-1",
    limit=15000,  # Explicitly set a limit
)

# Separate gated and non-gated models
non_gated_models = []
gated_models = []

for model in full_model_list:
    if getattr(model, 'gated', False):  # Check if model is gated
        gated_models.append(model)
    else:
        non_gated_models.append(model)



# Save non-gated models to CSV
df_non_gated = pd.DataFrame(non_gated_models)
non_gated_file = f"{len(non_gated_models)}non_gated.csv"
df_non_gated.to_csv(non_gated_file, index=True)  # `index=True` saves row numbers

# Save gated models to CSV (optional)
df_gated = pd.DataFrame(gated_models)
gated_file = f"gated_models_{time}.csv"
df_gated.to_csv(gated_file, index=True)

print(f"Non-gated models saved to: {Path(non_gated_file).resolve()}")
print(f"Gated models saved to: {Path(gated_file).resolve()}")
