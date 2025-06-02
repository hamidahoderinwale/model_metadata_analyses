# Install required packages
!pip install umap-learn matplotlib seaborn pandas sentence-transformers plotly scikit-learn

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
import umap
from sklearn.preprocessing import MinMaxScaler

# Set styling for better visualization
plt.style.use('fivethirtyeight')
sns.set_context("notebook", font_scale=1.2)

# --- Load and prepare data ---
input_csv = "model_library_count.csv"
df = pd.read_csv(input_csv)

print(f"Loaded {len(df)} libraries from CSV")
print("Sample data:")
print(df.head())

# --- Create embeddings ---
print("\nGenerating embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [f"Library {row['Library']} is used in {row['Model Count']} models" for _, row in df.iterrows()]
embeddings = model.encode(texts, show_progress_bar=True)

# --- Convert to DataFrame ---
embedding_cols = [f"dim_{i}" for i in range(embeddings.shape[1])]
embeddings_df = pd.DataFrame(embeddings, columns=embedding_cols)
final_df = pd.concat([df.reset_index(drop=True), embeddings_df], axis=1)

# --- Save output ---
output_csv = "model_library_with_embeddings.csv"
final_df.to_csv(output_csv, index=False)

# --- Create UMAP embedding ---
print("\nCreating UMAP projection...")
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric='cosine',
    random_state=42
)
embedding_2d = reducer.fit_transform(embeddings)

# Add to DataFrame
final_df["UMAP_1"] = embedding_2d[:, 0]
final_df["UMAP_2"] = embedding_2d[:, 1]

# Create log-scale version of model count for better visualization
final_df["log_count"] = np.log1p(final_df["Model Count"])

# Create size scale
size_scaler = MinMaxScaler(feature_range=(10, 100))
final_df["point_size"] = size_scaler.fit_transform(final_df[["log_count"]]) * 5

# --- Create categories based on popularity ---
def categorize_popularity(count):
    if count >= 100:
        return "Very Popular (100+)"
    elif count >= 50:
        return "Popular (50-99)"
    elif count >= 20:
        return "Common (20-49)"
    elif count >= 10:
        return "Moderate (10-19)"
    else:
        return "Niche (<10)"

final_df["Popularity"] = final_df["Model Count"].apply(categorize_popularity)

# Sort by popularity for better layering in visualization
final_df = final_df.sort_values("Model Count")

# --- MATPLOTLIB VISUALIZATION (STATIC) ---
plt.figure(figsize=(16, 10))
ax = plt.subplot(111)

# Create a custom colormap
colors = sns.color_palette("viridis", n_colors=len(final_df["Popularity"].unique()))

# Define a function to get the sort value based on category
def get_sort_value(category):
    if category == "Very Popular (100+)":
        return 5
    elif category == "Popular (50-99)":
        return 4
    elif category == "Common (20-49)":
        return 3
    elif category == "Moderate (10-19)":
        return 2
    else:  # "Niche (<10)"
        return 1

# Sort and create color map
color_map = dict(zip(
    sorted(final_df["Popularity"].unique(), key=get_sort_value, reverse=True),
    colors
))

# Plot scatter with categorized colors
for category, group in final_df.groupby("Popularity"):
    ax.scatter(
        group["UMAP_1"], group["UMAP_2"],
        s=group["point_size"] * 5,
        alpha=0.7,
        label=category,
        color=color_map[category],
        edgecolor='white',
        linewidth=0.5
    )

# Add labels for top libraries
top_n = 20
for _, row in final_df.nlargest(top_n, "Model Count").iterrows():
    plt.annotate(
        row["Library"],
        xy=(row["UMAP_1"], row["UMAP_2"]),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=12,
        weight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="gray")
    )

# Add grid and styling
plt.grid(True, linestyle='--', alpha=0.7)
plt.title("Library Ecosystem in ML Models", fontsize=20, pad=20)
plt.xlabel("UMAP Dimension 1", fontsize=14)
plt.ylabel("UMAP Dimension 2", fontsize=14)

# Add legend with custom placement
legend = plt.legend(
    title="Library Popularity",
    loc='upper right',
    frameon=True,
    framealpha=0.9,
    fontsize=12
)
legend.get_title().set_fontsize(14)

# Add usage info
plt.figtext(
    0.5, 0.01,
    f"Based on {final_df['Model Count'].sum()} total model implementations across {len(final_df)} libraries",
    ha='center',
    fontsize=12,
    style='italic'
)

plt.tight_layout()
plt.savefig("static_umap_libraries.png", dpi=300, bbox_inches='tight')
plt.show()

# --- PLOTLY VISUALIZATION (INTERACTIVE) ---
print("\nCreating interactive visualization...")

# Create hover text
final_df["hover_text"] = final_df.apply(
    lambda row: f"<b>{row['Library']}</b><br>Used in {row['Model Count']} models",
    axis=1
)

# Create interactive plot with Plotly
fig = px.scatter(
    final_df,
    x="UMAP_1",
    y="UMAP_2",
    size="point_size",
    color="Popularity",
    hover_name="Library",
    hover_data={"UMAP_1": False, "UMAP_2": False, "point_size": False, "Popularity": True, "Model Count": True},
    color_discrete_map={k: f"rgb{tuple(int(c*255) for c in color_map[k])}" for k in color_map},
    title="Interactive Library Ecosystem Visualization",
    category_orders={"Popularity": sorted(final_df["Popularity"].unique(), key=get_sort_value, reverse=True)}
)

# Update layout for better appearance
fig.update_layout(
    template="plotly_white",
    title={
        'text': "Interactive Library Ecosystem in ML Models",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=24)
    },
    legend_title_text="Library Popularity",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.2,
        xanchor="center",
        x=0.5
    ),
    xaxis_title="UMAP Dimension 1",
    yaxis_title="UMAP Dimension 2",
    height=800,
    width=1200,
    margin=dict(l=40, r=40, t=80, b=40),
    annotations=[
        dict(
            text=f"Based on {final_df['Model Count'].sum()} total model implementations across {len(final_df)} libraries",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.15,
            font=dict(size=12, color="gray", style="italic")
        )
    ]
)

# Add labels for top libraries
for _, row in final_df.nlargest(15, "Model Count").iterrows():
    fig.add_annotation(
        x=row["UMAP_1"],
        y=row["UMAP_2"],
        text=row["Library"],
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=15,
        ay=-30,
        bgcolor="white",
        bordercolor="#c7c7c7",
        borderwidth=1,
        borderpad=4,
        font=dict(size=12, color="#000000")
    )

# Update traces
fig.update_traces(
    marker=dict(line=dict(width=1, color='DarkSlateGrey')),
    selector=dict(mode='markers')
)

# Show and save the interactive visualization
fig.write_html("interactive_umap_libraries.html")
fig.show()

# --- BONUS: Create a 3D UMAP visualization ---
print("\nCreating 3D UMAP visualization...")

# Create 3D UMAP projection
reducer_3d = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=3,
    metric='cosine',
    random_state=42
)
embedding_3d = reducer_3d.fit_transform(embeddings)

# Add to DataFrame
final_df["UMAP_3D_1"] = embedding_3d[:, 0]
final_df["UMAP_3D_2"] = embedding_3d[:, 1]
final_df["UMAP_3D_3"] = embedding_3d[:, 2]

# Create 3D visualization
fig_3d = px.scatter_3d(
    final_df,
    x="UMAP_3D_1",
    y="UMAP_3D_2",
    z="UMAP_3D_3",
    size="point_size",
    color="Popularity",
    hover_name="Library",
    hover_data={"UMAP_3D_1": False, "UMAP_3D_2": False, "UMAP_3D_3": False, "point_size": False, "Model Count": True},
    color_discrete_map={k: f"rgb{tuple(int(c*255) for c in color_map[k])}" for k in color_map},
    title="3D Library Ecosystem Visualization",
    category_orders={"Popularity": sorted(final_df["Popularity"].unique(), key=get_sort_value, reverse=True)}
)

# Update 3D layout
fig_3d.update_layout(
    template="plotly_white",
    title={
        'text': "3D Library Ecosystem in ML Models",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=24)
    },
    scene=dict(
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        zaxis_title="UMAP Dimension 3"
    ),
    height=800,
    width=1200,
    margin=dict(l=0, r=0, t=50, b=0)
)

# Show and save the 3D visualization
fig_3d.write_html("3d_umap_libraries.html")
fig_3d.show()

print("\nVisualization complete! Files saved:")
print("- static_umap_libraries.png (Static visualization)")
print("- interactive_umap_libraries.html (Interactive 2D visualization)")
print("- 3d_umap_libraries.html (Interactive 3D visualization)")
print("- model_library_with_embeddings.csv (Enriched data with embeddings)")
