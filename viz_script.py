import pandas as pd
import ast
import matplotlib.pyplot as plt
from Bio import Phylo
from Bio.Phylo.BaseTree import Clade, Tree

# Load dataset
csv_as_input = input("give link to model tree csv: ")
filename = csv_as_input.split('_finetunes_')[0] 
df = pd.read_csv(csv_as_input)
df['children'] = df['children'].apply(ast.literal_eval)

def build_phylo_tree_from_dfs(df):
    clade_map = {row['model_id']: Clade(name=row['model_id']) for _, row in df.iterrows()}
    parent_links = {}
    for _, row in df.iterrows():
        parent = row['model_id']
        for child_url in row['children']:
            child = '/'.join(child_url.split("/")[-2:])
            parent_links[child] = parent
    all_models = set(df['model_id'])
    child_models = set(parent_links.keys())
    root_model_id = list(all_models - child_models)[0]
    for child, parent in parent_links.items():
        if child in clade_map and parent in clade_map:
            clade_map[parent].clades.append(clade_map[child])
    return Tree(root=clade_map[root_model_id])

# Build the tree
tree = build_phylo_tree_from_dfs(df)

# Plot the tree with model names as tick labels
fig = plt.figure(figsize=(12, 16))
ax = fig.add_subplot(1, 1, 1)

# Custom drawing to prevent line strikes
Phylo.draw(tree, axes=ax, do_show=False)

# Customize the display to remove the strike-through lines
for clade in tree.find_clades():
    if clade.name:
        # Find the existing text elements and adjust them
        for text in ax.texts:
            if text.get_text() == clade.name:
                # Move text slightly to avoid intersection with lines
                current_pos = text.get_position()
                text.set_position((current_pos[0] + 0.02, current_pos[1]))
                # Make sure text is rendered above the branch lines
                text.set_zorder(10)

# Remove the axes completely
ax.set_axis_off()

plt.tight_layout()
plt.show()

fig.savefig(filename, dpi=150, bbox_inches='tight')

print(f"Tree visualization {filename}")
