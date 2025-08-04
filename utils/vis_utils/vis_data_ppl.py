import json, glob, math, os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# List of file paths containing data series
file_paths = glob.glob("/mnt/nvme0n1p1/hongxin_li/LLaMA-Factory/ppl_results/*")  # Replace with your actual file paths
dataset_names = [os.path.basename(file_path).split('_')[1] for file_path in file_paths]
# Create subplots
fig = make_subplots(rows=len(file_paths), cols=3, subplot_titles=dataset_names)

num_files = len(file_paths)
cols = 3  # You can change this to any number that fits your needs
rows = math.ceil(num_files / cols)

for i, file_path in enumerate(file_paths):
    # Load data from each file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract the data series you want to plot
    data_series = data['ppls']  # Adjust based on your data structure

    # Determine the subplot location
    row = i // cols + 1
    col = i % cols + 1

    # Add histogram to the subplot
    fig.add_trace(
        go.Histogram(x=data_series, name=dataset_names[i]),
        row=row, col=col
    )
    # Update x and y axis titles for each subplot
    fig.update_xaxes(title_text="PPL", row=row, col=col)
    fig.update_yaxes(title_text="Frequency", row=row, col=col)

fig.update_layout(
    height=500*rows, width=1500,
    title_text="Histograms for Each File",
    showlegend=False
)


# Show the plot
fig.show()