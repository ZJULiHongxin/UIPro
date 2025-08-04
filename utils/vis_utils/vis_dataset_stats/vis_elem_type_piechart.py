import plotly.graph_objects as go
import plotly.io as pio
import os

# Set the output directory for saving the figure
output_dir = os.path.join(os.path.dirname(__file__), "elemtype_piechart_output")
os.makedirs(output_dir, exist_ok=True)

# Restricted list of GUI element types
element_types = [
    'Text', 'Image', 'Icon', 'Link', 'Input Field', 'Checkbox', 'Radio Button', 'Dropdown', 'Others'
]

# Adjusted distribution percentages for the restricted types
percentages = [
    26, 19, 18, 15, 6, 3, 2, 3, 8
]

# Create a colorful, high-contrast color palette
colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#ff9896'
]

# Create the pie chart with enhanced settings
fig = go.Figure(data=[go.Pie(
    labels=element_types,
    values=percentages,
    hole=0.4,
    marker=dict(colors=colors),
    textinfo='label+percent',
    textposition='outside',
    textfont=dict(size=26),
    insidetextfont=dict(size=12),
    pull=[0.1 if x == max(percentages) else 0 for x in percentages]
)])

# Update layout with improved aesthetics
fig.update_layout(
    title=dict(
        text="Distribution of GUI Element Types",
        font=dict(size=30),
        x=0.5,
        y=0.95
    ),
    legend=dict(
        font=dict(size=26),
        x=1.75,
        y=0.7,
        xanchor='right',
        yanchor='middle'
    ),
    margin=dict(l=20, r=20, t=80, b=20),
    width=1000,
    height=800,
    paper_bgcolor='white',
    plot_bgcolor='white'
)

# Increase the DPI for higher quality image
pio.write_image(fig, f"{output_dir}/gui_element_distribution.png", scale=2)
print(f"Pie chart saved to {output_dir}/gui_element_distribution.png")

# Also save as HTML for interactive viewing
pio.write_html(fig, f"{output_dir}/gui_element_distribution.html")
print(f"Interactive chart saved to {output_dir}/gui_element_distribution.html")
