import json
import os
import plotly.graph_objects as go
import plotly.io as pio
import re
from collections import Counter

# Set the output directory for saving the figure
output_dir = os.path.join(os.path.dirname(__file__), "acttype_ratio_output")
os.makedirs(output_dir, exist_ok=True)

def extract_action_type(data_sample):
    """Extract the action type from a data sample."""
    response = data_sample['conversations'][-1]['value']
    if response:
        # Try to extract action_type from JSON content
        value = response
        try:
            # First try to parse as valid JSON
            if 'action_type' in value:
                match = re.search(r'"action_type":\s*"([^"]+)"', value)
                if match:
                    return match.group(1)
            
            # Try to extract from a proper JSON structure
            action_json = re.search(r'\{.*\}', value)
            if action_json:
                try:
                    action_data = json.loads(action_json.group(0))
                    if isinstance(action_data, dict) and 'action_type' in action_data:
                        return action_data['action_type']
                except json.JSONDecodeError:
                    pass
        except:
            pass
    return None

def plot_action_type_distribution(data_samples):
    """Plot the distribution of action types."""
    action_types = []
    
    for sample in data_samples:
        action_type = extract_action_type(sample)
        if action_type:
            # Group less common action types into "Others"
            if action_type in ['long_press', 'drag', 'navigate_recent']:
                action_type = "Others"
            action_types.append(action_type)
    
    # Count occurrences of each action type
    counter = Counter(action_types)
    
    # Sort by frequency, but ensure "Others" is at the end if present
    sorted_items = counter.most_common()
    if any(item[0] == "Others" for item in sorted_items):
        # Remove "Others" from its current position
        others_item = next((item for item in sorted_items if item[0] == "Others"), None)
        sorted_items.remove(others_item)
        # Add "Others" at the end
        sorted_items.append(others_item)
    
    labels = [item[0].title() for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    # Create a colorful palette
    colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#bcbd22',  # olive
        '#17becf',  # cyan
        '#aaaaaa'   # grey - for "Others"
    ]
    
    # If "Others" is in the labels, use grey for it
    if "Others" in labels:
        others_index = labels.index("Others")
        others_color = '#aaaaaa'  # grey
        color_list = colors[:len(labels)]
        color_list[others_index] = others_color
    else:
        color_list = colors[:len(labels)]
    
    # Create horizontal bar chart with enhanced settings
    fig = go.Figure(data=[go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker=dict(
            color=color_list,
            line=dict(width=1)
        ),
        text=values,
        textposition='outside',
        textfont=dict(size=20)
    )])
    
    # Update layout with improved aesthetics
    fig.update_layout(
        title=dict(
            text="Distribution of Action Types",
            font=dict(size=32),
            x=0.5,
            y=0.95
        ),
        xaxis=dict(
            title=dict(
                text="Count",
                font=dict(size=26)
            ),
            tickfont=dict(size=22)
        ),
        yaxis=dict(
            title=dict(
                text="Action Type",
                font=dict(size=26)
            ),
            tickfont=dict(size=22)
        ),
        margin=dict(l=30, r=30, t=100, b=30),
        width=1200,
        height=800,
        paper_bgcolor='white',
        plot_bgcolor='rgba(240, 240, 240, 0.5)'
    )
    
    # Add data value annotations
    for i, value in enumerate(values):
        fig.add_annotation(
            x=value,
            y=labels[i],
            text=f"{value} ({(value/sum(values)*100):.1f}%)",
            showarrow=False,
            xshift=10,
            font=dict(size=20)
        )
    
    # Add a total count annotation
    fig.add_annotation(
        x=0.98,
        y=1.05,
        xref='paper',
        yref='paper',
        text=f"Total samples: {sum(values)}",
        showarrow=False,
        font=dict(size=22)
    )
    
    # Increase the DPI for higher quality image
    pio.write_image(fig, f"{output_dir}/action_type_distribution.png", scale=3)
    print(f"Bar chart saved to {output_dir}/action_type_distribution.png")
    
    # Also save as HTML for interactive viewing
    pio.write_html(fig, f"{output_dir}/action_type_distribution.html")
    print(f"Interactive chart saved to {output_dir}/action_type_distribution.html")
    
    # Create a pie chart as well
    fig_pie = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=color_list),
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(size=20),
        insidetextfont=dict(size=18),
        pull=[0.1 if x == max(values) and item != "Others" else 0 for x, item in zip(values, labels)]
    )])
    
    # Update pie chart layout
    fig_pie.update_layout(
        # title=dict(
        #     text="Distribution of Action Types",
        #     font=dict(size=32),
        #     x=0.5,
        #     y=0.95
        # ),
        legend=dict(
            font=dict(size=22),
        ),
        margin=dict(l=20, r=20, t=100, b=20),
        width=900,
        height=700,
        paper_bgcolor='white'
    )
    
    # Save pie chart
    pio.write_image(fig_pie, f"{output_dir}/action_type_pie.png", scale=3)
    print(f"Pie chart saved to {output_dir}/action_type_pie.png") 
    
    # Also save pie chart as HTML
    pio.write_html(fig_pie, f"{output_dir}/action_type_pie.html")
    print(f"Interactive pie chart saved to {output_dir}/action_type_pie.html")

if __name__ == "__main__":
    # Example data samples
    
    file = "/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/UIPro_processed/6mobile-planning_380024.json"
    data_samples = json.load(open(file))
    
    # In practice, you would load data from a file or database
    # data_samples = load_data_from_file('path/to/your/data.json')
    
    plot_action_type_distribution(data_samples)
