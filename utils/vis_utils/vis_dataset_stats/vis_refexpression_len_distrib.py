import os
import json
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Visualize GPT response length distribution')
    parser.add_argument('--data_path', type=str, required=False, default="/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/UIPro_processed/UIPro_18643k.json", help='Path to the dataset file')
    parser.add_argument('--output_dir', type=str, default=os.path.join(os.path.dirname(__file__), './refexp_length_distrib'), help='Output directory for visualization')
    parser.add_argument('--bins', type=int, default=10, help='Number of bins for the histogram')
    parser.add_argument('--max_length', type=int, default=150, help='Maximum response length to consider')
    parser.add_argument('--width', type=int, default=1000, help='Width of the output figure in pixels')
    parser.add_argument('--height', type=int, default=700, help='Height of the output figure in pixels')
    parser.add_argument('--dpi', type=int, default=400, help='DPI for the output image')
    parser.add_argument('--force_reprocess', action='store_true', help='Force reprocessing of the dataset even if preprocessed data exists')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if preprocessed data exists
    npy_path = os.path.join(args.output_dir, os.path.basename(args.data_path).split('.')[0] + '_length_data.npy')
    
    if os.path.exists(npy_path) and not args.force_reprocess:
        print(f"Loading preprocessed data from {npy_path}...")
        refexp_lengths = np.load(npy_path)
    else:
        # Load the dataset
        print(f"Loading dataset from {args.data_path}...")
        with open(args.data_path, 'r') as f:
            dataset = json.load(f)
        
        # Process each sample in the dataset
        print(f"Processing {len(dataset)} samples...")
        
        refexp_lengths = []
        
        for sample in tqdm(dataset):
            # Extract conversations
            if 'conversations' not in sample:
                continue
            
            convs = sample['conversations']
            for i in range(1, len(convs), 2):
                # Extract GPT responses (assuming every other message starting from index 1)
                prompt = convs[i-1]['value'].replace("<image>", "").strip()
                response_text = convs[i]['value']
                
                if 'list all' in prompt:
                    for line in response_text.split('\n'):
                        refexp = line[line.find("'")+1:line.rfind("'")]
                        refexp_length = len(refexp)
                        if refexp_length <= args.max_length:
                            refexp_lengths.append(refexp_length)
                elif response_text.startswith('(') and response_text.endswith(')') and response_text.count(',') in [1,3]:
                    if 'I want to ' in prompt:
                        refexp = prompt.split('. ')[0]
                        refexp_length = len(refexp)
                    elif 'Locate the text' in prompt: # Locate the text \"Lumbar Spine\" (with bbox)"
                        refexp = prompt[prompt.find("\"")+1:prompt.rfind("\"")]
                        refexp_length = len(refexp)
                    elif 'This element ' in prompt:
                        refexp = prompt[prompt.find("This element ")]
                        refexp_length = len(refexp)
                    elif 'Where is ' in prompt: # Where is Apps icon? (with point)
                        refexp = prompt.split("Where is ")[1].split("?")[0].strip()
                        refexp_length = len(refexp)
                    elif prompt.endswith('"'):
                        refexp = prompt[prompt.find("\"")+1:prompt.rfind("\"")]
                        refexp_length = len(refexp)
                    else:
                        print(f"Unknown prompt: {prompt}")
                        continue
                    
                    if refexp_length <= args.max_length:
                        refexp_lengths.append(refexp_length)
                else:
                    refexp_length = len(response_text)
                    if refexp_length <= args.max_length:
                        refexp_lengths.append(refexp_length)
                        
        # Save the raw data
        np.save(npy_path, np.array(refexp_lengths))
        print(f"Raw data saved to {npy_path}")
    
    # Calculate statistics
    mean_length = np.mean(refexp_lengths)
    median_length = np.median(refexp_lengths)
    min_length = np.min(refexp_lengths)
    max_length = np.max(refexp_lengths)
    std_dev = np.std(refexp_lengths)
    
    # Create histogram bins
    hist, bin_edges = np.histogram(refexp_lengths, bins=args.bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Convert frequency to percentage
    hist_percentage = (hist / len(refexp_lengths)) * 100
    
    # Create a gradient color scale based on bin heights
    colors = [[h / max(hist_percentage), f'rgba({int(255 * (1 - h / max(hist_percentage)))}, {int(50 + 200 * h / max(hist_percentage))}, {int(255 * h / max(hist_percentage))}, 0.8)'] for h in hist_percentage]
    
    # Create the plotly figure
    fig = go.Figure()
    
    # Add histogram bars with gradient colors
    for i in range(len(hist_percentage)):
        fig.add_trace(go.Bar(
            x=[bin_centers[i]],
            y=[hist_percentage[i]],
            width=(bin_edges[i+1] - bin_edges[i]) * 0.9,  # Slightly narrower than bin width
            marker_color=colors[i][1],
            name=f'{int(bin_edges[i])}-{int(bin_edges[i+1])}',
            hovertemplate=f'Range: {int(bin_edges[i])}-{int(bin_edges[i+1])}<br>Percentage: %{{y:.2f}}%<extra></extra>'
        ))
    
    # Add mean and median lines
    fig.add_trace(go.Scatter(
        x=[mean_length, mean_length],
        y=[0, max(hist_percentage) * 1.1],
        mode='lines',
        line=dict(color='rgba(255, 0, 0, 0.8)', width=3, dash='dash'),
        name=f'Mean: {mean_length:.2f}'
    ))
    
    fig.add_trace(go.Scatter(
        x=[median_length, median_length],
        y=[0, max(hist_percentage) * 1.1],
        mode='lines',
        line=dict(color='rgba(0, 128, 0, 0.8)', width=3, dash='dash'),
        name=f'Median: {median_length:.2f}'
    ))
    
    # Update layout with larger fonts
    fig.update_layout(
        title={
            'text': f'Length Distribution of GUI Element Referring Expressions',
            'font': {'size': 26, 'family': 'Arial, sans-serif', 'color': 'black'},
            'y': 0.87,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis={
            'title': {
                'text': 'Referring Expression Length',
                'font': {'size': 26, 'family': 'Arial, sans-serif'}
            },
            'tickfont': {'size': 24},
            'tickmode': 'array',
            'tickvals': list(range(0, args.max_length + 15, 15)),
            'ticktext': [str(i) for i in range(0, args.max_length + 15, 15)]
        },
        yaxis={
            'title': {
                'text': 'Percentage (%)',
                'font': {'size': 26, 'family': 'Arial, sans-serif'}
            },
            'tickfont': {'size': 24}
        },
        legend={
            'font': {'size': 24, 'family': 'Arial, sans-serif'},
            'x': 0.75,
            'y': 0.98,
            'bgcolor': 'rgba(255, 255, 255, 0.8)',
            'bordercolor': 'rgba(0, 0, 0, 0.2)',
            'borderwidth': 1,
            'orientation': 'v'
        },
        width=args.width,
        height=args.height,
        plot_bgcolor='white',
        hovermode='closest',
        bargap=0.05,
        margin=dict(l=80, r=80, t=120, b=80)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0, 0, 0, 0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0, 0, 0, 0.1)')
    
    # Add stats annotation
    stats_text = (
        f"Statistics:<br>"
        f"Mean: {mean_length:.2f}<br>"
        f"Median: {median_length:.2f}<br>"
    )
    
    # fig.add_annotation(
    #     xref="paper", yref="paper",
    #     x=0.71, y=0.98,
    #     text=stats_text,
    #     showarrow=False,
    #     font=dict(family="Arial, sans-serif", size=24),
    #     align="left",
    #     bgcolor="rgba(255, 255, 255, 0.8)",
    #     bordercolor="rgba(0, 0, 0, 0.2)",
    #     borderwidth=1,
    #     borderpad=6
    # )
    
    # Save the figure
    output_path = os.path.join(args.output_dir, os.path.basename(args.data_path).split('.')[0] + '_length_histogram.png')
    pio.write_image(fig, output_path, width=args.width, height=args.height, scale=args.dpi/100)
    
    # Also save as interactive HTML
    html_path = os.path.join(args.output_dir, os.path.basename(args.data_path).split('.')[0] + '_length_histogram.html')
    pio.write_html(fig, html_path)
    
    print(f"Histogram saved to {output_path}")
    print(f"Interactive HTML saved to {html_path}")
    
    # Print statistics
    print(f"Total expressions processed: {len(refexp_lengths)}")
    print(f"Mean expression length: {mean_length:.2f}")
    print(f"Median expression length: {median_length:.2f}")
    print(f"Min expression length: {min_length}")
    print(f"Max expression length: {max_length}")
    print(f"Standard deviation: {std_dev:.2f}")

if __name__ == "__main__":
    main()
