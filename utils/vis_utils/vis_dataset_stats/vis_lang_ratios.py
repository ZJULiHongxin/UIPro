import os
import sys
import json
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import argparse
from tqdm import tqdm
import re

# Add the parent directory to path to import misc functions
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from data_utils.misc import (
    contains_chinese, contains_japanese, contains_russian, 
    contains_arabic, contains_korean
)

def contains_english(text):
    # Simple check for English - contains ASCII letters and no other scripts
    return bool(re.search(r'[a-zA-Z]', text)) and not any([
        contains_chinese(text), contains_japanese(text), 
        contains_russian(text), contains_arabic(text), 
        contains_korean(text)
    ])

def main():
    parser = argparse.ArgumentParser(description='Visualize language distribution in dataset')
    parser.add_argument('--data_path', type=str, required=False, default="/mnt/nvme0n1p1/hongxin_li/UI_training_data/scaling_exp/UIPro_processed/UIPro_18643k.json", help='Path to the dataset file')
    parser.add_argument('--output_dir', type=str, default=os.path.join(os.path.dirname(__file__), './lang_ratios'), help='Output directory for visualization')
    parser.add_argument('--width', type=int, default=1200, help='Width of the output figure in pixels')
    parser.add_argument('--height', type=int, default=800, help='Height of the output figure in pixels')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for the output image')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the dataset
    print(f"Loading dataset from {args.data_path}...")
    with open(args.data_path, 'r') as f:
        dataset = json.load(f)
    
    # Process each sample in the dataset
    print(f"Processing {len(dataset)} samples...")
    
    # Initialize counters for each language
    lang_counts = {
        'English': 0,
        'Chinese': 0,
        'Japanese': 0,
        'Russian': 0,
        'Arabic': 0,
        'Korean': 0,
        'Mixed/Other': 0,
        'Total': 0
    }
    
    for sample in tqdm(dataset):
        # Extract conversations
        if 'conversations' not in sample:
            continue
        
        convs = sample['conversations']
        for conv in convs:
            text = conv['value'].replace("<image>", "").strip()
            lang_counts['Total'] += 1
            
            # Detect languages
            is_chinese = contains_chinese(text)
            is_japanese = contains_japanese(text)
            is_russian = contains_russian(text)
            is_arabic = contains_arabic(text)
            is_korean = contains_korean(text)
            is_english = contains_english(text)
            
            # Count language occurrences
            detected_count = sum([is_chinese, is_japanese, is_russian, is_arabic, is_korean, is_english])
            
            if detected_count == 0:
                lang_counts['Mixed/Other'] += 1
            elif detected_count > 1:
                lang_counts['Mixed/Other'] += 1
                # Also increment individual counters for mixed content
                if is_chinese: lang_counts['Chinese'] += 1
                if is_japanese: lang_counts['Japanese'] += 1
                if is_russian: lang_counts['Russian'] += 1
                if is_arabic: lang_counts['Arabic'] += 1
                if is_korean: lang_counts['Korean'] += 1
                if is_english: lang_counts['English'] += 1
            else:
                # Only one language detected
                if is_chinese: lang_counts['Chinese'] += 1
                elif is_japanese: lang_counts['Japanese'] += 1
                elif is_russian: lang_counts['Russian'] += 1
                elif is_arabic: lang_counts['Arabic'] += 1
                elif is_korean: lang_counts['Korean'] += 1
                elif is_english: lang_counts['English'] += 1
                else: lang_counts['Mixed/Other'] += 1
    
    # Calculate percentages
    total = lang_counts['Total']
    percentages = {}
    for lang, count in lang_counts.items():
        if lang != 'Total':
            percentages[lang] = (count / total) * 100
    
    # Sort languages by frequency
    sorted_langs = sorted(
        [lang for lang in lang_counts.keys() if lang != 'Total'],
        key=lambda x: lang_counts[x],
        reverse=True
    )
    
    # Create pie chart
    fig_pie = go.Figure(data=[go.Pie(
        labels=sorted_langs,
        values=[lang_counts[lang] for lang in sorted_langs],
        hole=0.4,
        marker=dict(
            colors=['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 
                    'rgb(214, 39, 40)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)', 
                    'rgb(227, 119, 194)']
        ),
        textinfo='label+percent',
        textfont=dict(size=16),
        insidetextorientation='radial'
    )])
    
    fig_pie.update_layout(
        title={
            'text': f'Language Distribution in Dataset<br>Total messages: {total}',
            'font': {'size': 24, 'family': 'Arial, sans-serif'},
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        legend={
            'font': {'size': 18, 'family': 'Arial, sans-serif'},
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': -0.2,
            'xanchor': 'center',
            'x': 0.5
        },
        width=args.width,
        height=args.height
    )
    
    # Bar chart for clarity
    fig_bar = go.Figure(data=[go.Bar(
        x=sorted_langs,
        y=[lang_counts[lang] for lang in sorted_langs],
        text=[f"{percentages[lang]:.2f}%" for lang in sorted_langs],
        textposition='auto',
        marker=dict(
            color=['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 
                   'rgb(214, 39, 40)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)', 
                   'rgb(227, 119, 194)']
        ),
        hovertemplate='%{x}: %{y} (%{text})<extra></extra>'
    )])
    
    fig_bar.update_layout(
        title={
            'text': f'Language Distribution in Dataset<br>Total messages: {total}',
            'font': {'size': 24, 'family': 'Arial, sans-serif'},
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis={
            'title': {
                'text': 'Language',
                'font': {'size': 20, 'family': 'Arial, sans-serif'}
            },
            'tickfont': {'size': 16}
        },
        yaxis={
            'title': {
                'text': 'Count',
                'font': {'size': 20, 'family': 'Arial, sans-serif'}
            },
            'tickfont': {'size': 16}
        },
        width=args.width,
        height=args.height,
        plot_bgcolor='white'
    )
    
    fig_bar.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0, 0, 0, 0.1)')
    fig_bar.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0, 0, 0, 0.1)')
    
    # Save figures
    base_filename = os.path.basename(args.data_path).split('.')[0]
    
    # Pie chart
    pie_path = os.path.join(args.output_dir, f'{base_filename}_lang_pie.png')
    pie_html_path = os.path.join(args.output_dir, f'{base_filename}_lang_pie.html')
    pio.write_image(fig_pie, pie_path, width=args.width, height=args.height, scale=args.dpi/100)
    pio.write_html(fig_pie, pie_html_path)
    
    # Bar chart
    bar_path = os.path.join(args.output_dir, f'{base_filename}_lang_bar.png')
    bar_html_path = os.path.join(args.output_dir, f'{base_filename}_lang_bar.html')
    pio.write_image(fig_bar, bar_path, width=args.width, height=args.height, scale=args.dpi/100)
    pio.write_html(fig_bar, bar_html_path)
    
    # Save raw data as JSON
    json_path = os.path.join(args.output_dir, f'{base_filename}_lang_stats.json')
    with open(json_path, 'w') as f:
        json.dump({
            'counts': lang_counts,
            'percentages': percentages
        }, f, indent=2)
    
    print(f"Pie chart saved to {pie_path}")
    print(f"Bar chart saved to {bar_path}")
    print(f"Interactive HTML files saved to {pie_html_path} and {bar_html_path}")
    print(f"Raw data saved to {json_path}")
    
    # Print statistics
    print("\nLanguage Distribution:")
    print(f"Total messages: {total}")
    for lang in sorted_langs:
        print(f"{lang}: {lang_counts[lang]} ({percentages[lang]:.2f}%)")
    
if __name__ == "__main__":
    main()
