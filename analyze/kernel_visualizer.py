import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys

class KernelVisualizer:
    def __init__(self):
        pass

    def draw_graph(self, data):
        """Visualizes the kernel graph."""
        fig, ax = plt.subplots(figsize=(12, 12))
        
        edges = data.get('edges', [])
        if not edges:
            ax.text(0.5, 0.5, "No edges found", ha='center', va='center')
            return fig

        # Extract PE coordinates
        x_coords = []
        y_coords = []
        
        for edge in edges:
            source_pes = edge.get('source_port_pes', [])
            for item in source_pes:
                pe = item.get('pe')
                if pe:
                    x_coords.append(pe.get('x'))
                    y_coords.append(pe.get('y'))
            
            # Also check target_port_pes if available (based on structure seen in similar logs, though not explicitly shown in head)
            target_pes = edge.get('target_port_pes', [])
            for item in target_pes:
                 pe = item.get('pe')
                 if pe:
                    x_coords.append(pe.get('x'))
                    y_coords.append(pe.get('y'))

        if x_coords and y_coords:
            ax.scatter(x_coords, y_coords, alpha=0.5, s=10)
            ax.set_xlabel("PE X")
            ax.set_ylabel("PE Y")
            ax.set_title("Kernel Graph PE Distribution")
            ax.grid(True, linestyle='--', alpha=0.3)
        else:
             ax.text(0.5, 0.5, "No PE coordinates found in edges", ha='center', va='center')

        return fig

    def draw_tree(self, data):
        """Visualizes the kernel tree."""
        fig, ax = plt.subplots(figsize=(12, 12))
        
        root = data.get('root')
        if not root:
             ax.text(0.5, 0.5, "No root found", ha='center', va='center')
             return fig

        TYPE_COLORS = {
            'wgt': 'orange',
            'grd': 'red',
            'act': 'green',
            'io': 'purple',
            'buf': 'brown',
            'ctx': 'cyan',
            'config': 'gray',
            'adapter': 'pink',
            'crc': 'yellow',
            'other': 'blue'
        }

        def get_color(name, parent_color='blue'):
            name = name.lower()
            for key, color in TYPE_COLORS.items():
                if key in name and key != 'other':
                    return color
            return parent_color

        # Recursive function to draw rectangles
        def draw_node(node, ax, parent_color='blue'):
            name = node.get('name', '')
            color = get_color(name, parent_color)
            
            rect_data = node.get('rect')
            if rect_data:
                x = rect_data.get('x', 0)
                y = rect_data.get('y', 0)
                w = rect_data.get('wd', 0)
                h = rect_data.get('ht', 0)
                
                # Draw rectangle
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor=color, alpha=0.1)
                ax.add_patch(rect)
                # Add a stronger border
                rect_border = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor='none', alpha=0.8)
                ax.add_patch(rect_border)

            children = node.get('children', [])
            for child in children:
                draw_node(child, ax, color)

        draw_node(root, ax)
        
        # Set limits based on root rect if available
        root_rect = root.get('rect')
        if root_rect:
            ax.set_xlim(root_rect.get('x', 0), root_rect.get('x', 0) + root_rect.get('wd', 100))
            ax.set_ylim(root_rect.get('y', 0), root_rect.get('y', 0) + root_rect.get('ht', 100))
        else:
            ax.autoscale()

        ax.set_title("Kernel Tree Layout")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Add Legend
        legend_elements = [patches.Patch(facecolor=color, edgecolor=color, alpha=0.5, label=key.upper())
                           for key, color in TYPE_COLORS.items()]
        # Place legend below the plot
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), 
                  ncol=5, title="Kernel Types")
        
        # Adjust layout to make room for legend
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        return fig
        return fig

def main():
    graph_path = "log/20251106/log-export-wsjob-jxy2dzzgicat7flgf9c3an-1e523ff4/cs_16392203599647071561/kernel_graph.json"
    tree_path = "log/20251106/log-export-wsjob-jxy2dzzgicat7flgf9c3an-1e523ff4/cs_16392203599647071561/kernel_tree.json"
    
    output_dir = "analyze"
    
    viz = KernelVisualizer()
    
    # Process Graph
    if os.path.exists(graph_path):
        print(f"Processing {graph_path}...")
        with open(graph_path, 'r') as f:
            graph_data = json.load(f)
        fig_graph = viz.draw_graph(graph_data)
        output_graph = os.path.join(output_dir, "kernel_graph.png")
        fig_graph.savefig(output_graph)
        print(f"Saved {output_graph}")
        plt.close(fig_graph)
    else:
        print(f"File not found: {graph_path}")

    # Process Tree
    if os.path.exists(tree_path):
        print(f"Processing {tree_path}...")
        with open(tree_path, 'r') as f:
            tree_data = json.load(f)
        fig_tree = viz.draw_tree(tree_data)
        output_tree = os.path.join(output_dir, "kernel_tree.png")
        fig_tree.savefig(output_tree)
        print(f"Saved {output_tree}")
        plt.close(fig_tree)
    else:
        print(f"File not found: {tree_path}")

if __name__ == "__main__":
    main()
