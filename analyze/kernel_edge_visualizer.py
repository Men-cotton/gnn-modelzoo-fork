import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
import numpy as np

class KernelEdgeVisualizer:
    def __init__(self):
        pass

    def get_pe_coords(self, data):
        edges = data.get('edges', [])
        x_coords = []
        y_coords = []
        labels = []
        
        for edge in edges:
            # Source PEs
            source_pes = edge.get('source_port_pes', [])
            for item in source_pes:
                pe = item.get('pe')
                if pe:
                    x_coords.append(pe.get('x'))
                    y_coords.append(pe.get('y'))
                    labels.append(edge.get('source_name', ''))
            
            # Target PEs
            target_pes = edge.get('target_port_pes', [])
            for item in target_pes:
                 pe = item.get('pe')
                 if pe:
                    x_coords.append(pe.get('x'))
                    y_coords.append(pe.get('y'))
                    labels.append(edge.get('target_name', ''))
        
        return x_coords, y_coords, labels

    def draw_graph_edges(self, data):
        """Visualizes the kernel graph focusing on edges (x < 30 or x > 720)."""
        x, y, labels = self.get_pe_coords(data)
        if not x:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No PE data", ha='center')
            return fig

        # Filter by thresholds
        x_left = []
        y_left = []
        
        x_right = []
        y_right = []
        
        for i, val in enumerate(x):
            if val < 30:
                x_left.append(val)
                y_left.append(y[i])
            elif val > 720:
                x_right.append(val)
                y_right.append(y[i])

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 10))
        
        # Plot Left
        if x_left:
            ax1.scatter(x_left, y_left, alpha=0.6, s=20, c='blue', edgecolors='none')
            ax1.set_title("Left Edge (x < 30)")
            ax1.set_xlabel("PE X")
            ax1.set_ylabel("PE Y")
            ax1.grid(True, linestyle='--', alpha=0.3)
        else:
            ax1.text(0.5, 0.5, "No data for x < 30", ha='center')

        # Plot Right
        if x_right:
            ax2.scatter(x_right, y_right, alpha=0.6, s=20, c='green', edgecolors='none')
            ax2.set_title("Right Edge (x > 720)")
            ax2.set_xlabel("PE X")
            ax2.grid(True, linestyle='--', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "No data for x > 720", ha='center')

        fig.suptitle("Kernel Graph - Edge Details (x < 30, x > 720)")
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

    def get_color(self, name, parent_color='blue'):
        name = name.lower()
        for key, color in self.TYPE_COLORS.items():
            if key in name and key != 'other':
                return color
        return parent_color

    def get_rects(self, node, rects_list, parent_color='blue'):
        name = node.get('name', '')
        color = self.get_color(name, parent_color)
        
        rect_data = node.get('rect')
        if rect_data:
            rects_list.append({
                'x': rect_data.get('x', 0),
                'y': rect_data.get('y', 0),
                'w': rect_data.get('wd', 0),
                'h': rect_data.get('ht', 0),
                'name': name,
                'color': color
            })
        
        for child in node.get('children', []):
            self.get_rects(child, rects_list, color)

    def draw_tree_edges(self, data):
        """Visualizes the kernel tree focusing on edges (x < 30 or x > 720)."""
        root = data.get('root')
        if not root:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No root data", ha='center')
            return fig

        rects = []
        self.get_rects(root, rects)
        
        if not rects:
             fig, ax = plt.subplots()
             ax.text(0.5, 0.5, "No rect data", ha='center')
             return fig

        # Find global max width to set right limit
        max_x_global = max([r['x'] + r['w'] for r in rects]) if rects else 762

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 10))

        # Helper to draw on ax
        def plot_rects_on_ax(ax, rect_subset, title, x_limit):
            ax.set_title(title)
            ax.set_xlabel("X")
            if ax == ax1: ax.set_ylabel("Y")
            
            if not rect_subset:
                ax.text(0.5, 0.5, "No data", ha='center')
                # Still set limits to show the empty region
                ax.set_xlim(x_limit)
                return

            for r in rect_subset:
                color = r.get('color', 'blue')
                rect = patches.Rectangle((r['x'], r['y']), r['w'], r['h'], 
                                       linewidth=1, edgecolor=color, facecolor=color, alpha=0.1)
                ax.add_patch(rect)
                rect_border = patches.Rectangle((r['x'], r['y']), r['w'], r['h'], 
                                       linewidth=1, edgecolor=color, facecolor='none', alpha=0.8)
                ax.add_patch(rect_border)
                
                # No text label as requested
            
            # Set limits explicitly
            ax.set_xlim(x_limit)
            
            # Set Y limits based on content (or global?)
            # Let's use the subset Y range
            sub_ys = [r['y'] for r in rect_subset] + [r['y']+r['h'] for r in rect_subset]
            if sub_ys:
                ax.set_ylim(min(sub_ys)-1, max(sub_ys)+1)
                # ax.invert_yaxis()
            ax.grid(True, linestyle=':', alpha=0.3)

        # Filter rects based on OVERLAP with regions
        # Left region: [0, 30]
        # Overlap: r.x < 30 AND r.x + r.w > 0
        left_rects = [r for r in rects if r['x'] < 30 and (r['x'] + r['w'] > 0)]
        
        # Right region: [720, max_x_global]
        # Overlap: r.x < max_x_global AND r.x + r.w > 720
        right_rects = [r for r in rects if r['x'] < max_x_global and (r['x'] + r['w'] > 720)]

        plot_rects_on_ax(ax1, left_rects, "Left Edge Tree (0-30)", (0, 30))
        plot_rects_on_ax(ax2, right_rects, f"Right Edge Tree (720-{max_x_global})", (720, max_x_global))

        fig.suptitle("Kernel Tree - Edge Details (Zoomed)")
        
        # Add Legend to Figure
        legend_elements = [patches.Patch(facecolor=color, edgecolor=color, alpha=0.5, label=key.upper())
                           for key, color in self.TYPE_COLORS.items()]
        # Place legend at the bottom
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
                   ncol=5, title="Kernel Types")
        
        # Adjust layout to make room for legend
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        return fig

def main():
    graph_path = "log/20251106/log-export-wsjob-jxy2dzzgicat7flgf9c3an-1e523ff4/cs_16392203599647071561/kernel_graph.json"
    tree_path = "log/20251106/log-export-wsjob-jxy2dzzgicat7flgf9c3an-1e523ff4/cs_16392203599647071561/kernel_tree.json"
    
    output_dir = "analyze"
    
    viz = KernelEdgeVisualizer()
    
    # Process Graph
    if os.path.exists(graph_path):
        print(f"Processing {graph_path}...")
        with open(graph_path, 'r') as f:
            graph_data = json.load(f)
        fig_graph = viz.draw_graph_edges(graph_data)
        output_graph = os.path.join(output_dir, "kernel_graph_edge.png")
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
        fig_tree = viz.draw_tree_edges(tree_data)
        output_tree = os.path.join(output_dir, "kernel_tree_edge.png")
        fig_tree.savefig(output_tree)
        print(f"Saved {output_tree}")
        plt.close(fig_tree)
    else:
        print(f"File not found: {tree_path}")

if __name__ == "__main__":
    main()
