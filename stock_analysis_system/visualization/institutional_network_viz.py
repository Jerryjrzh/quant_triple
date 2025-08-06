"""Institutional network visualization with force-directed graph layout and interactive exploration."""

import logging
import json
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from datetime import datetime, date
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import colorsys

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the institutional network."""
    INSTITUTION = "institution"
    STOCK = "stock"
    FUND = "fund"
    MANAGER = "manager"
    SECTOR = "sector"


class EdgeType(Enum):
    """Types of edges in the institutional network."""
    HOLDING = "holding"
    COLLABORATION = "collaboration"
    CORRELATION = "correlation"
    FLOW = "flow"
    SIMILARITY = "similarity"


@dataclass
class NetworkNode:
    """Network node data structure."""
    id: str
    label: str
    node_type: NodeType
    size: float = 10.0
    color: str = "#1f77b4"
    opacity: float = 1.0
    x: float = 0.0
    y: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    cluster_id: Optional[int] = None


@dataclass
class NetworkEdge:
    """Network edge data structure."""
    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0
    color: str = "#999999"
    opacity: float = 0.5
    width: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NetworkLayout:
    """Network layout configuration."""
    algorithm: str = "spring"  # spring, circular, kamada_kawai, etc.
    iterations: int = 50
    k: Optional[float] = None  # Optimal distance between nodes
    repulsion_strength: float = 1.0
    attraction_strength: float = 1.0
    center_gravity: float = 0.1


class InstitutionalNetworkVisualizer:
    """Advanced institutional network visualization system."""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.nodes: Dict[str, NetworkNode] = {}
        self.edges: List[NetworkEdge] = []
        self.layout_cache: Dict[str, Dict[str, Tuple[float, float]]] = {}
        self.color_schemes = {
            'institution_type': {
                'mutual_fund': '#1f77b4',
                'pension_fund': '#ff7f0e',
                'hedge_fund': '#2ca02c',
                'insurance': '#d62728',
                'bank': '#9467bd',
                'sovereign_fund': '#8c564b',
                'private_equity': '#e377c2',
                'other': '#7f7f7f'
            },
            'performance': ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd'],
            'risk': ['#fee5d9', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#99000d'],
            'size': ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']
        }
    
    def create_institutional_network(
        self,
        institutional_data: pd.DataFrame,
        holdings_data: pd.DataFrame,
        correlation_threshold: float = 0.5,
        min_holding_size: float = 0.01
    ) -> go.Figure:
        """Create comprehensive institutional network visualization."""
        
        # Build network graph
        self._build_network_graph(
            institutional_data, 
            holdings_data, 
            correlation_threshold, 
            min_holding_size
        )
        
        # Calculate layout
        layout = self._calculate_force_directed_layout()
        
        # Create visualization
        fig = self._create_network_figure(layout)
        
        # Add interactivity
        fig = self._add_network_interactivity(fig)
        
        return fig
    
    def create_holding_flow_network(
        self,
        flow_data: pd.DataFrame,
        time_window: str = "1M",
        min_flow_size: float = 1000000  # 1M minimum flow
    ) -> go.Figure:
        """Create network showing institutional holding flows over time."""
        
        # Process flow data
        flow_graph = self._build_flow_graph(flow_data, time_window, min_flow_size)
        
        # Calculate hierarchical layout for flow visualization
        layout = self._calculate_hierarchical_layout(flow_graph)
        
        # Create flow visualization
        fig = self._create_flow_network_figure(flow_graph, layout)
        
        return fig
    
    def create_sector_concentration_network(
        self,
        holdings_data: pd.DataFrame,
        sector_data: pd.DataFrame,
        concentration_threshold: float = 0.1
    ) -> go.Figure:
        """Create network showing institutional sector concentration patterns."""
        
        # Build sector concentration graph
        sector_graph = self._build_sector_concentration_graph(
            holdings_data, 
            sector_data, 
            concentration_threshold
        )
        
        # Use circular layout for sector visualization
        layout = self._calculate_circular_layout(sector_graph)
        
        # Create sector network figure
        fig = self._create_sector_network_figure(sector_graph, layout)
        
        return fig
    
    def create_correlation_network(
        self,
        correlation_matrix: pd.DataFrame,
        correlation_threshold: float = 0.7,
        max_connections: int = 100
    ) -> go.Figure:
        """Create network based on institutional correlation patterns."""
        
        # Build correlation graph
        corr_graph = self._build_correlation_graph(
            correlation_matrix, 
            correlation_threshold, 
            max_connections
        )
        
        # Calculate layout optimized for correlation visualization
        layout = self._calculate_correlation_layout(corr_graph)
        
        # Create correlation network figure
        fig = self._create_correlation_network_figure(corr_graph, layout)
        
        return fig
    
    def create_interactive_network_dashboard(
        self,
        institutional_data: pd.DataFrame,
        holdings_data: pd.DataFrame,
        flow_data: Optional[pd.DataFrame] = None
    ) -> go.Figure:
        """Create comprehensive interactive network dashboard."""
        
        # Create subplots for different network views
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Institutional Holdings Network',
                'Sector Concentration',
                'Correlation Network',
                'Flow Network'
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        # Build different network views
        holdings_net = self._build_holdings_network_traces(institutional_data, holdings_data)
        sector_net = self._build_sector_network_traces(institutional_data, holdings_data)
        corr_net = self._build_correlation_network_traces(institutional_data)
        
        # Add traces to subplots
        for trace in holdings_net:
            fig.add_trace(trace, row=1, col=1)
        
        for trace in sector_net:
            fig.add_trace(trace, row=1, col=2)
        
        for trace in corr_net:
            fig.add_trace(trace, row=2, col=1)
        
        if flow_data is not None:
            flow_net = self._build_flow_network_traces(flow_data)
            for trace in flow_net:
                fig.add_trace(trace, row=2, col=2)
        
        # Configure layout
        fig.update_layout(
            title=dict(
                text="Institutional Network Analysis Dashboard",
                x=0.5,
                font=dict(size=20)
            ),
            height=800,
            showlegend=False,
            hovermode='closest'
        )
        
        # Update subplot axes
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=i, col=j)
                fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=i, col=j)
        
        return fig
    
    def add_dynamic_filtering(
        self,
        fig: go.Figure,
        filter_options: Dict[str, List[str]]
    ) -> go.Figure:
        """Add dynamic filtering capabilities to network visualization."""
        
        # Create filter buttons
        filter_buttons = []
        
        for filter_name, options in filter_options.items():
            for option in options:
                filter_buttons.append(
                    dict(
                        args=[{"visible": self._get_visibility_for_filter(filter_name, option)}],
                        label=f"{filter_name}: {option}",
                        method="restyle"
                    )
                )
        
        # Add "Show All" button
        filter_buttons.insert(0, dict(
            args=[{"visible": True}],
            label="Show All",
            method="restyle"
        ))
        
        # Update layout with filter buttons
        fig.update_layout(
            updatemenus=[
                dict(
                    type="dropdown",
                    direction="down",
                    buttons=filter_buttons,
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.02,
                    yanchor="top"
                )
            ]
        )
        
        return fig
    
    def add_search_functionality(
        self,
        fig: go.Figure,
        searchable_fields: List[str] = None
    ) -> go.Figure:
        """Add search functionality to highlight specific nodes/edges."""
        
        if searchable_fields is None:
            searchable_fields = ['institution_name', 'stock_symbol', 'fund_name']
        
        # This would typically be implemented with custom JavaScript
        # For now, we'll add a placeholder annotation
        fig.add_annotation(
            text="Search functionality would be implemented with custom JavaScript callbacks",
            xref="paper", yref="paper",
            x=0.5, y=0.95,
            showarrow=False,
            font=dict(size=10, color="gray")
        )
        
        return fig
    
    def export_network_data(
        self,
        format: str = "json"
    ) -> Union[str, Dict[str, Any]]:
        """Export network data in various formats."""
        
        if format == "json":
            return {
                "nodes": [
                    {
                        "id": node.id,
                        "label": node.label,
                        "type": node.node_type.value,
                        "size": node.size,
                        "color": node.color,
                        "x": node.x,
                        "y": node.y,
                        "metadata": node.metadata,
                        "cluster_id": node.cluster_id
                    }
                    for node in self.nodes.values()
                ],
                "edges": [
                    {
                        "source": edge.source,
                        "target": edge.target,
                        "type": edge.edge_type.value,
                        "weight": edge.weight,
                        "color": edge.color,
                        "width": edge.width,
                        "metadata": edge.metadata
                    }
                    for edge in self.edges
                ]
            }
        
        elif format == "gexf":
            # Export as GEXF format for Gephi
            return nx.write_gexf(self.graph, None)
        
        elif format == "graphml":
            # Export as GraphML format
            return nx.write_graphml(self.graph, None)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _build_network_graph(
        self,
        institutional_data: pd.DataFrame,
        holdings_data: pd.DataFrame,
        correlation_threshold: float,
        min_holding_size: float
    ) -> None:
        """Build the network graph from institutional data."""
        
        # Clear existing graph
        self.graph.clear()
        self.nodes.clear()
        self.edges.clear()
        
        # Add institution nodes
        for _, institution in institutional_data.iterrows():
            node = NetworkNode(
                id=f"inst_{institution['institution_id']}",
                label=institution['institution_name'],
                node_type=NodeType.INSTITUTION,
                size=self._calculate_institution_size(institution),
                color=self._get_institution_color(institution),
                metadata={
                    'institution_type': institution.get('institution_type', 'other'),
                    'total_assets': institution.get('total_assets', 0),
                    'performance': institution.get('performance', 0)
                }
            )
            self.nodes[node.id] = node
            self.graph.add_node(node.id, **node.metadata)
        
        # Add stock nodes and holding edges
        stock_nodes = set()
        for _, holding in holdings_data.iterrows():
            if holding['holding_percentage'] >= min_holding_size:
                # Add stock node if not exists
                stock_id = f"stock_{holding['stock_code']}"
                if stock_id not in stock_nodes:
                    stock_node = NetworkNode(
                        id=stock_id,
                        label=holding['stock_name'],
                        node_type=NodeType.STOCK,
                        size=self._calculate_stock_size(holding),
                        color=self._get_stock_color(holding),
                        metadata={
                            'stock_code': holding['stock_code'],
                            'sector': holding.get('sector', 'unknown'),
                            'market_cap': holding.get('market_cap', 0)
                        }
                    )
                    self.nodes[stock_node.id] = stock_node
                    self.graph.add_node(stock_node.id, **stock_node.metadata)
                    stock_nodes.add(stock_id)
                
                # Add holding edge
                inst_id = f"inst_{holding['institution_id']}"
                edge = NetworkEdge(
                    source=inst_id,
                    target=stock_id,
                    edge_type=EdgeType.HOLDING,
                    weight=holding['holding_percentage'],
                    width=self._calculate_edge_width(holding['holding_percentage']),
                    metadata={
                        'holding_value': holding.get('holding_value', 0),
                        'holding_percentage': holding['holding_percentage']
                    }
                )
                self.edges.append(edge)
                self.graph.add_edge(inst_id, stock_id, weight=edge.weight, **edge.metadata)
        
        # Add correlation edges between institutions
        self._add_correlation_edges(institutional_data, correlation_threshold)
    
    def _add_correlation_edges(
        self,
        institutional_data: pd.DataFrame,
        correlation_threshold: float
    ) -> None:
        """Add correlation edges between institutions."""
        
        # Calculate correlation matrix (simplified)
        institutions = list(institutional_data['institution_id'])
        
        for i, inst1 in enumerate(institutions):
            for j, inst2 in enumerate(institutions[i+1:], i+1):
                # Calculate correlation (placeholder - would use actual correlation calculation)
                correlation = np.random.uniform(-1, 1)  # Placeholder
                
                if abs(correlation) >= correlation_threshold:
                    edge = NetworkEdge(
                        source=f"inst_{inst1}",
                        target=f"inst_{inst2}",
                        edge_type=EdgeType.CORRELATION,
                        weight=abs(correlation),
                        color='red' if correlation < 0 else 'green',
                        opacity=0.3,
                        width=abs(correlation) * 3,
                        metadata={'correlation': correlation}
                    )
                    self.edges.append(edge)
                    self.graph.add_edge(
                        f"inst_{inst1}", 
                        f"inst_{inst2}", 
                        weight=edge.weight, 
                        **edge.metadata
                    )
    
    def _calculate_force_directed_layout(
        self,
        layout_config: NetworkLayout = None
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate force-directed layout for the network."""
        
        if layout_config is None:
            layout_config = NetworkLayout()
        
        # Check cache first
        cache_key = f"{layout_config.algorithm}_{layout_config.iterations}_{len(self.graph.nodes)}"
        if cache_key in self.layout_cache:
            return self.layout_cache[cache_key]
        
        # Calculate layout based on algorithm
        if layout_config.algorithm == "spring":
            pos = nx.spring_layout(
                self.graph,
                k=layout_config.k,
                iterations=layout_config.iterations,
                weight='weight'
            )
        elif layout_config.algorithm == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self.graph, weight='weight')
        elif layout_config.algorithm == "circular":
            pos = nx.circular_layout(self.graph)
        elif layout_config.algorithm == "random":
            pos = nx.random_layout(self.graph)
        else:
            # Default to spring layout
            pos = nx.spring_layout(self.graph, iterations=layout_config.iterations)
        
        # Cache the layout
        self.layout_cache[cache_key] = pos
        
        # Update node positions
        for node_id, (x, y) in pos.items():
            if node_id in self.nodes:
                self.nodes[node_id].x = x
                self.nodes[node_id].y = y
        
        return pos
    
    def _create_network_figure(
        self,
        layout: Dict[str, Tuple[float, float]]
    ) -> go.Figure:
        """Create the main network visualization figure."""
        
        fig = go.Figure()
        
        # Add edges
        edge_traces = self._create_edge_traces(layout)
        for trace in edge_traces:
            fig.add_trace(trace)
        
        # Add nodes
        node_traces = self._create_node_traces(layout)
        for trace in node_traces:
            fig.add_trace(trace)
        
        # Configure layout
        fig.update_layout(
            title=dict(
                text="Institutional Network Visualization",
                x=0.5,
                font=dict(size=18)
            ),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Institutional Network Analysis<br>Node size: Total assets | Edge width: Holding percentage",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(size=10, color="gray")
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def _create_edge_traces(
        self,
        layout: Dict[str, Tuple[float, float]]
    ) -> List[go.Scatter]:
        """Create edge traces for the network visualization."""
        
        traces = []
        
        # Group edges by type for better visualization
        edge_groups = {}
        for edge in self.edges:
            edge_type = edge.edge_type.value
            if edge_type not in edge_groups:
                edge_groups[edge_type] = []
            edge_groups[edge_type].append(edge)
        
        # Create trace for each edge type
        for edge_type, edges in edge_groups.items():
            x_coords = []
            y_coords = []
            edge_info = []
            
            for edge in edges:
                if edge.source in layout and edge.target in layout:
                    x0, y0 = layout[edge.source]
                    x1, y1 = layout[edge.target]
                    
                    x_coords.extend([x0, x1, None])
                    y_coords.extend([y0, y1, None])
                    
                    edge_info.append({
                        'source': edge.source,
                        'target': edge.target,
                        'weight': edge.weight,
                        'type': edge.edge_type.value
                    })
            
            if x_coords:  # Only create trace if there are edges
                trace = go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    line=dict(
                        width=0.5 if edge_type == 'correlation' else 1.0,
                        color=edges[0].color if edges else '#999999'
                    ),
                    hoverinfo='none',
                    showlegend=True,
                    name=f"{edge_type.title()} Connections",
                    opacity=0.6
                )
                traces.append(trace)
        
        return traces
    
    def _create_node_traces(
        self,
        layout: Dict[str, Tuple[float, float]]
    ) -> List[go.Scatter]:
        """Create node traces for the network visualization."""
        
        traces = []
        
        # Group nodes by type
        node_groups = {}
        for node in self.nodes.values():
            node_type = node.node_type.value
            if node_type not in node_groups:
                node_groups[node_type] = []
            node_groups[node_type].append(node)
        
        # Create trace for each node type
        for node_type, nodes in node_groups.items():
            x_coords = []
            y_coords = []
            sizes = []
            colors = []
            texts = []
            hover_texts = []
            
            for node in nodes:
                if node.id in layout:
                    x, y = layout[node.id]
                    x_coords.append(x)
                    y_coords.append(y)
                    sizes.append(node.size)
                    colors.append(node.color)
                    texts.append(node.label)
                    
                    # Create hover text
                    hover_info = [f"<b>{node.label}</b>"]
                    hover_info.append(f"Type: {node.node_type.value}")
                    
                    for key, value in node.metadata.items():
                        if isinstance(value, (int, float)):
                            if key in ['total_assets', 'market_cap', 'holding_value']:
                                hover_info.append(f"{key.replace('_', ' ').title()}: ${value:,.0f}")
                            elif key in ['performance', 'holding_percentage']:
                                hover_info.append(f"{key.replace('_', ' ').title()}: {value:.2%}")
                            else:
                                hover_info.append(f"{key.replace('_', ' ').title()}: {value:.2f}")
                        else:
                            hover_info.append(f"{key.replace('_', ' ').title()}: {value}")
                    
                    hover_texts.append("<br>".join(hover_info))
            
            if x_coords:  # Only create trace if there are nodes
                trace = go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='markers+text',
                    marker=dict(
                        size=sizes,
                        color=colors,
                        opacity=0.8,
                        line=dict(width=1, color='white')
                    ),
                    text=texts,
                    textposition="middle center",
                    textfont=dict(size=8, color='white'),
                    hovertext=hover_texts,
                    hoverinfo='text',
                    showlegend=True,
                    name=f"{node_type.title()} Nodes"
                )
                traces.append(trace)
        
        return traces
    
    def _add_network_interactivity(self, fig: go.Figure) -> go.Figure:
        """Add interactive features to the network visualization."""
        
        # Add layout algorithm selector
        layout_buttons = [
            dict(
                args=[{"visible": True}],
                label="Spring Layout",
                method="restyle"
            ),
            dict(
                args=[{"visible": True}],
                label="Circular Layout", 
                method="restyle"
            ),
            dict(
                args=[{"visible": True}],
                label="Kamada-Kawai Layout",
                method="restyle"
            )
        ]
        
        fig.update_layout(
            updatemenus=[
                dict(
                    type="dropdown",
                    direction="down",
                    buttons=layout_buttons,
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.02,
                    yanchor="top"
                )
            ]
        )
        
        return fig
    
    def _build_flow_graph(
        self,
        flow_data: pd.DataFrame,
        time_window: str,
        min_flow_size: float
    ) -> nx.DiGraph:
        """Build directed graph for flow visualization."""
        
        flow_graph = nx.DiGraph()
        
        # Process flow data (simplified implementation)
        for _, flow in flow_data.iterrows():
            if flow['flow_amount'] >= min_flow_size:
                source = f"inst_{flow['source_institution']}"
                target = f"inst_{flow['target_institution']}"
                
                flow_graph.add_edge(
                    source,
                    target,
                    weight=flow['flow_amount'],
                    flow_type=flow.get('flow_type', 'unknown')
                )
        
        return flow_graph
    
    def _calculate_hierarchical_layout(
        self,
        graph: nx.DiGraph
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate hierarchical layout for flow visualization."""
        
        try:
            # Use graphviz layout if available
            pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
        except:
            # Fallback to spring layout
            pos = nx.spring_layout(graph)
        
        return pos
    
    def _calculate_circular_layout(
        self,
        graph: nx.Graph
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate circular layout for sector visualization."""
        
        return nx.circular_layout(graph)
    
    def _calculate_correlation_layout(
        self,
        graph: nx.Graph
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate layout optimized for correlation visualization."""
        
        # Use force-directed layout with correlation-based weights
        return nx.spring_layout(graph, weight='correlation', k=2, iterations=100)
    
    def _build_correlation_graph(
        self,
        correlation_matrix: pd.DataFrame,
        correlation_threshold: float,
        max_connections: int
    ) -> nx.Graph:
        """Build correlation graph from correlation matrix."""
        
        corr_graph = nx.Graph()
        
        # Add nodes
        for institution in correlation_matrix.index:
            corr_graph.add_node(institution)
        
        # Add edges based on correlation
        edges_added = 0
        correlations = []
        
        # Get all correlations above threshold
        for i, inst1 in enumerate(correlation_matrix.index):
            for j, inst2 in enumerate(correlation_matrix.columns[i+1:], i+1):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) >= correlation_threshold:
                    correlations.append((inst1, inst2, abs(corr_value), corr_value))
        
        # Sort by correlation strength and add top connections
        correlations.sort(key=lambda x: x[2], reverse=True)
        
        for inst1, inst2, abs_corr, corr_value in correlations[:max_connections]:
            corr_graph.add_edge(
                inst1,
                inst2,
                weight=abs_corr,
                correlation=corr_value
            )
            edges_added += 1
        
        return corr_graph
    
    def _calculate_institution_size(self, institution: pd.Series) -> float:
        """Calculate node size based on institution characteristics."""
        
        # Base size on total assets (normalized)
        assets = institution.get('total_assets', 0)
        if assets > 0:
            # Log scale for better visualization
            return max(10, min(50, 10 + 40 * np.log10(assets) / np.log10(1e12)))
        return 10
    
    def _calculate_stock_size(self, holding: pd.Series) -> float:
        """Calculate stock node size based on market cap or holding value."""
        
        market_cap = holding.get('market_cap', 0)
        if market_cap > 0:
            return max(8, min(30, 8 + 22 * np.log10(market_cap) / np.log10(1e12)))
        return 8
    
    def _calculate_edge_width(self, weight: float) -> float:
        """Calculate edge width based on weight."""
        
        return max(0.5, min(5.0, weight * 10))
    
    def _get_institution_color(self, institution: pd.Series) -> str:
        """Get color for institution node based on type."""
        
        inst_type = institution.get('institution_type', 'other')
        return self.color_schemes['institution_type'].get(inst_type, '#7f7f7f')
    
    def _get_stock_color(self, holding: pd.Series) -> str:
        """Get color for stock node based on sector."""
        
        sector = holding.get('sector', 'unknown')
        # Use a hash-based color assignment for sectors
        hash_value = hash(sector) % len(px.colors.qualitative.Set3)
        return px.colors.qualitative.Set3[hash_value]
    
    def _get_visibility_for_filter(
        self,
        filter_name: str,
        filter_value: str
    ) -> List[bool]:
        """Get visibility array for filtering."""
        
        # This would return a boolean array indicating which traces to show
        # Simplified implementation
        return [True] * 10  # Placeholder
    
    def _build_holdings_network_traces(
        self,
        institutional_data: pd.DataFrame,
        holdings_data: pd.DataFrame
    ) -> List[go.Scatter]:
        """Build traces for holdings network subplot."""
        
        # Simplified implementation for subplot
        return [
            go.Scatter(
                x=[0, 1, 0.5],
                y=[0, 0, 1],
                mode='markers',
                marker=dict(size=20, color='blue'),
                name='Holdings Network'
            )
        ]
    
    def _build_sector_network_traces(
        self,
        institutional_data: pd.DataFrame,
        holdings_data: pd.DataFrame
    ) -> List[go.Scatter]:
        """Build traces for sector network subplot."""
        
        # Simplified implementation for subplot
        return [
            go.Scatter(
                x=[0, 1, 0.5],
                y=[0, 0, 1],
                mode='markers',
                marker=dict(size=15, color='green'),
                name='Sector Network'
            )
        ]
    
    def _build_correlation_network_traces(
        self,
        institutional_data: pd.DataFrame
    ) -> List[go.Scatter]:
        """Build traces for correlation network subplot."""
        
        # Simplified implementation for subplot
        return [
            go.Scatter(
                x=[0, 1, 0.5],
                y=[0, 0, 1],
                mode='markers',
                marker=dict(size=18, color='red'),
                name='Correlation Network'
            )
        ]
    
    def _build_flow_network_traces(
        self,
        flow_data: pd.DataFrame
    ) -> List[go.Scatter]:
        """Build traces for flow network subplot."""
        
        # Simplified implementation for subplot
        return [
            go.Scatter(
                x=[0, 1, 0.5],
                y=[0, 0, 1],
                mode='markers',
                marker=dict(size=16, color='orange'),
                name='Flow Network'
            )
        ]


# Utility functions for institutional network visualization
def create_simple_institutional_network(
    institutions: List[str],
    holdings: Dict[str, List[str]],
    **kwargs
) -> go.Figure:
    """Create a simple institutional network from basic data."""
    
    visualizer = InstitutionalNetworkVisualizer()
    
    # Convert simple data to DataFrame format
    institutional_data = pd.DataFrame({
        'institution_id': range(len(institutions)),
        'institution_name': institutions,
        'institution_type': ['mutual_fund'] * len(institutions),
        'total_assets': [1e9] * len(institutions)  # Placeholder
    })
    
    holdings_data = []
    for inst_id, stocks in holdings.items():
        for stock in stocks:
            holdings_data.append({
                'institution_id': institutions.index(inst_id),
                'stock_code': stock,
                'stock_name': stock,
                'holding_percentage': 0.05,  # Placeholder
                'sector': 'technology'  # Placeholder
            })
    
    holdings_df = pd.DataFrame(holdings_data)
    
    return visualizer.create_institutional_network(
        institutional_data,
        holdings_df,
        **kwargs
    )


def analyze_network_metrics(
    visualizer: InstitutionalNetworkVisualizer
) -> Dict[str, Any]:
    """Analyze network metrics for the institutional network."""
    
    graph = visualizer.graph
    
    if len(graph.nodes) == 0:
        return {}
    
    metrics = {
        'num_nodes': len(graph.nodes),
        'num_edges': len(graph.edges),
        'density': nx.density(graph),
        'average_clustering': nx.average_clustering(graph),
        'num_connected_components': nx.number_connected_components(graph)
    }
    
    # Calculate centrality measures
    try:
        metrics['degree_centrality'] = nx.degree_centrality(graph)
        metrics['betweenness_centrality'] = nx.betweenness_centrality(graph)
        metrics['closeness_centrality'] = nx.closeness_centrality(graph)
        metrics['eigenvector_centrality'] = nx.eigenvector_centrality(graph, max_iter=1000)
    except:
        logger.warning("Could not calculate all centrality measures")
    
    return metrics