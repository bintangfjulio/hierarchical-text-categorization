"""
Hierarchy structure management for multi-level classification.
"""
from typing import Dict, List, Set, Tuple
from pathlib import Path
import pandas as pd
from collections import defaultdict


class HierarchyManager:
    """Manages category hierarchy structure."""
    
    def __init__(self, tree_file: Path):
        """
        Initialize hierarchy manager.
        
        Args:
            tree_file: Path to hierarchy tree file
        """
        self.tree_file = Path(tree_file)
        self.level_on_nodes: Dict[int, List[str]] = {}
        self.idx_on_section: Dict[int, List[str]] = {}
        self.section_on_idx: Dict[str, int] = {}
        self.section_parent_child: Dict[str, Set[str]] = {}
        self._is_loaded = False
    
    @classmethod
    def create_from_dataset(cls, dataset: pd.DataFrame, tree_file: Path) -> 'HierarchyManager':
        """
        Create hierarchy tree file from dataset.
        
        Args:
            dataset: DataFrame with hierarchy columns
            tree_file: Path to save tree file
            
        Returns:
            HierarchyManager instance
        """
        tree_file = Path(tree_file)
        tree_file.parent.mkdir(parents=True, exist_ok=True)
        
        hierarchy_paths = set()
        
        # Extract all hierarchy paths from dataset
        for col_idx, col in enumerate(dataset.columns):
            if col_idx > 0:  # Skip first column (text)
                for value in dataset[col].dropna().unique():
                    hierarchy_paths.add(str(value))
        
        # Sort and save
        hierarchy_paths = sorted(hierarchy_paths)
        
        with open(tree_file, 'w', encoding='utf-8') as f:
            for path in hierarchy_paths:
                f.write(f"{path}\n")
        
        return cls(tree_file)
    
    def load_hierarchy(self) -> None:
        """Load and parse hierarchy from tree file."""
        if self._is_loaded:
            return
        
        if not self.tree_file.exists():
            raise FileNotFoundError(f"Tree file not found: {self.tree_file}")
        
        section_parent_child = defaultdict(set)
        level_on_nodes = defaultdict(list)
        
        with open(self.tree_file, 'r', encoding='utf-8') as f:
            for line in f:
                path = line.strip()
                if not path:
                    continue
                
                nodes = [node.lower() for node in path.split(" > ")]
                
                # Build parent-child relationships
                for level, node in enumerate(nodes):
                    if level > 0:
                        parent = nodes[level - 1]
                        section_parent_child[parent].add(node)
                
                # Track nodes at each level
                level = len(nodes) - 1
                last_node = nodes[-1]
                level_on_nodes[level].append(last_node)
        
        # Add root section
        root_children = set(level_on_nodes[0]) if 0 in level_on_nodes else set()
        section_parent_child = {'root': root_children, **section_parent_child}
        
        # Create section-index mappings
        idx_on_section = {}
        section_on_idx = {}
        
        for idx, (parent, children) in enumerate(section_parent_child.items()):
            children_list = sorted(list(children))
            idx_on_section[idx] = children_list
            
            for child in children_list:
                section_on_idx[child] = idx
        
        # Sort nodes at each level
        for level in level_on_nodes:
            level_on_nodes[level] = sorted(list(set(level_on_nodes[level])))
        
        self.level_on_nodes = dict(level_on_nodes)
        self.idx_on_section = idx_on_section
        self.section_on_idx = section_on_idx
        self.section_parent_child = dict(section_parent_child)
        self._is_loaded = True
    
    def get_num_levels(self) -> int:
        """Get number of hierarchy levels."""
        if not self._is_loaded:
            self.load_hierarchy()
        return len(self.level_on_nodes)
    
    def get_num_classes_at_level(self, level: int) -> int:
        """Get number of classes at specific level."""
        if not self._is_loaded:
            self.load_hierarchy()
        return len(self.level_on_nodes.get(level, []))
    
    def get_num_classes_in_section(self, section: int) -> int:
        """Get number of classes in specific section."""
        if not self._is_loaded:
            self.load_hierarchy()
        return len(self.idx_on_section.get(section, []))
    
    def get_children_of_node(self, node: str) -> List[str]:
        """Get children nodes of given node."""
        if not self._is_loaded:
            self.load_hierarchy()
        return list(self.section_parent_child.get(node, []))
    
    def get_section_of_node(self, node: str) -> int:
        """Get section index of given node."""
        if not self._is_loaded:
            self.load_hierarchy()
        return self.section_on_idx.get(node, -1)
    
    def get_all_leaf_nodes(self) -> List[str]:
        """Get all leaf nodes (deepest level)."""
        if not self._is_loaded:
            self.load_hierarchy()
        max_level = max(self.level_on_nodes.keys())
        return self.level_on_nodes[max_level]
    
    def validate(self) -> bool:
        """Validate hierarchy structure."""
        if not self._is_loaded:
            self.load_hierarchy()
        
        # Check that all nodes have a section
        all_nodes = set()
        for nodes in self.level_on_nodes.values():
            all_nodes.update(nodes)
        
        for node in all_nodes:
            if node not in self.section_on_idx:
                raise ValueError(f"Node {node} not in section mapping")
        
        return True
    
    def get_hierarchy_info(self) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Get all hierarchy information.
        
        Returns:
            Tuple of (level_on_nodes, idx_on_section, section_on_idx, section_parent_child)
        """
        if not self._is_loaded:
            self.load_hierarchy()
        
        return (
            self.level_on_nodes,
            self.idx_on_section,
            self.section_on_idx,
            self.section_parent_child
        )
    
    def print_hierarchy(self) -> None:
        """Print hierarchy structure for debugging."""
        if not self._is_loaded:
            self.load_hierarchy()
        
        print("\n" + "="*50)
        print("HIERARCHY STRUCTURE")
        print("="*50)
        
        print(f"\nNumber of levels: {len(self.level_on_nodes)}")
        for level, nodes in sorted(self.level_on_nodes.items()):
            print(f"\nLevel {level}: {len(nodes)} classes")
            print(f"  Classes: {', '.join(nodes[:5])}" + 
                  (f" ... (+{len(nodes)-5} more)" if len(nodes) > 5 else ""))
        
        print(f"\nNumber of sections: {len(self.idx_on_section)}")
        for section, nodes in sorted(self.idx_on_section.items()):
            if len(nodes) > 1:  # Only show sections with multiple classes
                print(f"  Section {section}: {len(nodes)} classes")
        
        print("="*50 + "\n")