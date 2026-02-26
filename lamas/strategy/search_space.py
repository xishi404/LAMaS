#!/usr/bin/env python
# -*- coding: utf-8 -*-


class SearchSpace:
    """SearchSpace: 用于定义一个搜索空间，搜索空间中的节点是 ActionNode 类。"""

    def __init__(self):
        self.search_space = {}

    def add_node(self, node):
        self.search_space[node.key] = node

    def get_node(self, key):
        return self.search_space[key]
