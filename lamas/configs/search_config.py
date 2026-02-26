#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Callable, Optional

from pydantic import Field

from lamas.tools import SearchEngineType
from lamas.utils.yaml_model import YamlModel


class SearchConfig(YamlModel):
    """Config for Search"""

    api_type: SearchEngineType = SearchEngineType.DUCK_DUCK_GO
    api_key: str = ""
    cse_id: str = ""  # for google
    search_func: Optional[Callable] = None
    params: dict = Field(
        default_factory=lambda: {
            "engine": "google",
            "google_domain": "google.com",
            "gl": "us",
            "hl": "en",
        }
    )
