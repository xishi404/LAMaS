#!/usr/bin/env python
# -*- coding: utf-8 -*-
from enum import Enum

from lamas.actions.action import Action
from lamas.actions.action_output import ActionOutput
from lamas.actions.add_requirement import UserRequirement
from lamas.actions.debug_error import DebugError
from lamas.actions.design_api import WriteDesign
from lamas.actions.design_api_review import DesignReview
from lamas.actions.project_management import WriteTasks
from lamas.actions.research import CollectLinks, WebBrowseAndSummarize, ConductResearch
from lamas.actions.run_code import RunCode
from lamas.actions.search_and_summarize import SearchAndSummarize
from lamas.actions.write_code import WriteCode
from lamas.actions.write_code_review import WriteCodeReview
from lamas.actions.write_prd import WritePRD
from lamas.actions.write_prd_review import WritePRDReview
from lamas.actions.write_test import WriteTest
from lamas.actions.di.execute_nb_code import ExecuteNbCode
from lamas.actions.di.write_analysis_code import WriteAnalysisCode
from lamas.actions.di.write_plan import WritePlan


class ActionType(Enum):
    """All types of Actions, used for indexing."""

    ADD_REQUIREMENT = UserRequirement
    WRITE_PRD = WritePRD
    WRITE_PRD_REVIEW = WritePRDReview
    WRITE_DESIGN = WriteDesign
    DESIGN_REVIEW = DesignReview
    WRTIE_CODE = WriteCode
    WRITE_CODE_REVIEW = WriteCodeReview
    WRITE_TEST = WriteTest
    RUN_CODE = RunCode
    DEBUG_ERROR = DebugError
    WRITE_TASKS = WriteTasks
    SEARCH_AND_SUMMARIZE = SearchAndSummarize
    COLLECT_LINKS = CollectLinks
    WEB_BROWSE_AND_SUMMARIZE = WebBrowseAndSummarize
    CONDUCT_RESEARCH = ConductResearch
    EXECUTE_NB_CODE = ExecuteNbCode
    WRITE_ANALYSIS_CODE = WriteAnalysisCode
    WRITE_PLAN = WritePlan


__all__ = [
    "ActionType",
    "Action",
    "ActionOutput",
]
