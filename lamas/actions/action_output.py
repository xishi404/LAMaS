#!/usr/bin/env python
# coding: utf-8

from pydantic import BaseModel


class ActionOutput:
    content: str
    instruct_content: BaseModel

    def __init__(self, content: str, instruct_content: BaseModel):
        self.content = content
        self.instruct_content = instruct_content
