#!/usr/bin/env python
# -*- coding: utf-8 -*-


from lamas.actions import Action
from lamas.schema import Message


class ExecuteTask(Action):
    name: str = "ExecuteTask"
    i_context: list[Message] = []

    async def run(self, *args, **kwargs):
        pass
