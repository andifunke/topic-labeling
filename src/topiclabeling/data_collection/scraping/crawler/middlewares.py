# -*- coding: utf-8 -*-
# Definition of the middleware used for rotating user agents
# List of user agents is defined in settings.py
from random import choice
from scrapy import signals
from scrapy.exceptions import NotConfigured


class RotateUserAgentMiddleware(object):
    """Middleware used for rotating user-agent for each request"""

    def __init__(self, user_agents):
        self.enabled = False
        self.user_agents = user_agents

    @classmethod
    def from_crawler(cls, crawler):
        """Get user agents from settings.py"""
        user_agents = crawler.settings.get("USER_AGENT_CHOICES", [])
        if not user_agents:
            raise NotConfigured("USER_AGENT_CHOICES not set or empty")
        ret = cls(user_agents)
        crawler.signals.connect(ret.spider_opened, signal=signals.spider_opened)
        return ret

    def spider_opened(self, spider):
        self.enabled = getattr(spider, "rotate_user_agent", self.enabled)

    def process_request(self, request, spider):
        """Select user agent randomly on request"""
        if self.enabled and self.user_agents:
            request.headers["user-agent"] = choice(self.user_agents)
