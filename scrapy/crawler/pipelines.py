# -*- coding: utf-8 -*-
# Definition of item pipelines
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
from w3lib.html import remove_tags
from scrapy.exceptions import DropItem


class RemoveTagsPipeline(object):
    """removing formatting tags (span, a, ...) from extracted paragraphs"""

    def process_item(self, item, spider):
        ps = []
        for p in item['text']:
            ps.append(remove_tags(p))
        item['text'] = '\n'.join(ps)

        if item['description']:
            item['description'] = item['description'].strip()

        # print(item['text'])
        return item


class DropIfEmptyFieldPipeline(object):

    def process_item(self, item, spider):
        if not item['text']:
            raise DropItem()
        else:
            return item
