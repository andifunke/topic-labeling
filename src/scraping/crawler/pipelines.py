"""
Definition of item pipelines
See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
"""
from w3lib.html import remove_tags, remove_tags_with_content
from scrapy.exceptions import DropItem


class RemoveTagsPipeline(object):
    """removing formatting tags (span, a, ...) from extracted paragraphs"""

    def process_item(self, item, spider):
        ps = [
            remove_tags(remove_tags_with_content(p, ("script",)))
            .strip()
            .replace(u"\xa0", u" ")
            for p in item["text"]
        ]
        item["text"] = "\n".join(ps)

        # additional stripping for description
        if item["description"]:
            item["description"] = item["description"].strip()

        return item


class DropIfEmptyFieldPipeline(object):
    def process_item(self, item, spider):
        if not item["text"]:
            raise DropItem()
        else:
            return item
