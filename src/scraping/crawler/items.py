"""
Definition of the model for the scraped item
See documentation in:
http://doc.scrapy.org/en/latest/topics/items.html
"""
import scrapy


class CrawlerItem(scrapy.Item):
    """Model for the scraped items"""

    url = scrapy.Field()
    visited = scrapy.Field()
    published = scrapy.Field()
    title = scrapy.Field()
    description = scrapy.Field()
    text = scrapy.Field()
    author = scrapy.Field()
    keywords = scrapy.Field()
