# -*- coding: utf-8 -*-
import datetime

from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.selector import HtmlXPathSelector
from scrapy.http.request import Request

from crawler.items import CrawlerItem
from crawler.utils import get_first

class SpiegelSpider(CrawlSpider):
    """Spider for 'Spiegel Online'"""
    name = 'spiegel'
    rotate_user_agent = True
    allowed_domains = ['www.spiegel.de']
    start_urls = ['http://www.spiegel.de']
    rules = (
        Rule(
            LinkExtractor(
                allow=(
                    '(politik|wirtschaft|panorama|thema)\/.*\/$',
                    'thema\/.+\.html$',
                    ),
                deny=('forum')
            ),
            follow=True
        ),
        Rule(
            LinkExtractor(
                allow=('(politik|wirtschaft|panorama)\/.+\.html$'),
                deny=('forum')
            ),
            callback='parse_page',
        ),
    )

    def parse_page(self, response):
        """Scrapes information from pages into items"""
        item = CrawlerItem()
        item['url'] = response.url.encode('utf-8')
        item['visited'] = datetime.datetime.now().isoformat().encode('utf-8')
        item['published'] = get_first(response.selector.xpath('//meta[@name="date"]/@content').extract())
        item['title'] = get_first(response.selector.css('.headline').xpath('./text()').extract())
        item['description'] = get_first(response.selector.xpath('//meta[@name="description"]/@content').extract())
        item['text'] = "".join([s.strip().encode('utf-8') for s in response.selector.xpath('//div[@class="article-section clearfix"]/p/text()').extract()])
        item['author'] = [s.encode('utf-8') for s in response.selector.xpath('//p[@class="author"]/a/text()').extract()]
        item['keywords'] = [s.encode('utf-8') for s in response.selector.xpath('//meta[@name="news_keywords"]/@content').extract()]
        return item
