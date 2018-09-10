# -*- coding: utf-8 -*-
import datetime

from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.selector import HtmlXPathSelector
from scrapy.http.request import Request

from crawler.items import CrawlerItem
from crawler.utils import get_first

class ZeitSpider(CrawlSpider):
    """Spider for 'Zeit Online'"""
    name = 'zeit'
    rotate_user_agent = True
    allowed_domains = ['www.zeit.de']
    start_urls = [
        'http://www.zeit.de/index',
        'http://www.zeit.de/suche/index',
    ]
    rules = (
        Rule(
            LinkExtractor(
                allow=(
                    '(politik|gesellschaft|wirtschaft|suche).*\/index',
                    'thema\/',
                )
            ),
            follow=True
        ),
        Rule(
            LinkExtractor(
                allow=('(politik|gesellschaft|wirtschaft)(\/.+)*\/\d{4}-\d{1,2}\/.+'),
                deny=('-fs')
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
        item['title'] = get_first(response.selector.xpath('//meta[@property="og:title"]/@content').extract())
        item['description'] = get_first(response.selector.xpath('//meta[@name="description"]/@content').extract())
        item['text'] = "".join([s.strip().encode('utf-8') for s in response.selector.css('.article__item').css('.paragraph').xpath('.//text()').extract()])
        item['author'] = [s.encode('utf-8') for s in response.selector.css('.byline').css('span[itemprop="name"]').xpath('./text()').extract()]
        item['keywords'] = [s.encode('utf-8') for s in response.selector.xpath('//meta[@name="keywords"]/@content').extract()]
        # Handle next pages
        next_page = get_first(response.selector.xpath('//link[@rel="next"]/@href').extract())
        if next_page:
            self.logger.debug("Next page found: "+next_page)
            yield Request(next_page,callback=self.parse_page)
        yield item
