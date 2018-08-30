# -*- coding: utf-8 -*-
import datetime

from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
# from scrapy.selector import HtmlXPathSelector
# from scrapy.http.request import Request

from crawler.items import CrawlerItem
# from crawler.utils import get_first


class FazSpider(CrawlSpider):
    """Spider for 'Frankfurter Allgemeine Zeitung'"""
    name = 'faz'
    rotate_user_agent = True
    allowed_domains = ['www.faz.net']
    start_urls = ['http://www.faz.net']
    rules = (
        Rule(
            LinkExtractor(
                allow=(
                    'aktuell\/(politik|wirtschaft)\/.*\/$',
                    'aktuell\/(politik|wirtschaft)\/.*\/s\d+\.html',
                ),
            ),
            follow=True
        ),
        Rule(
            LinkExtractor(
                allow=('aktuell\/(politik|wirtschaft).+\.html'),
            ),
            callback='parse_page',
        ),
    )

    def parse_page(self, response):
        """Scrapes information from pages into items"""
        item = CrawlerItem()
        item['url'] = response.url
        item['visited'] = datetime.datetime.now().isoformat()
        item['text'] = response.css('p.atc-TextParagraph').extract()
        item['keywords'] = [s for s in response.xpath('//meta[@name="keywords"]/@content').extract()]
        item['published'] = response.css('.atc-MetaTime::attr(datetime)').extract_first()
        item['title'] = response.xpath('//meta[@property="og:title"]/@content').extract_first()
        # item['title2'] = response.xpath('/html/head/meta[6]/@content').extract_first()
        item['description'] = response.css('.atc-IntroText::text').extract_first()
        # item['description'] = response.xpath('/html/head/meta[3]/@content').extract_first()
        # empty:
        item['author'] = [s for s in response.selector.xpath('//span[@class="Autor"]/span[@class="caps last"]/a/span/text()').extract()]
        return item
