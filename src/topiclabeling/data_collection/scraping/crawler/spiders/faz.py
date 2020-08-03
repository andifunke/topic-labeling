# -*- coding: utf-8 -*-
import datetime

from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

from topiclabeling.scraping.crawler.items import CrawlerItem


class FazSpider(CrawlSpider):
    """Spider for 'Frankfurter Allgemeine Zeitung'"""
    name = 'faz'

    custom_settings = {
        'LOG_FILE': f"../../logs/{name}.log",
        'LOG_ENABLED': True,
        'LOG_LEVEL': "INFO",
        'LOG_STDOUT': False,
    }

    def __init__(self, categories='politik|wirtschaft', *args, **kwargs):
        super(FazSpider, self).__init__(*args, **kwargs)

        self.rotate_user_agent = True
        self.allowed_domains = ['www.faz.net']
        self.start_urls = ['http://www.faz.net']

        FazSpider.rules = (
            Rule(
                LinkExtractor(
                    allow=(
                        'aktuell\/(' + categories + ')\/.*\/$',
                        'aktuell\/(' + categories + ')\/.*\/s\d+\.html',
                    ),
                ),
                follow=True
            ),
            Rule(
                LinkExtractor(
                    allow=('aktuell\/(' + categories + ').+\.html'),
                ),
                callback='parse_page',
            ),
        )
        super(FazSpider, self)._compile_rules()

    def parse_page(self, response):
        """Scrapes information from pages into items"""
        item = CrawlerItem()
        item['url'] = response.url
        item['visited'] = datetime.datetime.now().isoformat()
        item['text'] = response.css('p.atc-TextParagraph').extract()
        item['keywords'] = [s for s in response.xpath('//meta[@name="keywords"]/@content').extract()]
        item['published'] = response.css('.atc-MetaTime::attr(datetime)').extract_first()
        item['title'] = response.xpath('//meta[@property="og:title"]/@content').extract_first()
        item['description'] = response.css('.atc-IntroText::text').extract_first()
        item['author'] = [
            s for s in
            response
                .selector
                .xpath('//span[@class="Autor"]/span[@class="caps last"]/a/span/text()')
                .extract()
        ]
        return item
