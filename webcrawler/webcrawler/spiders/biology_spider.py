import scrapy

class spider1(scrapy.Spider):
    name = "biology"
    start_urls = ['https://www.studysmarter.de/schule/biologie/entwicklungsbiologie/']
    def parse(self, response):
       for explaination in response.css("div.col-lg-7.col-md-12.content-column"): ## api-content contains the whole content
        yield {
            "title": explaination.css("h2.h1.hero__title.hero__title-ab::text").get(),
            "contents": explaination.css("div#api-content").getall(),
        }

        next_page = response.css('li.second-level-name a::attr(href)').getall() ## reads out all second-level links that are listed on page
        for item in next_page: ## iterate through list of all second level links
            if item is not None:
                item = response.urljoin(item)
                yield scrapy.Request(item, callback=self.parse)