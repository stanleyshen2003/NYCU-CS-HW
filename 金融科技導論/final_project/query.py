from typing import List

from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
from google.protobuf.json_format import MessageToDict

def client_create(
    project_id: str,
    location: str,
    engine_id: str,
) -> List[discoveryengine.SearchResponse]:
    
    client_options = (None)

    # Create a client
    client = discoveryengine.SearchServiceClient(client_options=client_options)

    # The full resource name of the search app serving config
    serving_config = f"projects/{project_id}/locations/{location}/collections/default_collection/engines/{engine_id}/servingConfigs/default_config"

    content_search_spec = discoveryengine.SearchRequest.ContentSearchSpec(
        snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
            return_snippet=False
        )
    )
    
    return client, serving_config, content_search_spec

def query_gcp(client, serving_config, content_search_spec, search_query, category):

    # Refer to the `SearchRequest` reference for all supported fields:
    # https://cloud.google.com/python/docs/reference/discoveryengine/latest/google.cloud.discoveryengine_v1.types.SearchRequest
    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=search_query,
        page_size=200,
        content_search_spec=content_search_spec,
        query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
            condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO,
        ),
        spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
            mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO
        ),
    )

    response = client.search(request)
    if category == 'faq':
        result = [int(dict(doc.document.struct_data)['pid']) for doc in response.results]
    else:
        result = [int(dict(doc.document.derived_struct_data)['title']) for doc in response.results]

    return result

class Query():
    def __init__(self):
        project_id = "tw-rd-tam-stanleyshen"
        location = "global"          # Values: "global", "us", "eu"

        self.indeces = {"insurance":0, "faq":1, "finance":2}
        self.engine_ids = {"insurance":"esun-insurance_1729776330935", "faq":"esun-faq_1729619178528", "finance":"esun-finance_1730708397021"}

        self.clients = []

        for category in self.indeces.keys():
            client, serving_config, content_search_spec = client_create(project_id, location, self.engine_ids[category])
            self.clients.append([client, serving_config, content_search_spec])

    def query(self, search_query, category):
        index = self.indeces[category]
        client, serving_config, content_search_spec = self.clients[index]
        result = query_gcp(client, serving_config, content_search_spec, search_query, category)
        return result
    

if __name__ == '__main__':
    query_util = Query()
    print(query_util.query("有關留學生在學期間，應每年向本行繳交留學生成績單或註冊證明等在學證明，及入出國及移民署核發之該留學生入出國日期證明書等相關文件，需於每年幾月繳交且如何繳交 ？", "faq"))