# from elasticsearch import Elasticsearch
# from elasticsearch.helpers import bulk
# import json
# import time
# import zipfile
# from utils.helper import download_zip
# from pprint import pprint


# class ElasticSearchHandler:
#     def __init__(self, hostname, port, scheme, index_name):
#         self.es = Elasticsearch([{'host': hostname, 'port': port, 'scheme': scheme}])
#         self.index_name = index_name
        
#     def createindexmapping(self):
#         if self.es.indices.exists(index=self.index_name):
#             print(f"Index {self.index_name} already exists! Mapping is already done for this index.")
#         else:
#             final_mapping = {
#                 "mappings": {
#                     "properties": {
#                         "resume_file": {"type": "text"},
#                         "total_experience": {"type": "integer"},
#                         "designations": {"type": "keyword"},
#                         "degree_array": {"type": "keyword"},
#                         "technical_domain": {"type": "text"},
#                         "programming_languages": {"type": "text"},
#                         "databases":{"type":"text"},
#                         "technical_skills": {"type": "text"},
#                         "cloud_services": {"type": "text"},
#                         "devops":{"type":"text"},
#                         # "deployment_tools": {"type": "keyword"},
#                         # "code_versioning": {"type": "keyword"},
#                         "api_protocols": {"type": "text"},
#                         "big_data": {"type": "text"},
#                         "web_framework": {"type": "text"},
#                         "css_framework": {"type": "text"},
#                         "search_engine": {"type": "text"},
#                         "projects": {
#                             "type": "nested",
#                             "properties": {
#                                 "project_name": {"type": "text"},
#                                 "skills": {"type": "keyword"},
#                                 "industry": {"type": "text"}
#                             },
#                         },
#                         "other_skills": {"type": "keyword"}
#                     }
#                 }
#             }
#             self.es.indices.create(index=self.index_name, body=final_mapping)
#             print(f"Index {self.index_name} created and successfully mapped it.")

#     def indexingResumeFromFile(self, file_path):
#         try:
#             with open(file_path, 'r') as file:
#                 data = json.load(file)
#                 self.es.index(index=self.index_name, body=data)
#                 print("Indexed resume data from file.")
#         except Exception as e:
#             print(f"Error: {e}")

#     def indexingResumeFromDictionary(self, dict_data):
#         try:
#             if dict_data:
#                 self.es.index(index=self.index_name, body=dict_data)
#                 print("Indexed resume data from dictionary.")
#         except Exception as e:
#             print(f"Error: {e}")

#     def indexing_bulk_resumeFromDictionary(self, dict_data_list):
#         try:
#             start = time.time()
#             if dict_data_list:
#                 actions = [
#                     {
#                         "_index": self.index_name,
#                         "_source": data
#                     }
#                     for data in dict_data_list
#                 ]
#                 success, _ = bulk(self.es, actions, raise_on_error=False)
#                 print(f"Indexed {success} resumes from dictionary.")
#             end = time.time()
#             print("time taken ",end-start)
#         except Exception as e:
#             print(f"Error: {e}")


#     def get_indexed_resume_names(self):
#         '''
#         Get names of already indexed resumes
#         '''
#         query={
#             "_source": ["resume_file"],
#             "query": {"match_all":{}}
#         }
      
#         load_data_from_elastic=[]
#         try:
#             resp = self.es.search(index="gemini_extracted_resumes", body=query)
#             for data in resp['hits']['hits']:
#                 load_data_from_elastic.append(data['_source']['resume_file'])
#         except Exception as e:
#             print(e.__str__())
#         return load_data_from_elastic
        

#     def extract_keys(self,input_dict, keys):
#         return {k: input_dict[k] for k in keys if k in input_dict}
    
#     def search_resumes(self,min_exp_range,max_exp_range,compulsory_programming_languages, programming_languages_compulsory=False, programming_languages='', 
#                    cloud_services_compulsory=False,must_cloud_services='',cloud_services='', databases_compulsory=False,must_databases='' ,databases='',
#                   devops_compulsory=False, must_devops='',devops='',big_data_compulsory=False,must_big_data='',big_data='',
#                   framework_compulsory=False,must_framework='',framework=''):
#         should_fields = []
#         must_fields = []
#         include_all = False

#         # taking all fields as comma separated string
#         #Extracting compulsory and optional field names in list. If compulsory or optional field is empty replace it by empty list.
#         if compulsory_programming_languages=='': 
#             compulsory_programming_languages = []
#         else:
#             compulsory_programming_languages=[language.strip() for language in compulsory_programming_languages.lower().split(',')]
#             # must_fields.append("compulsory_programming_languages")

#         if programming_languages =='':
#             programming_languages = []
#         else:
#             programming_languages = [language.strip() for language in programming_languages.lower().split(',')]
#             should_fields.append("programming_languages")
        
#         if cloud_services == '':
#             cloud_services = []
#         else:
#             cloud_services = [cld.strip() for cld in cloud_services.lower().split(',')]
#             should_fields.append("cloud_services")

#         if must_cloud_services=='':
#             must_cloud_services=[]
#         else:
#             must_cloud_services=[cld.strip() for cld in must_cloud_services.lower().split(',')]
    
#         if databases == '':
#             databases =[]
#         else:
#             databases = [db.strip() for db in databases.lower().split(',')]
#             should_fields.append("databases")

#         if must_databases=='':
#             must_databases=[]
#         else:
#             must_databases=[db.strip() for db in must_databases.lower().split(',')]

#         if devops=='':
#             devops=[]
#         else:
#             devops=[dev.strip() for dev in devops.lower().split(',')]
#             should_fields.append("devops")

#         if must_devops=='':
#             must_devops=[]
#         else:
#             must_devops=[dev.strip() for dev in must_devops.lower().split(',')]

#         if big_data=='':
#             big_data=[]
#         else:
#             big_data=[data.strip() for data in big_data.lower().split(',')]
#             should_fields.append("big_data")

#         if must_big_data=='':
#             must_big_data=[]
#         else:
#             must_big_data=[data.strip() for data in must_big_data.lower().split(',')]

#         if framework=='':
#             framework=[]
#         else:
#             web_framework=[data.strip() for data in framework.lower().split(',')]
#             should_fields.append("web_framework")

#         if must_framework=='':
#             must_framework=[]
#         else:
#             must_framework=[data.strip() for data in must_framework.lower().split(',')]

#         #dictionary for compulsory fields with key as field names and value as list of field values.
#         must_dict={"programming_languages":[compulsory_programming_languages,programming_languages_compulsory],
#             "cloud_services":[must_cloud_services,cloud_services_compulsory],
#             "databases":[must_databases,databases_compulsory],
#             "devops":[must_devops,devops_compulsory],
#             "big_data":[must_big_data,big_data_compulsory],
#             "web_framework":[must_framework,framework_compulsory]}

#         min_exp, max_exp = min_exp_range,max_exp_range
#         project_skills = []
#         project_skills.extend(programming_languages)
#         project_skills.extend(cloud_services)
#         project_skills.extend(databases)

#         print(project_skills, "skills")
        
#         # generate dynamic search query for es
#         search_body = {
#         "size":200,
#         "query": {
#             "bool": {
#                 "filter": [{
#                     "range": {
#                         "total_experience": {
#                             "gte": min_exp,
#                             "lte": max_exp
#                         }
#                     }
#                 }],
#                 "must": [],
#                 "should": [],
#                 }
#             },
#             "sort": [
#             {"_score": {"order": "desc"}},      # Sort by relevance score (optional)               
#         ]
#         }

#         #iterating over keys of compulsory fields dictionary.
#         for fields in must_dict.keys():
#             if must_dict[fields][1]:      #==True: # compulsory flag is true then generate term query for compulsory fields else terms query.
#                 for skill in must_dict[fields][0]:
#                     terms_query = {"term":{fields:skill}}
#                     search_body["query"]["bool"]["must"].append(terms_query)
#             else:
#                 if must_dict[fields][0]:
#                     terms_query={"terms":{fields:must_dict[fields][0]}}
#                     search_body["query"]["bool"]["must"].append(terms_query)
#         print("search body ------", search_body)

#         #generating terms query for should fields.
#         for field in should_fields:
#             if eval(field.replace("'", "\'")):
#                 terms_query = {
#                     "terms": {
#                         field: eval(field.replace("'", "\'"))  # You can replace [term_value] with the appropriate term values
#                     }
#                 }
#                 search_body["query"]["bool"]["should"].append(terms_query)
                
#         if search_body["query"]["bool"].get("should"):
#             search_body['query']['bool']['minimum_should_match']=1

#         print("final search body","===============",end='\n')
#         pprint(search_body)
                
#         resp = self.es.search(index=self.index_name,body = search_body)
#         resumes = []
#         specified_keys = ["total_experience", "programming_languages", "databases",'cloud_services',"web_framework","devops","api_protocols",
#                           "big_data","web_framework","ides"]

#         # Extract specified keys
#         final_dict = {}
#         for one in resp['hits']['hits']:
#             output_dict = self.extract_keys(one['_source'], specified_keys)
#             final_dict[one['_source']['resume_file']] = output_dict

#             resumes.append(one['_source']['resume_file'])

#         # resume_path=download_zip(resumes)
#         return json.dumps(final_dict, indent=4)











































#========================================================================================================================================================


#BELOW WORKING





# utils/elastic_search_inbuild.py

# utils/elastic_search_inbuild.py

import json
from pprint import pprint
from elasticsearch import Elasticsearch

class ElasticSearchHandler:
    def __init__(self, hostname, port, scheme, index_name):
        self.es = Elasticsearch([{"host": hostname, "port": port, "scheme": scheme}])
        self.index_name = index_name

    def extract_keys(self, source_dict, keys):
        """Extract only required keys from source dict."""
        return {k: source_dict.get(k, None) for k in keys}

    def search_resumes(
        self,
        min_exp_range,
        max_exp_range,
        compulsory_programming_languages,
        programming_languages_compulsory=False,
        programming_languages='',
        cloud_services_compulsory=False,
        must_cloud_services='',
        cloud_services='',
        databases_compulsory=False,
        must_databases='',
        databases='',
        devops_compulsory=False,
        must_devops='',
        devops='',
        big_data_compulsory=False,
        must_big_data='',
        big_data='',
        framework_compulsory=False,
        must_framework='',
        framework=''
    ):
        should_fields = []
        must_fields = []

        # ✅ Convert all inputs into lowercase lists
        compulsory_programming_languages = (
            [x.strip().lower() for x in compulsory_programming_languages.split(",")]
            if compulsory_programming_languages else []
        )
        programming_languages = (
            [x.strip().lower() for x in programming_languages.split(",")]
            if programming_languages else []
        )
        cloud_services = (
            [x.strip().lower() for x in cloud_services.split(",")]
            if cloud_services else []
        )
        must_cloud_services = (
            [x.strip().lower() for x in must_cloud_services.split(",")]
            if must_cloud_services else []
        )
        databases = (
            [x.strip().lower() for x in databases.split(",")]
            if databases else []
        )
        must_databases = (
            [x.strip().lower() for x in must_databases.split(",")]
            if must_databases else []
        )
        devops = (
            [x.strip().lower() for x in devops.split(",")]
            if devops else []
        )
        must_devops = (
            [x.strip().lower() for x in must_devops.split(",")]
            if must_devops else []
        )
        big_data = (
            [x.strip().lower() for x in big_data.split(",")]
            if big_data else []
        )
        must_big_data = (
            [x.strip().lower() for x in must_big_data.split(",")]
            if must_big_data else []
        )
        framework = (
            [x.strip().lower() for x in framework.split(",")]
            if framework else []
        )
        must_framework = (
            [x.strip().lower() for x in must_framework.split(",")]
            if must_framework else []
        )

        # ✅ Dictionary for compulsory fields
        must_dict = {
            "programming_languages": [compulsory_programming_languages, programming_languages_compulsory],
            "cloud_services": [must_cloud_services, cloud_services_compulsory],
            "databases": [must_databases, databases_compulsory],
            "devops": [must_devops, devops_compulsory],
            "big_data": [must_big_data, big_data_compulsory],
            "web_framework": [must_framework, framework_compulsory]
        }

        # ✅ Convert experience ranges (fallback max = 50)
        try:
            min_exp = int(min_exp_range) if min_exp_range else 0
        except ValueError:
            min_exp = 0
        try:
            max_exp = int(max_exp_range) if max_exp_range else 50
        except ValueError:
            max_exp = 50

        # ✅ Build search query
        search_body = {
            "size": 200,
            "query": {
                "bool": {
                    "filter": [
                        {"range": {"total_experience": {"gte": min_exp, "lte": max_exp}}}
                    ],
                    "must": [],
                    "should": []
                }
            },
            "sort": [{"_score": {"order": "desc"}}]
        }

        # ✅ Add compulsory conditions
        for field, (values, is_compulsory) in must_dict.items():
            if values:
                if is_compulsory:
                    search_body["query"]["bool"]["must"].append({"terms": {field: values}})
                else:
                    search_body["query"]["bool"]["should"].append({"terms": {field: values}})

        # ✅ Add optional fields
        if programming_languages:
            search_body["query"]["bool"]["should"].append({"terms": {"programming_languages": programming_languages}})
        if cloud_services:
            search_body["query"]["bool"]["should"].append({"terms": {"cloud_services": cloud_services}})
        if databases:
            search_body["query"]["bool"]["should"].append({"terms": {"databases": databases}})
        if devops:
            search_body["query"]["bool"]["should"].append({"terms": {"devops": devops}})
        if big_data:
            search_body["query"]["bool"]["should"].append({"terms": {"big_data": big_data}})
        if framework:
            search_body["query"]["bool"]["should"].append({"terms": {"web_framework": framework}})

        if search_body["query"]["bool"]["should"]:
            search_body["query"]["bool"]["minimum_should_match"] = 1

        print("\n📌 Final ES Query:\n", json.dumps(search_body, indent=4))

        # ✅ Fire query to Elasticsearch
        resp = self.es.search(index=self.index_name, body=search_body)

        # ✅ Process results
        resumes = {}

        hits = resp.get("hits", {}).get("hits", [])
        print(f"\n🔍 Found {len(hits)} resumes in ES")

        for hit in hits:
            src = hit["_source"]
            file_path = src.get("resume_file", "N/A")
            score = hit.get("_score", 0)
            resumes[file_path] = {
                "resume_path": file_path,
                "score": score,
                "total_experience": src.get("total_experience"),
                "programming_languages": src.get("programming_languages", []),
                "cloud_services": src.get("cloud_services", []),
                "databases": src.get("databases", []),
                "devops": src.get("devops", []),
                "big_data": src.get("big_data", []),
                "web_framework": src.get("web_framework", [])
            }
            print(f"📄 Resume: {file_path} | Score: {score}")

        return resumes