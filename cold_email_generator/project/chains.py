from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

import os
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=1, 
            groq_api_key=groq_api_key, 
            model_name="llama-3.1-8b-instant",
   
    )
        
    def extract_job(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM WEBSITE:
        {page_data}
        ### INSTRUCTION
        The scraped text is from the career's page of a website.
        Your job is to extract the job posting and return them in JSON format containing the following keys: 'role', 'experience', 'skills', and 'description'

        Only return the valid JSON.

        ### VALID JSON (NO PREAMBLE):
        """
        )

        chain_extract = prompt_extract | self.llm

        response = chain_extract.invoke(input ={ "page_data" : cleaned_text})

        try:
            json_parser = JsonOutputParser()
            response = json_parser.parse(response.content)

        except OutputParserException:
            raise OutputParserException("Unable to parse jobs.")
        
        return response if isinstance(response, list) else [response]   


    def write_email(self, job, links):
        # prompt template for creating email
        prompt_email = PromptTemplate.from_template(
        """
        ### JOB DESCRIPTION:
        {job_description}
        
        ### INSTRUCTION:
        You are Ashley, a business development executive at Random Consulting. Random Consulting is an AI & Software Consulting company dedicated to facilitating
        the seamless integration of business processes through automated tools.
        Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, process optimization, cost reduction, and heightened overall efficiency.

        Your job is to write a cold email to the client regarding the job mentioned above describing the capability of Random Consulting in fulfilling their needs.       

        Also add the most relevant ones from the following links to showcase Random Consulting's portfolio: {links_list}            

        Remember you are Ashley, Business Development Executive at Random Consulting. 

        Do not provide a preamble.
        ### EMAIL (NO PREAMBLE):
        
        """
        )

        chain_email = prompt_email | self.llm

        response_email = chain_email.invoke({"job_description": str(job), "links_list" : links})

        return response_email.content
            

groq_api_key = os.getenv("GROQ_API_KEY")
