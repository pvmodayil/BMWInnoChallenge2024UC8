'''
    GPT4 Vision model based Image Analysis
    ------------------------------------------------
    
    This program analyses the images of rooftops and identifies if there are any abnormalities.
    It mainly checks to see if there are any construction work or if the rooftop hatches are open or closed.

'''
# import modules
#---------------------------------------------------------------
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core import SimpleDirectoryReader
from langchain_community.utils.openai_functions import (
    convert_pydantic_to_openai_function,
)
from pydantic import BaseModel, Field
from typing  import Optional
import json, pprint, os
from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser
from datetime import datetime
import streamlit as st


# specify the path to the images, and output
#-----------------------------------------------------------------
image_dir = "path-to-images"
json_file_path = "output.json"

# setup the gpt model with openai api key and set the number of tokens
#-----------------------------------------------------------------
openai_mm_llm = OpenAIMultiModal(
    model="gpt-4-vision-preview", api_key="your-api-key", max_new_tokens=1024
)

# setup the pydantic class to change the mode of the gpt-model resposne
#-----------------------------------------------------------------
class Abc(BaseModel):
    open_hatch: str = Field(default="No", description="Indicates if the small roof window on top of building is open.")
    construction: str = Field(default="No", description="Indicates if construction or construction materials is present.")

# ensure the directory exists else raise exception
#-----------------------------------------------------------------
if not os.path.isdir(image_dir):
    raise Exception(f"The directory {image_dir} does not exist")

# setup the streamlit app
#-----------------------------------------------------------------
st.title('Drone Image Analysis')

# write hackathon and team info on sidebar
st.sidebar.title("BMW Innovation Challenge: Use Case 8")
st.sidebar.write("Members: Siddhi Gunaji, Chaitya Teli, Mridul Koshy, Philip Modayil")


# the image analysis start button
#-----------------------------------------------------------------
if st.button('Process Images'):
    if not os.path.isdir(image_dir):
        st.error(f"The directory {image_dir} does not exist")
    else:
        json_file_path = "output.json"
        results = []

        # go through each image and send it to openai
        #-----------------------------------------------------------------
        for image_file in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_file)
            # Load each image individually
            image_reader = SimpleDirectoryReader(input_files=[image_path])
            imgdocs = image_reader.load_data()

            # setup the prompt for the model
            #-----------------------------------------------------------------
            prompt_template_str = """
            can you summarize what is in the image, ONLY look at roof of building, hatches are nothing but small open/close windows on top of building \
            and return the answer with json format \
            """

            # setup the gpt model object
            #-----------------------------------------------------------------
            openai_program = MultiModalLLMCompletionProgram.from_defaults(
                output_parser=PydanticOutputParser(Abc),
                image_documents=imgdocs,
                prompt_template_str=prompt_template_str,
                multi_modal_llm=openai_mm_llm,
                verbose=False,
            )
            
            # get response from the model
            #-----------------------------------------------------------------
            response = openai_program()
            response_dict = dict(response)
            data_with_metadata = {
                "image_file": image_file,
                "timestamp": datetime.now().isoformat(),
                "response": response_dict  # include response_dict instead of response
            }
            results.append(data_with_metadata)
            
            # setup the output format in the app (left(col1):image, right(col2):model response)
            #-----------------------------------------------------------------
            col1, col2 = st.columns(2)
            
            # display image in the first column
            col1.image(image_path, use_column_width=True)

            # display result in the second column
            col2.write(f"Image: {image_file}")
            if data_with_metadata["response"]["open_hatch"] == "Yes":
                col2.error("Hatch Open = Yes") # colour coded display to identify abnormal activity
            elif data_with_metadata["response"]["open_hatch"] == "No":
                col2.write("Hatch Open = No")

            if data_with_metadata["response"]["construction"] == "Yes":
                col2.error("Construction = Yes") # colour coded display to identify abnormal activity
            elif data_with_metadata["response"]["construction"] == "No":
                col2.write("Construction = No")