import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from redlines import Redlines

prompt_template = """As an experienced English language professor with attention to detail. 
                Your primary role is to assist students in improving their English language skills. 
                Follow these step-by-step responsibilities:

                - Analyze the given paragraph and break it down into individual sentences.
                - Identify and correct any grammatical errors in each sentence.
                - Opt for more suitable word choices.
                - Suggest revised sentences that are more precise, nature, fluent and native-like to enhance comprehension.
                - Explain the changes made and justify the reasons for these revisions.
                - Offer specific advice on areas that need improvement.
                - Output a Python list of dictionary objects with the following format:

                {format_instructions}

                Use the steps above to improve the paragraphs below provided by the student, which are delimited by triple backticks:

                The student-provided paragraph: ```{student_paragraphs}```
                """

example_paragraph1 = """Thank you, couch Javier, for your email! My daughter Emily will bring \
a check to your class tomorrow for last month's fencing class tuition. \
I will also discuss the needs of fencing equipment and private lesson with Emily this evening. \
We were thinking about getting her own equipment once she advances to the next level.
"""

example_paragraph2 = """Tom, glad to hear you are promoted to the Sr. Director of Data Science recently. \
I am writing this email to you to congratulate your new role. Great Job done! \
Wish we will meet soon in NYC again. Keep in touch. Cheers, Javier.
"""

example_paragraph3 = """my wife really enjoys her new role at XYZ corporation. Close proximity \
to home (means no long commute, more time to sleep), super nice boss. The school will end for \
the summer break. Hope we will have more quality family time soon.
"""

original_para_schema = ResponseSchema(name="Original Paragraph(s)",
                                      description="The paragraph(s) provided by the student, unaltered.")

revised_para_schema = ResponseSchema(name="Revised Paragraph(s)",
                                     description="The improved paragraph(s) after revision.")

revision_schema = ResponseSchema(name="Revision",
                                 description="A list of dictionaries, each including the keys "
                                             "'Original sentence', 'Revised sentence', and 'Reasons'.")

advice_schema = ResponseSchema(name="Targeted Advice for Improvement",
                               description="list improvement areas in markdown ordered list format, each item start with a dash sign '-'. ")

output_parser = StructuredOutputParser.from_response_schemas([original_para_schema,
                                                              revised_para_schema,
                                                              revision_schema,
                                                              advice_schema])

format_instructions = output_parser.get_format_instructions()


def predict(student_paragraphs):


    lingu_prompt = ChatPromptTemplate.from_template(template=prompt_template)

    lingu_prompt_messages = lingu_prompt.format_messages(student_paragraphs=student_paragraphs,
                                                         format_instructions=format_instructions)

    chat = ChatOpenAI(temperature=0.0)
    response = chat(lingu_prompt_messages)
    output_dict = output_parser.parse(response.content)

    return output_dict


st.set_page_config(page_title="NativeLinguMentor",
                   layout="wide",
                   page_icon=":robot:")

st.header("NativeLinguMentor")
st.caption("""
        _**NativeLinguMentor** is your personal intelligent English language mentor designed to 
        polish your English skills for a native-like flair. Powered by innovative technology, 
        the app deconstructs text into sentences, identifies and corrects grammar and syntax issues, 
        and provides revisions to ensure a native and fluent tone. **NativeLinguMentor** provides 
        targeted tips to refine your language skills and empowers users with the tools needed 
        for more effective communication. © 2023 Xinjie Qiu ℠_
        """)

col1, col2 = st.columns(2)

with col1:
    input_text = st.text_area("**Your paragraph(s) here: (one example provided):**",
                              example_paragraph1,
                              height=150)
    submitted = st.button(':violet[Click here to improve It!]')

    if submitted & (input_text != ""):
        with st.spinner(text="This may take a moment..."):
            output_dict = predict(student_paragraphs=input_text)

        original = output_dict.get('Original Paragraph(s)')
        revised = output_dict.get('Revised Paragraph(s)')
        advices = output_dict.get('Targeted Advice for Improvement')
        corrections = output_dict.get('Revision')

        corrections_markdown = ""
        for correction in corrections:
            corrections_markdown += " - **Original:** " + correction['Original sentence'] +"\n"
            if correction['Original sentence'] != correction['Revised sentence']:
                corrections_markdown += "     - **Revised:** " + correction['Revised sentence'] +"\n"
            corrections_markdown += "     - **Reasons:** " + correction['Reasons'] +"\n"

        st.divider()
        st.write("**Sentence by Sentence Correction:**")
        st.markdown(corrections_markdown)

with col2:
    if submitted & (input_text != ""):
        st.text_area("**AI polished paragraph(s):**",
                   revised,
                   height=150)

        diff = Redlines(original, revised)
        st.divider()
        st.write("**Corrections:**")
        st.markdown(diff.output_markdown, unsafe_allow_html=True)

        st.divider()
        st.write("**Areas for Improvement:**")
        st.write(advices)



with st.expander("See raw JASON output"):
    if submitted:
        st.write(output_dict)
