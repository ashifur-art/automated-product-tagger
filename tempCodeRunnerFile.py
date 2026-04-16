from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# ১. মডেল সেটআপ
llm = ChatOllama(model="llama3.2")

# ২. PDF লোড করা - আপনার আসল PDF ফাইলের পাথ দিন
# উদাহরণ: "C:/Users/ashif/Documents/myfile.pdf" অথবা "D:/myfolder/syllabus.pdf"
pdf_path = "your_actual_pdf_file.pdf"  # ← এখানে আপনার আসল PDF ফাইলের নাম দিন
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# ৩. ডকুমেন্ট ছোট ছোট ভাগে ভাগ করা
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# ৪. এম্বেডিং ও ভেক্টর ডাটাবেস তৈরি
embeddings = OllamaEmbeddings(model="llama3.2")
vectorstore = Chroma.from_documents(chunks, embeddings)

# ৫. রিট্রিভার তৈরি
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ৬. প্রম্পট ডিজাইন
system_prompt = (
    "তুমি একজন সহায়ক এআই অ্যাসিস্ট্যান্ট। "
    "নিচে দেওয়া context ব্যবহার করে প্রশ্নের উত্তর দাও। "
    "যদি উত্তর খুঁজে না পাও, তবে বলো যে তুমি জানো না। "
    "বেশি কথা না বলে ৩ লাইনের মধ্যে উত্তর শেষ করো।"
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# ৭. রিট্রিভাল চেইন তৈরি করা
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# ৮. প্রশ্ন করা
response = rag_chain.invoke({"input": "এই পিডিএফ অনুযায়ী ২য় সপ্তাহে কী শিখতে হবে?"})

print("AI-এর উত্তর:")
print(response["answer"])