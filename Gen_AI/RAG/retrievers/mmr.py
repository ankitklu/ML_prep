from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings



docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]


embeddings = HuggingFaceEmbeddings()


vectorstore = Chroma.from_documents(docs, embeddings)


similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

print("\n===== Similarity Search Results =====\n")

similarity_docs = similarity_retriever.invoke("What is gradient descent?")

for doc in similarity_docs:
    print(doc.page_content)


mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3}
)

print("\n===== MMR Results =====\n")

mmr_docs = mmr_retriever.invoke("What is gradient descent?")

for doc in mmr_docs:
    print(doc.page_content)