from services.qa_service import QAService
# from core.vector_store import get_vectorstore
test = QAService()
print(test.get_chat_response("Các bước sơ cấp cứu bao gồm những gì?", "0"))

# print(test.get_chat_response("Nhiệm vụ của người cấp cứu là gì?", "0"))