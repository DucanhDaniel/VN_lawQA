from services.qa_service import QAService
# from core.vector_store import get_vectorstore
test = QAService()
print(test.get_chat_response("Giới thiệu hiến pháp Việt Nam", "0"))

print(test.get_chat_response("Hiến pháp Việt Nam quy định như thế nào về quyền con người?", "0"))