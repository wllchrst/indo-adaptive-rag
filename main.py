from vector_database import SeedHandler, DatabaseHandler

d = DatabaseHandler()

result = d.query("wikipedia_id", "Eskatologi Islam", 5)

print(result['documents'])
# for doc in result['documents']:
#     print(doc)