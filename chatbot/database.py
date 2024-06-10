import psycopg
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_postgres import PostgresChatMessageHistory

class Database:
	def __init__(self, dbname, user, password, host, port):
		self.conn = psycopg.connect(
			dbname=dbname,
			user=user,
			password=password,
			host=host,
			port=port
		)
		self.collection_table_name = 'collections'
		self.chat_history_table_name = 'chat_history'

	def insert_collection(self, collection_name, description, host, port, hash_, last_update):
		with self.conn.cursor() as cursor:
			cursor.execute("""
				INSERT INTO collections (collection_name, desc_collection, host, port, hash_collection, last_update)
				VALUES (%s, %s, %s, %s, %s, %s)
			""", (collection_name, description, host, port, hash_, last_update))
			self.conn.commit()
			
	def update_collection(self, collection_name, description, host, port, hash_, last_update):
		with self.conn.cursor() as cursor:
			cursor.execute("""
				UPDATE collections
				SET desc_collection = %s, host = %s, port = %s, hash_collection = %s, last_update = %s
				WHERE collection_name = %s
			""", (description, host, port, hash_, last_update, collection_name))
			self.conn.commit()

	def get_all_collections(self):

		with self.conn.cursor() as cursor:
			cursor.execute("SELECT collection_name FROM collections")
			collections = cursor.fetchall()
			json_collections = []
			for collection in collections:
				json_collections.append(
					self.get_collection(collection[0])
				)
			return json_collections
	
	def get_collection(self, collection_name):
		with self.conn.cursor() as cursor:
			cursor.execute("SELECT * FROM collections WHERE collection_name = %s", (collection_name,))
			collection = cursor.fetchone()
			return {
				"id" : collection[0],
				"collection": collection[1],
				"description": collection[2],
				"host": collection[3],
				"port": collection[4],
				"hash": collection[5],
				"last_update": collection[6]
			}
		
	def delete_collection(self, collection_name):
		with self.conn.cursor() as cursor:
			cursor.execute("DELETE FROM collections WHERE collection_name = %s", (collection_name,))
			self.conn.commit()
	
	def test_if_table_exists(self, table_name):
		with self.conn.cursor() as cursor:
			cursor.execute("""
				SELECT EXISTS (
					SELECT 1
					FROM information_schema.tables
					WHERE table_name = %s
				)
			""", (table_name,))
			return cursor.fetchone()[0]

	def create_chat_history_table(self):
		if not self.test_if_table_exists(self.chat_history_table_name):
			PostgresChatMessageHistory.create_tables(self.conn, self.chat_history_table_name)
			return True
		return False

	def test_if_chat_history_exists(self, session_id):
		with self.conn.cursor() as cursor:
			cursor.execute(f"SELECT EXISTS (SELECT 1 FROM {self.chat_history_table_name} WHERE session_id = %s)", (session_id,))
			if cursor.fetchone()[0]:
				return True
			return False
	
	def init_chat_history(self, session_id):
		if not self.test_if_chat_history_exists(session_id):
			PostgresChatMessageHistory(self.chat_history_table_name, session_id, sync_connection=self.conn)

	def insert_chat_messages(self, session_id, messages):
		chat_history = PostgresChatMessageHistory(self.chat_history_table_name, session_id, sync_connection=self.conn)
		chat_history.clear()
		chat_history.add_messages(messages)

	def get_chat_messages(self, session_id):
		chat_history = PostgresChatMessageHistory(self.chat_history_table_name, session_id, sync_connection=self.conn)
		return chat_history
	
	def print_chat_history_shema(self):
		with self.conn.cursor() as cursor:
			cursor.execute(f"SELECT * FROM {self.chat_history_table_name} LIMIT 0")
			print(cursor.description)