import psycopg
from langchain_postgres import PostgresChatMessageHistory
import uuid

class Database:
    def __init__(self, dbname, user, password, host, port):
        self.connect_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        self.collection_table_name = 'collections'
        self.chat_history_table_name = 'chat_history'
    
    async def connect(self):
        return await psycopg.AsyncConnection.connect(self.connect_url)

    async def insert_collection(self, collection_name, description, host, port, search_k, hash_, last_update):
        async with await self.connect() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO collections (collection_name, desc_collection, host, port, search_k, hash_collection, last_update)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (collection_name, description, host, port, search_k, hash_, last_update))
                await conn.commit()
            
    async def update_collection(self, collection_name, description, host, port, search_k, hash_, last_update):
        async with await self.connect() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("""
                    UPDATE collections
                    SET desc_collection = %s, host = %s, port = %s, search_k = %s, hash_collection = %s, last_update = %s
                    WHERE collection_name = %s
                """, (description, host, port, search_k, hash_, last_update, collection_name))
                await conn.commit()
    
    async def update_search_k(self, collection_name, search_k):
        async with await self.connect() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("""
                    UPDATE collections
                    SET search_k = %s
                    WHERE collection_name = %s
                """, (search_k, collection_name))
                await conn.commit()

    async def get_all_collections(self):
        async with await self.connect() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT collection_name FROM collections")
                collections = await cursor.fetchall()
                json_collections = []
                for collection in collections:
                    json_collections.append(
                        await self.get_collection(collection[0])
                    )
                return json_collections
    
    async def get_collection(self, collection_name):
        async with await self.connect() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT * FROM collections WHERE collection_name = %s", (collection_name,))
                collection = await cursor.fetchone()
                return {
                    "id" : collection[0],
                    "collection": collection[1],
                    "description": collection[2],
                    "host": collection[3],
                    "port": collection[4],
                    "search_k": collection[5],
                    "hash": collection[6],
                    "last_update": collection[7]
                }
        
    async def delete_collection(self, collection_name):
        async with await self.connect() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("DELETE FROM collections WHERE collection_name = %s", (collection_name,))
                await conn.commit()
    
    async def test_if_table_exists(self, table_name):
        async with await self.connect() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(f"SELECT EXISTS ( SELECT 1 FROM information_schema.tables WHERE table_name = '{table_name}' )")
                return (await cursor.fetchone())[0]

    async def create_chat_history_table(self):
        if not await self.test_if_table_exists(self.chat_history_table_name):
            async with await self.connect() as conn:
                await PostgresChatMessageHistory.acreate_tables(conn, self.chat_history_table_name)
                return True
        return False
    
    async def test_if_chat_history_exists(self, session_id):
        async with await self.connect() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(f"SELECT EXISTS (SELECT 1 FROM {self.chat_history_table_name} WHERE session_id = %s)", (session_id,))
                return (await cursor.fetchone())[0]
    
    async def init_chat_history(self, session_id):
        if not await self.test_if_table_exists(self.chat_history_table_name):
            async with await self.connect() as conn:
                PostgresChatMessageHistory(self.chat_history_table_name, session_id, async_connection=conn)

    async def insert_chat_messages(self, session_id, messages):
        async with await self.connect() as conn:
            chat_history = PostgresChatMessageHistory(self.chat_history_table_name, session_id, async_connection=conn)
            await chat_history.aclear()
            await chat_history.aadd_messages(messages)

    async def get_chat_messages(self, session_id):
        async with await self.connect() as conn:
            chat_history = PostgresChatMessageHistory(self.chat_history_table_name, session_id, async_connection=conn)
            return await chat_history.aget_messages()
    
    async def print_chat_history_schema(self):
        async with await self.connect() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(f"SELECT * FROM {self.chat_history_table_name} LIMIT 0")
                print(cursor.description)
    
    async def create_user(self):
        user_uuid = str(uuid.uuid4())
        async with await self.connect() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO users (uuid)
                    VALUES (%s)
                """, (user_uuid,))
                await conn.commit()
        return user_uuid

    async def get_user(self, user_uuid):
        async with await self.connect() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT * FROM users WHERE uuid = %s", (user_uuid,))
                return await cursor.fetchone()  
            
    async def test_if_user_exists(self, user_uuid: str):
        async with await self.connect() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT EXISTS (SELECT 1 FROM users WHERE uuid = %s)", (user_uuid,))
                return (await cursor.fetchone())[0]
            
    async def delete_user(self, user_uuid):
        async with await self.connect() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("DELETE FROM users WHERE uuid = %s", (user_uuid,))
                await conn.commit()
    
    async def add_session(self, user_uuid, session_id):
        async with await self.connect() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO sessions_users (user_uuid, session_id)
                    VALUES (%s, %s)
                """, (user_uuid, session_id))
                await conn.commit()
                
    async def get_sessions(self, user_uuid):
        async with await self.connect() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT session_id FROM sessions_users WHERE user_uuid = %s", (user_uuid,))
                return await cursor.fetchall()