import asyncpg
import logging


class DatabaseUtility:
    def __init__(self, host, port, db, user, password, schema):
        self.db_config = {
            "host": host,
            "port": port,
            "database": db,
            "user": user,
            "password": password,
        }
        self.schema = schema
        self.pool = None

    async def connect(self):
        """Create a connection pool to PostgreSQL."""
        if not self.pool:
            try:
                self.pool = await asyncpg.create_pool(**self.db_config)
                async with self.pool.acquire() as connection:
                    await connection.execute(f"SET search_path TO {self.schema}")
                logging.info(
                    f"Connected to PostgreSQL with schema '{self.schema}'.")
            except Exception as e:
                logging.error(f"Failed to connect to PostgreSQL: {e}")
                raise

    async def close(self):
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            logging.info("PostgreSQL connection pool closed.")

    async def fetch_data(self, query, *args):
        """
        Fetch data from PostgreSQL.
        :param query: SQL query to execute.
        :param args: Parameters for the query.
        :return: List of rows fetched from the database.
        """
        await self.connect()
        try:
            async with self.pool.acquire() as connection:
                result = await connection.fetch(query, *args)
                return [dict(record) for record in result]
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            raise

    async def post_data(self, query, *args):
        """
        Post data to PostgreSQL.
        :param query: SQL query to execute.
        :param args: Parameters for the query.
        :return: Number of rows affected.
        """
        await self.connect()
        try:
            async with self.pool.acquire() as connection:
                result = await connection.execute(query, *args)
                return result
        except Exception as e:
            logging.error(f"Error posting data: {e}")
            raise
