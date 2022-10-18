import psycopg


class Client:
    def __init__(self, host, port, user, password, dbname):
        self.connection = psycopg.connect(
            host=host, port=port, user=user, password=password, dbname=dbname
        )
        self.cursor = self.connection.cursor()

    def query(self, query, params=None):
        self.cursor.execute(query, params)
        self.connection.commit()
        return self.cursor.fetchall()
