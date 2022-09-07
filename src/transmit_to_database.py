import psycopg2

dbname = "dr_trash"
user = "application"
password = "password"
host = "seheon.email"
port = "5432"

schema_name = 'application'
table_name = {
    'trash': f'{schema_name}.trash',
    'trashcan': f'{schema_name}.trashcan',
}


def connect():
    global client
    global cursor

    client = psycopg2.connect(dbname=dbname, user=user,
                              password=password, host=host, port=port)
    cursor = client.cursor()
    print(f"Connected to database {dbname} on {host}:{port} as {user}")


def query(query_string: str, query_args: tuple = None):
    cursor.execute(query_string, query_args)
    return cursor.fetchall()


def close():
    cursor.close()
    client.close()
    print(f"Closed connection to database {dbname} on {host}:{port} as {user}")


def main():
    connect()
    all_trash = query(f"SELECT * FROM {table_name['trash']};")
    print(all_trash)
    close()


def trash(trashcan_id: int, type: str, ok: bool):
    query(
        f"INSERT INTO {table_name['trash']} (\"trashcanId\", \"type\", \"ok\") VALUES (%s, '%s', %s)", (trashcan_id, type, ok))


if __name__ == '__main__':
    main()
else:
    connect()
