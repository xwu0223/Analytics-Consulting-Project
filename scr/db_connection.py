def mysql_conn(host, database,user):
    import mysql.connector
    from mysql.connector import Error
    pw = 'Jose@1992'
    try:
        connection = mysql.connector.connect(host=host,
                                            database=database,
                                            user=user,
                                            password=pw)
        if connection.is_connected():
            db_Info = connection.get_server_info()
            print("Connected to MySQL Server version ", db_Info)
            cursor = connection.cursor()
            cursor.execute("select database();")
            record = cursor.fetchone()
            print("You're connected to database: ", record)

    except Error as e:
        print("Error while connecting to MySQL", e)
    return connection